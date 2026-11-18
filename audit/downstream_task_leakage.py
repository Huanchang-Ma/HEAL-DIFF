from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - lightweight test environments may not ship numpy
    np = None

from .metrics import (
    aggregate_patient_scores,
    bootstrap_auc_ci,
    roc_auc_binary,
    tpr_at_fpr_from_neg_quantiles,
)


ORDERED_SUBGROUP_FIELDS = {
    "history_len_bin",
    "label_size_bin",
    "label_rarity_bin",
    "label_pattern_bin",
}


def _ensure_mmhealth_import(mmhealth_root: Path) -> None:
    mm_src = mmhealth_root / "src"
    if str(mm_src) not in sys.path:
        sys.path.insert(0, str(mm_src))


def _task_metrics(task: str) -> Tuple[str, List[str]]:
    task = task.upper()
    if task in {"PHE", "REC"}:
        return "jaccard_samples", ["jaccard_samples", "f1_samples", "pr_auc_samples", "roc_auc_samples", "precision_samples", "recall_samples"]
    if task == "DIAG":
        return "roc_auc_samples", ["topk_acc", "topk_precision", "pr_auc_samples", "roc_auc_samples"]
    if task == "LOS":
        return "accuracy", ["roc_auc_weighted_ovr", "accuracy", "cohen_kappa", "f1_weighted", "mae", "rmse"]
    if task in {"MOR", "REA"}:
        return "pr_auc", ["roc_auc", "pr_auc", "precision", "recall"]
    raise ValueError(f"Unsupported task: {task}")


def _is_best(best_score: float, score: float, monitor_criterion: str) -> bool:
    if monitor_criterion == "max":
        return float(score) > float(best_score)
    if monitor_criterion == "min":
        return float(score) < float(best_score)
    raise ValueError(f"Unsupported monitor criterion: {monitor_criterion}")


def _resolve_ckpt(
    mmhealth_root: Path,
    task: str,
    dataset: str,
    model: str,
    exp_num: str,
    ckpt: str | None,
    allow_fallback: bool = True,
) -> Path:
    if ckpt:
        p = Path(ckpt)
        if p.exists():
            return p
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    default = mmhealth_root / "log" / "ckpt" / task / f"{dataset}-{model}-{exp_num}" / "best.ckpt"
    if default.exists():
        return default

    # fallback: pick newest matching checkpoint
    parent = mmhealth_root / "log" / "ckpt" / task
    if not parent.exists():
        raise FileNotFoundError(f"No checkpoint directory: {parent}")
    if not allow_fallback:
        raise FileNotFoundError(f"Checkpoint not found for exact experiment: {default}")
    cands = sorted(parent.glob(f"{dataset}-{model}-*/best.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No best.ckpt found under {parent} for {dataset}-{model}-*")
    return cands[0]


def _empty_visit_list_like(value: object) -> List[List[str]]:
    if isinstance(value, list):
        return [[] for _ in value]
    return []


def _optimizer_grouped_parameters(
    model: object,
    weight_decay: float,
) -> List[Dict[str, object]]:
    param = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    return [
        {
            "params": [p for n, p in param if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in param if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


def _sanitize_config_tree(value: object) -> object:
    if isinstance(value, Mapping):
        out: Dict[object, object] = {}
        for key, inner in value.items():
            if isinstance(inner, classmethod):
                continue
            if callable(inner):
                continue
            out[key] = _sanitize_config_tree(inner)
        return out
    if isinstance(value, list):
        return [_sanitize_config_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_config_tree(v) for v in value)
    return value


def _replace_multihead_attention_for_dp(module: object) -> int:
    import torch
    import torch.nn as nn
    from opacus.layers import DPMultiheadAttention

    class BoolMaskDPMultiheadAttention(DPMultiheadAttention):
        def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            is_causal=False,
        ):
            if key_padding_mask is not None and key_padding_mask.dtype is not torch.bool:
                key_padding_mask = key_padding_mask != 0
            if attn_mask is not None and attn_mask.dtype is not torch.bool:
                attn_mask = attn_mask != 0
            return super().forward(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            dp_attn = BoolMaskDPMultiheadAttention(
                embed_dim=child.embed_dim,
                num_heads=child.num_heads,
                dropout=child.dropout,
                bias=child.in_proj_bias is not None,
                add_bias_kv=child.bias_k is not None,
                add_zero_attn=child.add_zero_attn,
                kdim=child.kdim,
                vdim=child.vdim,
                batch_first=child.batch_first,
            )
            dp_attn.load_state_dict(child.state_dict())
            setattr(module, name, dp_attn)
            replaced += 1
        else:
            replaced += _replace_multihead_attention_for_dp(child)
    return replaced


def _freeze_unused_parameters_for_dp(module: object) -> List[str]:
    frozen: List[str] = []
    dataset_name = ""
    if hasattr(module, "config") and isinstance(getattr(module, "config"), Mapping):
        dataset_name = str(getattr(module, "config").get("DATASET", ""))

    def freeze_named_child(child_name: str) -> None:
        child = getattr(module, child_name, None)
        if child is None:
            return
        for param_name, param in child.named_parameters(recurse=True):
            if getattr(param, "requires_grad", False):
                param.requires_grad = False
                frozen.append(f"{child_name}.{param_name}")

    # These branches are instantiated on MIII/PHE but not exercised in this audit path.
    freeze_named_child("incomplete_emb")
    freeze_named_child("positional_enc")
    if dataset_name != "MIV-Note":
        freeze_named_child("note_linear")
    return frozen


def _ordered_bin(rank: int, label: str) -> str:
    return f"{int(rank):02d}:{label}"


def _to_code_list(value: object) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    if value is None:
        return []
    return [str(value)]


def _extract_label_codes(record: Mapping[str, object]) -> List[str]:
    if "labels" in record:
        return _to_code_list(record.get("labels"))
    if "label" in record:
        return _to_code_list(record.get("label"))
    return []


def _history_len(record: Mapping[str, object]) -> int:
    visit_hist = record.get("visit_id_hist")
    if isinstance(visit_hist, list):
        return len(visit_hist)
    conditions = record.get("conditions")
    if isinstance(conditions, list):
        return len(conditions)
    return 0


def _build_label_stats(records: Sequence[Mapping[str, object]]) -> Dict[str, Dict[object, int]]:
    doc_freq: Dict[str, int] = defaultdict(int)
    pattern_freq: Dict[Tuple[str, ...], int] = defaultdict(int)
    for record in records:
        codes = sorted(set(_extract_label_codes(record)))
        if not codes:
            continue
        for code in codes:
            doc_freq[code] += 1
        pattern_freq[tuple(codes)] += 1
    return {
        "doc_freq": dict(doc_freq),
        "pattern_freq": dict(pattern_freq),
    }


def _extract_record_subgroups(
    record: Mapping[str, object],
    label_stats: Mapping[str, Mapping[object, int]],
) -> Dict[str, str]:
    groups: Dict[str, str] = {}

    history_len = _history_len(record)
    if history_len <= 1:
        groups["history_len_bin"] = _ordered_bin(1, "hist_1")
    elif history_len <= 3:
        groups["history_len_bin"] = _ordered_bin(2, "hist_2to3")
    elif history_len <= 7:
        groups["history_len_bin"] = _ordered_bin(3, "hist_4to7")
    else:
        groups["history_len_bin"] = _ordered_bin(4, "hist_8plus")

    labels = sorted(set(_extract_label_codes(record)))
    label_n = len(labels)
    if label_n <= 1:
        groups["label_size_bin"] = _ordered_bin(1, "label_1")
    elif label_n <= 3:
        groups["label_size_bin"] = _ordered_bin(2, "label_2to3")
    elif label_n <= 5:
        groups["label_size_bin"] = _ordered_bin(3, "label_4to5")
    else:
        groups["label_size_bin"] = _ordered_bin(4, "label_6plus")

    if not labels:
        groups["label_rarity_bin"] = "missing"
        groups["label_pattern_bin"] = "missing"
        return groups

    doc_freq = label_stats.get("doc_freq", {})
    pattern_freq = label_stats.get("pattern_freq", {})
    min_df = min(int(doc_freq.get(code, 0)) for code in labels) if doc_freq else 0
    pattern_count = int(pattern_freq.get(tuple(labels), 0)) if pattern_freq else 0

    if min_df <= 2:
        groups["label_rarity_bin"] = _ordered_bin(3, "rare_df_1to2")
    elif min_df <= 5:
        groups["label_rarity_bin"] = _ordered_bin(2, "rare_df_3to5")
    else:
        groups["label_rarity_bin"] = _ordered_bin(1, "rare_df_6plus")

    if pattern_count <= 1:
        groups["label_pattern_bin"] = _ordered_bin(3, "pattern_singleton")
    elif pattern_count <= 3:
        groups["label_pattern_bin"] = _ordered_bin(2, "pattern_freq_2to3")
    else:
        groups["label_pattern_bin"] = _ordered_bin(1, "pattern_freq_4plus")
    return groups


def _aggregate_patient_groups(records: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, str]]:
    grouped: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        patient_id = str(record.get("patient_id", "")).strip()
        if not patient_id:
            continue
        subgroups = record.get("subgroups", {})
        if not isinstance(subgroups, Mapping):
            continue
        for field, value in subgroups.items():
            grouped[str(field)][patient_id].append(str(value))

    out: Dict[str, Dict[str, str]] = {}
    for field, per_patient in grouped.items():
        out[field] = {}
        for patient_id, values in per_patient.items():
            uniq = sorted(set(str(v) for v in values))
            if len(uniq) == 1:
                out[field][patient_id] = uniq[0]
            elif field in ORDERED_SUBGROUP_FIELDS:
                out[field][patient_id] = max(uniq)
            else:
                out[field][patient_id] = "mixed"
    return out


def _patient_scores_from_losses(
    losses: Sequence[float],
    patient_ids: Sequence[str],
    agg: str,
) -> Dict[str, float]:
    per_patient: MutableMapping[str, List[float]] = defaultdict(list)
    for patient_id, loss in zip(patient_ids, losses):
        per_patient[str(patient_id)].append(float(-loss))
    return {
        patient_id: float(aggregate_patient_scores(scores, agg))
        for patient_id, scores in per_patient.items()
        if scores
    }


def _subgroup_patient_mia(
    member_patient_scores: Mapping[str, float],
    nonmember_patient_scores: Mapping[str, float],
    member_groups: Mapping[str, Mapping[str, str]],
    nonmember_groups: Mapping[str, Mapping[str, str]],
    subgroup_fields: Sequence[str],
    fpr_targets: Sequence[float],
    bootstrap_iters: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
    min_patients_per_split: int,
) -> Dict[str, List[Dict[str, object]]]:
    result: Dict[str, List[Dict[str, object]]] = {}
    for field in subgroup_fields:
        member_values = member_groups.get(field, {})
        nonmember_values = nonmember_groups.get(field, {})
        values = sorted(set(member_values.values()) | set(nonmember_values.values()))
        rows: List[Dict[str, object]] = []
        for value in values:
            member_ids = [pid for pid, group in member_values.items() if group == value and pid in member_patient_scores]
            nonmember_ids = [pid for pid, group in nonmember_values.items() if group == value and pid in nonmember_patient_scores]
            if len(member_ids) < int(min_patients_per_split) or len(nonmember_ids) < int(min_patients_per_split):
                continue
            member_scores = [float(member_patient_scores[pid]) for pid in member_ids]
            nonmember_scores = [float(nonmember_patient_scores[pid]) for pid in nonmember_ids]
            labels = [1] * len(member_scores) + [0] * len(nonmember_scores)
            scores = member_scores + nonmember_scores
            auc = float(roc_auc_binary(labels, scores))
            rows.append(
                {
                    "value": str(value),
                    "auc": auc,
                    "auc_ci": bootstrap_auc_ci(
                        labels,
                        scores,
                        n_bootstrap=int(bootstrap_iters),
                        confidence=float(bootstrap_confidence),
                        seed=int(bootstrap_seed),
                    ),
                    "tpr_at_fpr": tpr_at_fpr_from_neg_quantiles(
                        member_scores=member_scores,
                        nonmember_scores=nonmember_scores,
                        fpr_targets=fpr_targets,
                    ),
                    "n_member_patients": len(member_scores),
                    "n_nonmember_patients": len(nonmember_scores),
                }
            )
        result[field] = rows
    return result


def _rewrite_record_for_feature_source(record: Mapping[str, object], feature_source: str) -> Dict[str, object]:
    out = dict(record)
    feature_source = str(feature_source).lower()
    if feature_source == "mixed":
        return out
    if feature_source == "synthetic_only":
        out["conditions"] = list(record.get("new_conditions", _empty_visit_list_like(record.get("conditions"))))
        out["procedures"] = list(record.get("new_procedures", _empty_visit_list_like(record.get("procedures"))))
        out["drugs_hist"] = list(record.get("new_drugs", _empty_visit_list_like(record.get("drugs_hist"))))
        out["new_conditions"] = _empty_visit_list_like(out.get("conditions"))
        out["new_procedures"] = _empty_visit_list_like(out.get("procedures"))
        out["new_drugs"] = _empty_visit_list_like(out.get("drugs_hist"))
        return out
    if feature_source == "raw_only":
        out["new_conditions"] = _empty_visit_list_like(record.get("new_conditions"))
        out["new_procedures"] = _empty_visit_list_like(record.get("new_procedures"))
        out["new_drugs"] = _empty_visit_list_like(record.get("new_drugs"))
        return out
    raise ValueError(f"Unsupported feature source: {feature_source}")


def _rewrite_records_for_feature_source(
    records: Sequence[Mapping[str, object]],
    feature_source: str,
) -> List[Dict[str, object]]:
    return [_rewrite_record_for_feature_source(record, feature_source) for record in records]


def _per_sample_loss(mode: str, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    if np is None:
        raise ModuleNotFoundError("numpy is required for downstream_task_leakage")
    eps = 1e-8
    mode = mode.lower()
    if mode == "multilabel":
        yt = y_true.astype(np.float64)
        yp = np.clip(y_prob.astype(np.float64), eps, 1.0 - eps)
        bce = -(yt * np.log(yp) + (1.0 - yt) * np.log(1.0 - yp))
        return bce.mean(axis=1)
    if mode == "binary":
        yt = y_true.astype(np.float64).reshape(-1)
        yp = np.clip(y_prob.astype(np.float64).reshape(-1), eps, 1.0 - eps)
        return -(yt * np.log(yp) + (1.0 - yt) * np.log(1.0 - yp))
    if mode == "multiclass":
        yp = np.clip(y_prob.astype(np.float64), eps, 1.0)
        if y_true.ndim == 2:
            yi = np.argmax(y_true, axis=1).astype(np.int64)
        else:
            yi = y_true.astype(np.int64).reshape(-1)
        rows = np.arange(yp.shape[0])
        return -np.log(yp[rows, yi])
    raise ValueError(f"Unsupported mode: {mode}")


def _patient_mia_from_losses(
    member_losses: Sequence[float],
    member_ids: Sequence[str],
    nonmember_losses: Sequence[float],
    nonmember_ids: Sequence[str],
    agg: str,
    fpr_targets: Sequence[float],
    bootstrap_iters: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> Dict[str, object]:
    member_score_map = _patient_scores_from_losses(member_losses, member_ids, agg)
    nonmember_score_map = _patient_scores_from_losses(nonmember_losses, nonmember_ids, agg)
    m_scores = [float(x) for x in member_score_map.values()]
    nm_scores = [float(x) for x in nonmember_score_map.values()]
    labels = [1] * len(m_scores) + [0] * len(nm_scores)
    scores = [float(x) for x in m_scores + nm_scores]
    auc = float(roc_auc_binary(labels, scores))
    auc_ci = bootstrap_auc_ci(
        labels,
        scores,
        n_bootstrap=int(bootstrap_iters),
        confidence=float(bootstrap_confidence),
        seed=int(bootstrap_seed),
    )
    tpr = tpr_at_fpr_from_neg_quantiles(
        member_scores=[float(x) for x in m_scores],
        nonmember_scores=[float(x) for x in nm_scores],
        fpr_targets=fpr_targets,
    )
    return {
        "auc": auc,
        "auc_ci": auc_ci,
        "tpr_at_fpr": tpr,
        "n_member_patients": len(m_scores),
        "n_nonmember_patients": len(nm_scores),
        "member_scores": member_score_map,
        "nonmember_scores": nonmember_score_map,
    }


def _iter_codes(value: object) -> List[str]:
    out: List[str] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_iter_codes(item))
        return out
    if value is None:
        return out
    token = str(value).strip()
    if token:
        out.append(token)
    return out


def _build_audit_mlp_arrays(
    train_records: Sequence[Mapping[str, object]],
    test_records: Sequence[Mapping[str, object]],
    feature_keys: Sequence[str],
) -> Dict[str, object]:
    if np is None:
        raise ModuleNotFoundError("numpy is required for downstream_task_leakage")

    feature_vocab: Dict[str, int] = {}
    label_vocab: Dict[str, int] = {}
    all_records = list(train_records) + list(test_records)
    for record in all_records:
        for feature_key in feature_keys:
            for code in sorted(set(_iter_codes(record.get(feature_key, [])))):
                if code not in feature_vocab:
                    feature_vocab[code] = len(feature_vocab)
        for label in sorted(set(_extract_label_codes(record))):
            if label not in label_vocab:
                label_vocab[label] = len(label_vocab)

    def encode(records: Sequence[Mapping[str, object]]) -> Tuple["np.ndarray", "np.ndarray", List[str]]:
        x = np.zeros((len(records), len(feature_vocab)), dtype=np.float32)
        y = np.zeros((len(records), len(label_vocab)), dtype=np.float32)
        patient_ids: List[str] = []
        for row, record in enumerate(records):
            patient_ids.append(str(record.get("patient_id", "")))
            for feature_key in feature_keys:
                for code in set(_iter_codes(record.get(feature_key, []))):
                    col = feature_vocab.get(code)
                    if col is not None:
                        x[row, col] = 1.0
            for label in set(_extract_label_codes(record)):
                col = label_vocab.get(label)
                if col is not None:
                    y[row, col] = 1.0
        return x, y, patient_ids

    x_train, y_train, pid_train = encode(train_records)
    x_test, y_test, pid_test = encode(test_records)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "pid_train": pid_train,
        "x_test": x_test,
        "y_test": y_test,
        "pid_test": pid_test,
        "feature_dim": int(x_train.shape[1]),
        "label_dim": int(y_train.shape[1]),
        "feature_vocab_size": int(len(feature_vocab)),
        "label_vocab_size": int(len(label_vocab)),
    }


def _multilabel_utility_scores(y_true: "np.ndarray", y_prob: "np.ndarray", threshold: float = 0.5) -> Dict[str, float]:
    if np is None:
        raise ModuleNotFoundError("numpy is required for downstream_task_leakage")

    truth = y_true.astype(bool)
    pred = y_prob >= float(threshold)
    tp = np.logical_and(truth, pred).sum(axis=1).astype(np.float64)
    pred_n = pred.sum(axis=1).astype(np.float64)
    true_n = truth.sum(axis=1).astype(np.float64)
    union = np.logical_or(truth, pred).sum(axis=1).astype(np.float64)

    jaccard = np.divide(tp, union, out=np.ones_like(tp), where=union > 0)
    precision = np.divide(tp, pred_n, out=np.ones_like(tp), where=pred_n > 0)
    recall = np.divide(tp, true_n, out=np.ones_like(tp), where=true_n > 0)
    denom = pred_n + true_n
    f1 = np.divide(2.0 * tp, denom, out=np.ones_like(tp), where=denom > 0)
    return {
        "jaccard_samples": float(np.mean(jaccard)) if len(jaccard) else 0.0,
        "f1_samples": float(np.mean(f1)) if len(f1) else 0.0,
        "precision_samples": float(np.mean(precision)) if len(precision) else 0.0,
        "recall_samples": float(np.mean(recall)) if len(recall) else 0.0,
    }


def _run_audit_mlp(
    *,
    args: argparse.Namespace,
    task: str,
    dataset: str,
    mmhealth_root: Path,
    train_records: Sequence[Mapping[str, object]],
    test_records: Sequence[Mapping[str, object]],
    monitor: str,
    fpr_targets: Sequence[float],
) -> Dict[str, object]:
    if task not in {"PHE", "REC"}:
        raise ValueError("audit_mlp currently supports multilabel tasks only (PHE/REC).")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    try:
        from opacus import PrivacyEngine
        from opacus.utils.batch_memory_manager import BatchMemoryManager
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DP-SGD requested but opacus is not installed in the active environment."
        ) from exc

    class AuditMLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(x)

    encoded = _build_audit_mlp_arrays(
        train_records=train_records,
        test_records=test_records,
        feature_keys=[
            "conditions",
            "procedures",
            "drugs_hist",
            "new_conditions",
            "new_procedures",
            "new_drugs",
        ],
    )

    device = ("cuda:" + args.gpu) if torch.cuda.is_available() else "cpu"
    ckpt_path = mmhealth_root / "log" / "ckpt" / task / f"{dataset}-audit_mlp-{args.exp_num}" / "best.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    x_train = torch.tensor(encoded["x_train"], dtype=torch.float32)
    y_train = torch.tensor(encoded["y_train"], dtype=torch.float32)
    x_test = torch.tensor(encoded["x_test"], dtype=torch.float32)
    y_test = torch.tensor(encoded["y_test"], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=int(args.batch_size), shuffle=False, drop_last=False)

    model = AuditMLP(
        input_dim=int(encoded["feature_dim"]),
        output_dim=int(encoded["label_dim"]),
        hidden_dim=max(int(encoded["label_dim"]), 128),
        dropout=0.1,
    ).to(device)

    bce = nn.BCEWithLogitsLoss(reduction="none")

    def evaluate_model(loader: DataLoader) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", float]:
        model.eval()
        logits_all: List[torch.Tensor] = []
        labels_all: List[torch.Tensor] = []
        sample_losses: List[torch.Tensor] = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits)
                per_sample = bce(logits, yb).mean(dim=1)
                logits_all.append(probs.cpu())
                labels_all.append(yb.cpu())
                sample_losses.append(per_sample.cpu())
        y_prob = torch.cat(logits_all, dim=0).numpy() if logits_all else np.zeros((0, int(encoded["label_dim"])), dtype=np.float32)
        y_true = torch.cat(labels_all, dim=0).numpy() if labels_all else np.zeros((0, int(encoded["label_dim"])), dtype=np.float32)
        losses = torch.cat(sample_losses, dim=0).numpy() if sample_losses else np.zeros((0,), dtype=np.float32)
        return y_true, y_prob, losses, float(losses.mean()) if len(losses) else 0.0

    dp_training: Dict[str, object] = {}
    if ckpt_path.exists() and not args.train_if_missing:
        state_dict = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=bool(args.dp_secure_mode))
        private_model, private_optimizer, private_train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=float(args.dp_noise_multiplier),
            max_grad_norm=float(args.dp_max_grad_norm),
            poisson_sampling=False,
            grad_sample_mode=str(args.dp_grad_sample_mode),
        )
        best_score = -1.0 * float("inf")
        history: List[Dict[str, object]] = []
        for epoch in range(50):
            private_model.train()
            train_losses: List[float] = []
            with BatchMemoryManager(
                data_loader=private_train_loader,
                max_physical_batch_size=max(int(args.dp_physical_batch_size), 1),
                optimizer=private_optimizer,
            ) as memory_safe_loader:
                for xb, yb in memory_safe_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = private_model(xb)
                    loss = bce(logits, yb).mean()
                    loss.backward()
                    private_optimizer.step()
                    private_optimizer.zero_grad()
                    train_losses.append(float(loss.item()))

            y_true_val, y_prob_val, _, _ = evaluate_model(test_loader)
            val_scores = _multilabel_utility_scores(y_true_val, y_prob_val)
            score = float(val_scores.get(monitor, 0.0))
            epsilon = float(privacy_engine.get_epsilon(float(args.dp_delta)))
            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss_mean": float(mean(train_losses)) if train_losses else 0.0,
                    "val_monitor": score,
                    "epsilon": epsilon,
                }
            )
            if _is_best(best_score, score, "max"):
                best_score = score
                torch.save(model.state_dict(), str(ckpt_path))

        if ckpt_path.exists():
            state_dict = torch.load(str(ckpt_path), map_location=device)
            model.load_state_dict(state_dict, strict=False)
        dp_training = {
            "checkpoint": str(ckpt_path),
            "best_monitor": float(best_score),
            "epsilon": float(privacy_engine.get_epsilon(float(args.dp_delta))),
            "delta": float(args.dp_delta),
            "noise_multiplier": float(args.dp_noise_multiplier),
            "max_grad_norm": float(args.dp_max_grad_norm),
            "physical_batch_size": int(args.dp_physical_batch_size),
            "grad_sample_mode": str(args.dp_grad_sample_mode),
            "history": history,
        }

    y_true_tr, y_prob_tr, tr_losses, loss_tr = evaluate_model(train_loader)
    y_true_te, y_prob_te, te_losses, loss_te = evaluate_model(test_loader)

    train_scores = _multilabel_utility_scores(y_true_tr, y_prob_tr)
    test_scores = _multilabel_utility_scores(y_true_te, y_prob_te)
    train_scores["loss"] = float(loss_tr)
    test_scores["loss"] = float(loss_te)

    mia = _patient_mia_from_losses(
        member_losses=tr_losses.tolist(),
        member_ids=[str(x) for x in encoded["pid_train"]],
        nonmember_losses=te_losses.tolist(),
        nonmember_ids=[str(x) for x in encoded["pid_test"]],
        agg=args.agg,
        fpr_targets=fpr_targets,
        bootstrap_iters=int(args.bootstrap_iters),
        bootstrap_confidence=float(args.bootstrap_confidence),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    subgroup_results: Dict[str, List[Dict[str, object]]] = {}
    if args.subgroup_analysis:
        subgroup_fields = [field.strip() for field in str(args.subgroup_fields).split(",") if field.strip()]
        member_groups = _aggregate_patient_groups(train_records)
        nonmember_groups = _aggregate_patient_groups(test_records)
        subgroup_results = _subgroup_patient_mia(
            member_patient_scores=mia["member_scores"],
            nonmember_patient_scores=mia["nonmember_scores"],
            member_groups=member_groups,
            nonmember_groups=nonmember_groups,
            subgroup_fields=subgroup_fields,
            fpr_targets=fpr_targets,
            bootstrap_iters=int(args.bootstrap_iters),
            bootstrap_confidence=float(args.bootstrap_confidence),
            bootstrap_seed=int(args.bootstrap_seed),
            min_patients_per_split=int(args.min_subgroup_patients),
        )

    utility_gap = {
        "monitor": monitor,
        "monitor_train_minus_test": float(train_scores.get(monitor, 0.0) - test_scores.get(monitor, 0.0)),
        "loss_train_minus_test": float(train_scores.get("loss", 0.0) - test_scores.get("loss", 0.0)),
    }

    result = {
        "meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mmhealth_root": str(mmhealth_root),
            "task": task,
            "dataset": dataset,
            "model": "audit_mlp",
            "mode": "multilabel",
            "checkpoint": str(ckpt_path),
            "data_source": "audit_vectorized_records",
            "feature_source": str(args.feature_source),
            "dp_sgd": True,
            "aggregation": args.agg,
            "fpr_targets": fpr_targets,
            "bootstrap_iters": int(args.bootstrap_iters),
            "bootstrap_confidence": float(args.bootstrap_confidence),
            "bootstrap_seed": int(args.bootstrap_seed),
            "feature_vocab_size": int(encoded["feature_vocab_size"]),
            "label_vocab_size": int(encoded["label_vocab_size"]),
        },
        "counts": {
            "n_train_samples": int(len(encoded["pid_train"])),
            "n_test_samples": int(len(encoded["pid_test"])),
            "n_train_patients": int(len(set(str(x) for x in encoded["pid_train"]))),
            "n_test_patients": int(len(set(str(x) for x in encoded["pid_test"]))),
        },
        "downstream_utility": {
            "train": train_scores,
            "test": test_scores,
            "gap": utility_gap,
        },
        "downstream_patient_mia": mia,
        "subgroup_patient_mia": subgroup_results,
        "privacy_training": dp_training,
        "loss_summary": {
            "train_mean": float(mean(tr_losses.tolist())) if len(tr_losses) else 0.0,
            "test_mean": float(mean(te_losses.tolist())) if len(te_losses) else 0.0,
            "train_p50": float(np.median(tr_losses)) if len(tr_losses) else 0.0,
            "test_p50": float(np.median(te_losses)) if len(te_losses) else 0.0,
        },
    }
    result["downstream_patient_mia"].pop("member_scores", None)
    result["downstream_patient_mia"].pop("nonmember_scores", None)
    return result


def _train_dp_model(
    *,
    model: object,
    trainer_cls: object,
    train_dataloader: object,
    val_dataloader: object,
    metrics: Sequence[str],
    monitor: str,
    monitor_criterion: str,
    device: str,
    exp_path: Path,
    epochs: int,
    optimizer_params: Mapping[str, object],
    weight_decay: float,
    max_grad_norm: Optional[float],
    dp_noise_multiplier: float,
    dp_max_grad_norm: float,
    dp_delta: float,
    dp_secure_mode: bool,
    dp_physical_batch_size: int,
    dp_grad_sample_mode: str,
) -> Tuple[Dict[str, object], object]:
    import torch
    import torch.nn as nn

    try:
        from opacus import PrivacyEngine
        from opacus.utils.batch_memory_manager import BatchMemoryManager
        from opacus.validators import ModuleValidator
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised on server
        raise ModuleNotFoundError(
            "DP-SGD requested but opacus is not installed in the active environment."
        ) from exc

    try:
        from utils import adjust_learning_rate  # type: ignore
    except ModuleNotFoundError:
        adjust_learning_rate = None

    class LossOnlyModule(nn.Module):
        def __init__(self, inner_model: nn.Module):
            super().__init__()
            self.model = inner_model

        def forward(self, **data):  # type: ignore[override]
            return self.model(return_dict=False, **data)

    exp_path.mkdir(parents=True, exist_ok=True)
    replaced_multihead = _replace_multihead_attention_for_dp(model)
    frozen_params = _freeze_unused_parameters_for_dp(model)
    validator_errors = ModuleValidator.validate(model, strict=False)
    if validator_errors:
        raise RuntimeError(f"Unsupported modules remain after DP replacements: {validator_errors}")
    model = model.to(device)
    train_wrapper = LossOnlyModule(model).to(device)
    optimizer = torch.optim.AdamW(
        _optimizer_grouped_parameters(train_wrapper, weight_decay),
        **dict(optimizer_params),
    )

    privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=bool(dp_secure_mode))
    make_private_kwargs = {
        "module": train_wrapper,
        "optimizer": optimizer,
        "data_loader": train_dataloader,
        "noise_multiplier": float(dp_noise_multiplier),
        "max_grad_norm": float(dp_max_grad_norm),
        "poisson_sampling": False,
    }
    if str(dp_grad_sample_mode).strip():
        try:
            private_model, private_optimizer, private_train_loader = privacy_engine.make_private(
                grad_sample_mode=str(dp_grad_sample_mode),
                **make_private_kwargs,
            )
        except TypeError:
            private_model, private_optimizer, private_train_loader = privacy_engine.make_private(
                **make_private_kwargs,
            )
    else:
        private_model, private_optimizer, private_train_loader = privacy_engine.make_private(
            **make_private_kwargs,
        )

    wrapped_module = getattr(private_model, "_module", private_model)
    eval_model = wrapped_module.model
    eval_trainer = trainer_cls(
        model=eval_model,
        metrics=list(metrics),
        device=device,
        enable_logging=False,
    )

    best_score = -1.0 * float("inf") if monitor_criterion == "max" else float("inf")
    best_ckpt = exp_path / "best.ckpt"
    history: List[Dict[str, object]] = []
    optimizer_lr = float(dict(optimizer_params).get("lr", 0.0))
    steps_per_epoch = len(train_dataloader)
    for epoch in range(int(epochs)):
        private_model.train()
        train_losses: List[float] = []
        with BatchMemoryManager(
            data_loader=private_train_loader,
            max_physical_batch_size=max(int(dp_physical_batch_size), 1),
            optimizer=private_optimizer,
        ) as memory_safe_loader:
            for step, data in enumerate(memory_safe_loader, start=1):
                if adjust_learning_rate is not None and optimizer_lr > 0:
                    adjust_learning_rate(private_optimizer, step, {"LR": optimizer_lr}, max(steps_per_epoch, 1))
                loss = private_model(**data)
                loss.backward()
                missing_grad_sample = [
                    name
                    for name, param in eval_model.named_parameters()
                    if getattr(param, "requires_grad", False) and getattr(param, "grad_sample", None) is None
                ]
                if missing_grad_sample:
                    raise RuntimeError(
                        "Missing grad_sample for parameters: "
                        + ", ".join(missing_grad_sample[:20])
                    )
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(eval_model.parameters(), float(max_grad_norm))
                private_optimizer.step()
                private_optimizer.zero_grad()
                train_losses.append(float(loss.item()))

        scores = eval_trainer.evaluate(val_dataloader, aux_data={})
        epoch_epsilon = float(privacy_engine.get_epsilon(float(dp_delta)))
        history.append(
            {
                "epoch": int(epoch),
                "train_loss_mean": float(mean(train_losses)) if train_losses else 0.0,
                "val_monitor": float(scores.get(monitor, 0.0)),
                "epsilon": epoch_epsilon,
            }
        )
        score = float(scores.get(monitor, 0.0))
        if _is_best(best_score, score, monitor_criterion):
            best_score = score
            torch.save(eval_model.state_dict(), str(best_ckpt))

    if best_ckpt.exists():
        state_dict = torch.load(str(best_ckpt), map_location=device)
        eval_model.load_state_dict(state_dict, strict=False)

    return (
        {
            "checkpoint": str(best_ckpt),
            "best_monitor": float(best_score),
            "epsilon": float(privacy_engine.get_epsilon(float(dp_delta))),
            "delta": float(dp_delta),
            "noise_multiplier": float(dp_noise_multiplier),
            "max_grad_norm": float(dp_max_grad_norm),
            "physical_batch_size": int(dp_physical_batch_size),
            "grad_sample_mode": str(dp_grad_sample_mode),
            "module_validator_errors_after_replacement": len(validator_errors),
            "n_replaced_multihead_attention": int(replaced_multihead),
            "n_frozen_unused_parameters": len(frozen_params),
            "frozen_unused_parameters": frozen_params,
            "history": history,
        },
        eval_model,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Downstream task leakage audit (utility gap + patient-level MIA from downstream loss).")
    p.add_argument("--mmhealth-root", type=str, default="MMHealth")
    p.add_argument("--task", type=str, default="PHE")
    p.add_argument("--dataset", type=str, default="MIII")
    p.add_argument("--model", type=str, default="ours")
    p.add_argument("--exp-num", type=str, default="20")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--feature-source", type=str, choices=["mixed", "synthetic_only", "raw_only"], default="mixed")
    p.add_argument("--train-if-missing", action="store_true")
    p.add_argument("--dp-sgd", action="store_true")
    p.add_argument("--dp-noise-multiplier", type=float, default=1.0)
    p.add_argument("--dp-max-grad-norm", type=float, default=1.0)
    p.add_argument("--dp-delta", type=float, default=1e-5)
    p.add_argument("--dp-secure-mode", action="store_true")
    p.add_argument("--dp-physical-batch-size", type=int, default=8)
    p.add_argument("--dp-grad-sample-mode", type=str, default="functorch")
    p.add_argument("--agg", type=str, choices=["min", "mean", "median"], default="mean")
    p.add_argument("--fpr-targets", type=str, default="0.001,0.005,0.01,0.05")
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--bootstrap-confidence", type=float, default=0.95)
    p.add_argument("--bootstrap-seed", type=int, default=528)
    p.add_argument("--subgroup-analysis", action="store_true")
    p.add_argument(
        "--subgroup-fields",
        type=str,
        default="history_len_bin,label_size_bin,label_rarity_bin,label_pattern_bin",
    )
    p.add_argument("--min-subgroup-patients", type=int, default=30)
    p.add_argument("--output", type=str, default="outputs/downstream_task_leakage.json")
    args = p.parse_args()

    if np is None:
        raise ModuleNotFoundError("numpy is required for downstream_task_leakage")

    import torch

    mmhealth_root = Path(args.mmhealth_root).resolve()
    _ensure_mmhealth_import(mmhealth_root)

    from config import config as mm_config  # type: ignore
    from data import convert_dataset  # type: ignore
    from loader import get_dataloader, get_special_input  # type: ignore
    from train_hp import get_task_fn, underlying_model  # type: ignore
    from trainer import Trainer, get_metrics_fn  # type: ignore
    from utils import load_pickle  # type: ignore

    mm_config["TASK"] = args.task
    mm_config["DATASET"] = args.dataset
    mm_config["MODEL"] = args.model
    mm_config["GPU"] = args.gpu
    mm_config["BATCH"] = int(args.batch_size)
    mm_config["USE_CUDA"] = True

    task = str(args.task).upper()
    dataset = str(args.dataset)
    model_name = str(args.model)
    model_config = _sanitize_config_tree(mm_config)
    monitor, metric_list = _task_metrics(task)

    processed = mmhealth_root / "data" / task / dataset / "processed"
    train_samples_path = processed / "train_samples.pkl"
    test_samples_path = processed / "test_samples.pkl"
    train_dataset_path = processed / "train_dataset.pkl"
    test_dataset_path = processed / "test_dataset.pkl"
    prestand_path = processed / "datasets_pre_stand.pkl"
    if not prestand_path.exists():
        raise FileNotFoundError(f"Missing datasets_pre_stand.pkl: {prestand_path}")

    base_samples = load_pickle(str(prestand_path))
    sample_dataset = convert_dataset(base_samples, dataset_name=dataset, task_name=task, valid=False, all=True)

    if train_samples_path.exists() and test_samples_path.exists():
        train_records = load_pickle(str(train_samples_path))
        test_records = load_pickle(str(test_samples_path))
        data_source = "augmented_samples"
    else:
        if not train_dataset_path.exists() or not test_dataset_path.exists():
            raise FileNotFoundError(f"Missing train/test datasets under {processed}")
        train_records = load_pickle(str(train_dataset_path))
        test_records = load_pickle(str(test_dataset_path))
        data_source = "raw_split_dataset"

    train_records = _rewrite_records_for_feature_source(train_records, str(args.feature_source))
    test_records = _rewrite_records_for_feature_source(test_records, str(args.feature_source))
    if args.subgroup_analysis:
        label_stats = _build_label_stats(list(train_records) + list(test_records))
        train_records = [{**record, "subgroups": _extract_record_subgroups(record, label_stats)} for record in train_records]
        test_records = [{**record, "subgroups": _extract_record_subgroups(record, label_stats)} for record in test_records]
    fpr_targets = [float(x) for x in str(args.fpr_targets).split(",") if str(x).strip()]

    if model_name == "audit_mlp":
        result = _run_audit_mlp(
            args=args,
            task=task,
            dataset=dataset,
            mmhealth_root=mmhealth_root,
            train_records=train_records,
            test_records=test_records,
            monitor=monitor,
            fpr_targets=fpr_targets,
        )
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Downstream leakage report written: {out_path}")
        return

    train_dataset = convert_dataset(train_records, dataset_name=dataset, task_name=task, valid=False)
    test_dataset = convert_dataset(test_records, dataset_name=dataset, task_name=task, valid=False)

    collate_fn = get_special_input(mm_config)
    train_loader = get_dataloader(train_dataset, batch_size=mm_config["BATCH"], shuffle=False, drop_last=False, collate_fn=collate_fn)
    test_loader = get_dataloader(test_dataset, batch_size=mm_config["BATCH"], shuffle=False, drop_last=False, collate_fn=collate_fn)

    task_fn, mode = get_task_fn(mm_config)
    _ = task_fn  # keep interface parity with main pipeline
    ckpt_path: Path
    trainer = None
    dp_training: Dict[str, object] = {}
    allow_fallback = str(args.feature_source) == "mixed"
    try:
        ckpt_path = _resolve_ckpt(
            mmhealth_root,
            task,
            dataset,
            model_name,
            args.exp_num,
            args.ckpt,
            allow_fallback=allow_fallback,
        )
        model = underlying_model(
            model_config,
            sample_dataset,
            feature_keys=mm_config["FEATURE"],
            label_key=mm_config["LABEL"],
            mode=mode,
        )
        state_dict = torch.load(str(ckpt_path), map_location=("cuda:" + mm_config["GPU"]) if torch.cuda.is_available() else "cpu")
        model.load_state_dict(state_dict, strict=False)
        trainer = Trainer(
            model=model,
            metrics=metric_list,
            device=("cuda:" + mm_config["GPU"]) if torch.cuda.is_available() else "cpu",
            enable_logging=False,
        )
    except FileNotFoundError:
        if not args.train_if_missing:
            raise
        if str(args.feature_source) == "mixed":
            if args.dp_sgd:
                raise ValueError("DP-SGD branch is currently supported for explicit raw_only/synthetic_only retraining, not mixed fallback training.")
            import main_unify  # type: ignore

            main_unify.run_single_config(augment=True, retrain=False, exp_num=args.exp_num, pretrain=False)
            ckpt_path = _resolve_ckpt(
                mmhealth_root,
                task,
                dataset,
                model_name,
                args.exp_num,
                args.ckpt,
                allow_fallback=allow_fallback,
            )
            model = underlying_model(
                model_config,
                sample_dataset,
                feature_keys=mm_config["FEATURE"],
                label_key=mm_config["LABEL"],
                mode=mode,
            )
            state_dict = torch.load(str(ckpt_path), map_location=("cuda:" + mm_config["GPU"]) if torch.cuda.is_available() else "cpu")
            model.load_state_dict(state_dict, strict=False)
            trainer = Trainer(
                model=model,
                metrics=metric_list,
                device=("cuda:" + mm_config["GPU"]) if torch.cuda.is_available() else "cpu",
                enable_logging=False,
            )
        else:
            model = underlying_model(
                model_config,
                sample_dataset,
                feature_keys=mm_config["FEATURE"],
                label_key=mm_config["LABEL"],
                mode=mode,
            )
            if args.dp_sgd:
                exp_name = f"{dataset}-{model_name}-{args.exp_num}"
                exp_path = Path(mm_config["LOGDIR"] + task + "/") / exp_name
                dp_training, trained_model = _train_dp_model(
                    model=model,
                    trainer_cls=Trainer,
                    train_dataloader=train_loader,
                    val_dataloader=test_loader,
                    metrics=metric_list,
                    monitor=monitor,
                    monitor_criterion="max",
                    device=("cuda:" + mm_config["GPU"]) if torch.cuda.is_available() else "cpu",
                    exp_path=exp_path,
                    epochs=int(mm_config["EPOCH"]),
                    optimizer_params={"lr": mm_config["LR"]},
                    weight_decay=float(mm_config["WD"]),
                    max_grad_norm=0.1,
                    dp_noise_multiplier=float(args.dp_noise_multiplier),
                    dp_max_grad_norm=float(args.dp_max_grad_norm),
                    dp_delta=float(args.dp_delta),
                    dp_secure_mode=bool(args.dp_secure_mode),
                    dp_physical_batch_size=int(args.dp_physical_batch_size),
                    dp_grad_sample_mode=str(args.dp_grad_sample_mode),
                )
                trainer = Trainer(
                    model=trained_model,
                    metrics=metric_list,
                    device=("cuda:" + mm_config["GPU"]) if torch.cuda.is_available() else "cpu",
                    enable_logging=False,
                )
                ckpt_path = Path(str(dp_training["checkpoint"]))
            else:
                trainer = Trainer(
                    model=model,
                    metrics=metric_list,
                    device=("cuda:" + mm_config["GPU"]) if torch.cuda.is_available() else "cpu",
                    output_path=mm_config["LOGDIR"] + task + "/",
                    exp_name=f"{dataset}-{model_name}-{args.exp_num}",
                    enable_logging=True,
                )
                trainer.train(
                    train_dataloader=train_loader,
                    val_dataloader=test_loader,
                    epochs=mm_config["EPOCH"],
                    warm_epochs=mm_config["WARM_EPOCH"],
                    weight_decay=mm_config["WD"],
                    monitor=monitor,
                    optimizer_params={"lr": mm_config["LR"]},
                    max_grad_norm=0.1,
                    load_best_model_at_last=True,
                    aux_data={},
                )
                ckpt_path = Path(trainer.exp_path) / "best.ckpt"

    assert trainer is not None
    y_true_tr, y_prob_tr, loss_tr, pid_tr = trainer.inference(train_loader, additional_outputs=None, return_patient_ids=True)
    y_true_te, y_prob_te, loss_te, pid_te = trainer.inference(test_loader, additional_outputs=None, return_patient_ids=True)

    metrics_fn = get_metrics_fn(mode)
    train_scores = metrics_fn(y_true_tr, y_prob_tr, metrics=metric_list, aux_data={}, patient_ids=pid_tr)
    test_scores = metrics_fn(y_true_te, y_prob_te, metrics=metric_list, aux_data={}, patient_ids=pid_te)
    train_scores["loss"] = float(loss_tr)
    test_scores["loss"] = float(loss_te)

    tr_losses = _per_sample_loss(mode, y_true_tr, y_prob_tr)
    te_losses = _per_sample_loss(mode, y_true_te, y_prob_te)

    mia = _patient_mia_from_losses(
        member_losses=tr_losses.tolist(),
        member_ids=[str(x) for x in pid_tr],
        nonmember_losses=te_losses.tolist(),
        nonmember_ids=[str(x) for x in pid_te],
        agg=args.agg,
        fpr_targets=fpr_targets,
        bootstrap_iters=int(args.bootstrap_iters),
        bootstrap_confidence=float(args.bootstrap_confidence),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    subgroup_results: Dict[str, List[Dict[str, object]]] = {}
    if args.subgroup_analysis:
        subgroup_fields = [field.strip() for field in str(args.subgroup_fields).split(",") if field.strip()]
        member_groups = _aggregate_patient_groups(train_records)
        nonmember_groups = _aggregate_patient_groups(test_records)
        subgroup_results = _subgroup_patient_mia(
            member_patient_scores=mia["member_scores"],
            nonmember_patient_scores=mia["nonmember_scores"],
            member_groups=member_groups,
            nonmember_groups=nonmember_groups,
            subgroup_fields=subgroup_fields,
            fpr_targets=fpr_targets,
            bootstrap_iters=int(args.bootstrap_iters),
            bootstrap_confidence=float(args.bootstrap_confidence),
            bootstrap_seed=int(args.bootstrap_seed),
            min_patients_per_split=int(args.min_subgroup_patients),
        )

    utility_gap = {
        "monitor": monitor,
        "monitor_train_minus_test": float(train_scores.get(monitor, 0.0) - test_scores.get(monitor, 0.0)),
        "loss_train_minus_test": float(train_scores.get("loss", 0.0) - test_scores.get("loss", 0.0)),
    }

    result = {
        "meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mmhealth_root": str(mmhealth_root),
            "task": task,
            "dataset": dataset,
            "model": model_name,
            "mode": mode,
            "checkpoint": str(ckpt_path),
            "data_source": data_source,
            "feature_source": str(args.feature_source),
            "dp_sgd": bool(args.dp_sgd),
            "aggregation": args.agg,
            "fpr_targets": fpr_targets,
            "bootstrap_iters": int(args.bootstrap_iters),
            "bootstrap_confidence": float(args.bootstrap_confidence),
            "bootstrap_seed": int(args.bootstrap_seed),
        },
        "counts": {
            "n_train_samples": int(len(pid_tr)),
            "n_test_samples": int(len(pid_te)),
            "n_train_patients": int(len(set(str(x) for x in pid_tr))),
            "n_test_patients": int(len(set(str(x) for x in pid_te))),
        },
        "downstream_utility": {
            "train": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in train_scores.items()},
            "test": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in test_scores.items()},
            "gap": utility_gap,
        },
        "downstream_patient_mia": mia,
        "subgroup_patient_mia": subgroup_results,
        "privacy_training": dp_training,
        "loss_summary": {
            "train_mean": float(mean(tr_losses.tolist())) if len(tr_losses) else 0.0,
            "test_mean": float(mean(te_losses.tolist())) if len(te_losses) else 0.0,
            "train_p50": float(np.median(tr_losses)) if len(tr_losses) else 0.0,
            "test_p50": float(np.median(te_losses)) if len(te_losses) else 0.0,
        },
    }
    result["downstream_patient_mia"].pop("member_scores", None)
    result["downstream_patient_mia"].pop("nonmember_scores", None)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Downstream leakage report written: {out_path}")


if __name__ == "__main__":
    main()
