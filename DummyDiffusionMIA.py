# secmi_full.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


# ==========================================================
# 0) Seed
# ==========================================================
def set_seed(seed: int = 528) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================================================
# 1) Minimal fallback utils (in case your project utils not importable)
# ==========================================================
def fallback_get_last_visit(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: [B,V,D], mask: [B,V] (True valid)
    return last valid visit embedding [B,D]
    """
    B, V, D = x.shape
    # last index per batch (if all False -> use 0)
    idx = mask.long().sum(dim=1) - 1
    idx = torch.clamp(idx, min=0)
    out = x[torch.arange(B, device=x.device), idx, :]
    return out


def _to_tensor(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(np.asarray(x), device=device, dtype=dtype)


def _ensure_bcd(x: torch.Tensor) -> torch.Tensor:
    """
    normalize latent to [B,C,D]
    """
    if x.dim() == 2:   # [C,D]
        return x.unsqueeze(0)
    if x.dim() == 3:   # [B,C,D]
        return x
    raise ValueError(f"Expected latent dim=2/3, got {tuple(x.shape)}")


def _pad_or_trim_channels(x: torch.Tensor, in_chans: int) -> torch.Tensor:
    """
    x: [B,C,D] -> [B,in_chans,D]
    """
    B, C, D = x.shape
    if C == in_chans:
        return x
    if C < in_chans:
        pad = torch.zeros(B, in_chans - C, D, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)
    return x[:, :in_chans, :]


# ==========================================================
# 2) Diffusion schedule helpers
# ==========================================================
def get_alphas_cumprod_from_model(generated_model: nn.Module) -> torch.Tensor:
    """
    Try best to find alphas_cumprod / betas from your diffusion model wrapper.
    """
    # direct attributes
    for name in ("alphas_cumprod", "alpha_cumprod", "alpha_bars", "alpha_bar"):
        if hasattr(generated_model, name):
            v = getattr(generated_model, name)
            if torch.is_tensor(v):
                return v.detach()

    # diffusion submodule
    if hasattr(generated_model, "diffusion"):
        diff = generated_model.diffusion
        for name in ("alphas_cumprod", "alpha_cumprod", "alpha_bars", "alpha_bar"):
            if hasattr(diff, name):
                v = getattr(diff, name)
                if torch.is_tensor(v):
                    return v.detach()

    # betas -> compute
    betas = None
    for name in ("betas", "beta"):
        if hasattr(generated_model, name) and torch.is_tensor(getattr(generated_model, name)):
            betas = getattr(generated_model, name).detach()
            break
    if betas is None and hasattr(generated_model, "diffusion") and hasattr(generated_model.diffusion, "betas"):
        betas = generated_model.diffusion.betas.detach()

    if betas is None:
        raise AttributeError(
            "[SecMI] Cannot find alphas_cumprod or betas in generated_model. "
            "Please expose generated_model.alphas_cumprod or generated_model.diffusion.betas."
        )

    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def q_sample(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    x0: [B,C,D], t: [B] long, eps: [B,C,D]
    """
    a = alphas_cumprod.to(x0.device)[t].view(-1, 1, 1)
    return torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * eps


# ==========================================================
# 3) Model call adapter (auto)
# ==========================================================
def _extract_eps_from_output(out: Any) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for x in out:
            if torch.is_tensor(x):
                return x
        raise TypeError("[SecMI] model output tuple/list has no tensor")
    if isinstance(out, dict):
        for k in ("eps", "epsilon", "noise", "pred_eps", "pred_noise", "model_output", "out"):
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        # fallback: any tensor value
        for v in out.values():
            if torch.is_tensor(v):
                return v
        raise KeyError(f"[SecMI] model output dict has no tensor values. keys={list(out.keys())}")
    raise TypeError(f"[SecMI] unsupported model output type: {type(out)}")


@torch.no_grad()
def predict_eps_auto(generated_model: nn.Module, xt: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Try calling:
      1) model(xt,t,cond)
      2) model(xt,t)
      3) model.nnet(...)
      4) model.model(...)
    """
    candidates = []
    if hasattr(generated_model, "nnet"):
        candidates.append(getattr(generated_model, "nnet"))
    if hasattr(generated_model, "model"):
        candidates.append(getattr(generated_model, "model"))
    candidates.append(generated_model)

    last_err = None
    for fn in candidates:
        # (xt,t,cond)
        if cond is not None:
            try:
                out = fn(xt, t, cond)
                return _extract_eps_from_output(out)
            except Exception as e:
                last_err = e
        # (xt,t)
        try:
            out = fn(xt, t)
            return _extract_eps_from_output(out)
        except Exception as e:
            last_err = e
        # (xt)
        try:
            out = fn(xt)
            return _extract_eps_from_output(out)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"[SecMI] cannot call diffusion model to predict eps. last_err={repr(last_err)}")


# ==========================================================
# 4) ROC/AUC (no sklearn)
# ==========================================================
def roc_curve_np(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = y_true.astype(np.int32)
    y_score = y_score.astype(np.float64)

    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    P = max(1, int(y_true.sum()))
    N = max(1, int((1 - y_true).sum()))

    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)

    tpr = tps / P
    fpr = fps / N
    thr = y_score

    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thr = np.concatenate([[np.inf], thr])
    return fpr, tpr, thr


def auc_np(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.trapz(tpr, fpr))


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


# ==========================================================
# 5) X0 provider: (A) directly from sample["z"] OR (B) from raw EHR via your Diffrm.encode_patient
# ==========================================================
@dataclass
class MMHealthX0Provider:
    """
    If sample already contains 'z'/'x0'/'latent': use it directly.
    Else, build z by calling encoder.encode_patient(...) for each feature:
        z = stack([last_cond, last_proc, last_drug], dim=1) -> [B,3,dim]
    """
    encoder: Optional[Any]  # your Diffrm instance (or any object with encode_patient)
    config: Dict[str, Any]
    diff_config: Dict[str, Any]
    latent_keys: Sequence[str] = ("z", "x0", "latent", "latents")
    use_note_for_miv: bool = False  # 你 forward 里把 note 加在 conditions 上；默认不加（更稳）

    def __call__(self, sample: Any, device: torch.device, in_chans: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # ---------- 1) latent already provided ----------
        if isinstance(sample, dict):
            for k in self.latent_keys:
                if k in sample:
                    x0 = _to_tensor(sample[k], device=device)
                    x0 = _ensure_bcd(x0)
                    x0 = _pad_or_trim_channels(x0, in_chans)
                    cond = None
                    if "cond" in sample:
                        cond = _to_tensor(sample["cond"], device=device)
                    return x0, cond

        # ---------- 2) build latent from raw EHR ----------
        if self.encoder is None:
            raise RuntimeError(
                "[SecMI] sample does not contain latent (z/x0/latent), and MMHealthX0Provider.encoder is None.\n"
                "Please either: (1) make dataset return sample['z']  OR  (2) pass your Diffrm model as encoder."
            )

        if not isinstance(sample, dict):
            raise TypeError(f"[SecMI] raw EHR sample must be dict, got {type(sample)}")

        # expected keys (your forward signature)
        # patient_id, conditions, procedures, drugs_hist are needed to compute embeddings
        conditions = sample.get("conditions", None)
        procedures = sample.get("procedures", None)
        drugs_hist = sample.get("drugs_hist", sample.get("drugs", None))

        if conditions is None or procedures is None or drugs_hist is None:
            raise KeyError(
                "[SecMI] raw EHR sample missing keys. Need conditions/procedures/drugs_hist (or drugs). "
                f"keys={list(sample.keys())}"
            )

        # wrap into batch=1 for encode_patient()
        # encode_patient expects raw_values: List[List[List[str]]] with shape [B,V,M]
        B = 1
        cond_raw = [conditions]
        proc_raw = [procedures]
        drug_raw = [drugs_hist]

        # new_* must exist in kwargs for your encode_patient signature, but in encode_patient you allow new_feature=0
        # we pass new_feature=0
        _, _, cond_mask, cond_vis = self.encoder.encode_patient("conditions", cond_raw, 0)  # [B,V,D]
        _, _, proc_mask, proc_vis = self.encoder.encode_patient("procedures", proc_raw, 0)
        _, _, drug_mask, drug_vis = self.encoder.encode_patient("drugs", drug_raw, 0)

        # visit-level masks [B,V]
        cond_vmask = (torch.sum(cond_mask, dim=-1) != 0)
        proc_vmask = (torch.sum(proc_mask, dim=-1) != 0)
        drug_vmask = (torch.sum(drug_mask, dim=-1) != 0)

        # get_last_visit: try your project's utils.get_last_visit; fallback if not found
        get_last = None
        try:
            from utils import get_last_visit as project_get_last_visit  # type: ignore
            get_last = project_get_last_visit
        except Exception:
            get_last = fallback_get_last_visit

        cond_last = get_last(cond_vis, cond_vmask)  # [B,D]
        proc_last = get_last(proc_vis, proc_vmask)
        drug_last = get_last(drug_vis, drug_vmask)

        # optional: for MIV-Note you add note_emb to conditions embedding in forward
        if self.use_note_for_miv and self.config.get("DATASET", "") == "MIV-Note" and "note" in sample:
            # your forward uses self.note_linear(pad_list(note)) and adds to condition_vis_emb per visit;
            # implementing that exactly needs pad_list + note_linear; here we only do a simple last-visit add if possible
            try:
                note = sample["note"]
                # note may be [V,768] or list; try to tensorize and align to V
                note_t = _to_tensor(note, device=device)
                if note_t.dim() == 2:
                    # [V,768] -> project linear -> [V,D], pick last valid note
                    note_proj = self.encoder.note_linear(note_t).unsqueeze(0)  # [1,V,D]
                    note_vmask = (note_proj.sum(dim=-1) != 0)  # [1,V]
                    note_last = get_last(note_proj, note_vmask)  # [1,D]
                    cond_last = cond_last + note_last
            except Exception:
                pass

        # stack to [B,3,D]
        x0 = torch.stack([cond_last, proc_last, drug_last], dim=1)  # [1,3,D]
        x0 = _pad_or_trim_channels(x0, in_chans)
        return x0, None


# ==========================================================
# 6) SecMI-stat main
# ==========================================================
@torch.no_grad()
def run_secmi_mia_eval(
    generated_model: nn.Module,
    train_dataset: Any,
    test_dataset: Any,
    diff_config: Dict[str, Any],
    config: Dict[str, Any],
    x0_provider: Optional[MMHealthX0Provider] = None,
) -> Dict[str, Any]:
    """
    SecMI-stat:
      score = - MSE(eps_hat, eps)  (larger => more likely member)
    """
    seed = int(config.get("SECMI_SEED", config.get("SEED", 528)))
    set_seed(seed)

    generated_model.eval()
    device = next(generated_model.parameters()).device

    in_chans = int(diff_config.get("nnet", {}).get("in_chans", 4))
    num_queries = int(config.get("SECMI_NUM_QUERIES", 8))
    use_random_t = bool(config.get("SECMI_RANDOM_T", False))
    max_member = int(config.get("SECMI_N_MEMBER", 2000))
    max_nonmem = int(config.get("SECMI_N_NONMEMBER", 2000))
    t_fixed_ratio = float(config.get("SECMI_T_RATIO", 1.0))  # 1.0 means near T-1

    # schedule
    alphas_cumprod = get_alphas_cumprod_from_model(generated_model).to(device)
    T = int(alphas_cumprod.numel())
    t_fixed = int(max(1, min(T - 1, round((T - 1) * t_fixed_ratio))))

    # sample indices
    n_mem = min(max_member, len(train_dataset))
    n_non = min(max_nonmem, len(test_dataset))
    mem_idx = np.random.choice(len(train_dataset), size=n_mem, replace=False)
    non_idx = np.random.choice(len(test_dataset), size=n_non, replace=False)

    def get_x0_cond(sample: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # provider first
        if x0_provider is not None:
            return x0_provider(sample, device=device, in_chans=in_chans)

        # otherwise: must contain latent keys
        if not isinstance(sample, dict):
            raise TypeError("[SecMI] No x0_provider and sample is not dict; cannot infer latent.")
        for k in ("z", "x0", "latent", "latents"):
            if k in sample:
                x0 = _to_tensor(sample[k], device=device)
                x0 = _ensure_bcd(x0)
                x0 = _pad_or_trim_channels(x0, in_chans)
                cond = None
                if "cond" in sample:
                    cond = _to_tensor(sample["cond"], device=device)
                return x0, cond
        raise KeyError("[SecMI] No x0_provider and sample has no latent keys (z/x0/latent).")

    def secmi_errors(idxs: np.ndarray, dataset: Any) -> np.ndarray:
        errs = []
        for i in idxs:
            sample = dataset[int(i)]
            x0, cond = get_x0_cond(sample)  # [B,C,D]
            x0 = _ensure_bcd(x0)

            B = x0.size(0)
            e_acc = torch.zeros(B, device=device, dtype=torch.float32)

            for _ in range(num_queries):
                if use_random_t:
                    t = torch.randint(low=1, high=T, size=(B,), device=device, dtype=torch.long)
                else:
                    t = torch.full((B,), t_fixed, device=device, dtype=torch.long)

                eps = torch.randn_like(x0)
                xt = q_sample(x0, t, eps, alphas_cumprod)
                eps_hat = predict_eps_auto(generated_model, xt, t, cond)

                diff = (eps - eps_hat).reshape(B, -1)
                e = (diff * diff).mean(dim=1)
                e_acc += e

            e_acc = e_acc / float(num_queries)
            errs.append(e_acc.detach().cpu().numpy())

        return np.concatenate(errs, axis=0)

    err_mem = secmi_errors(mem_idx, train_dataset)
    err_non = secmi_errors(non_idx, test_dataset)

    scores = np.concatenate([-err_mem, -err_non], axis=0)
    labels = np.concatenate([np.ones_like(err_mem), np.zeros_like(err_non)], axis=0)

    fpr, tpr, thr = roc_curve_np(labels, scores)
    auc = auc_np(fpr, tpr)

    j = tpr - fpr
    best_i = int(np.argmax(j))
    t_sec = float(thr[best_i])

    return {
        "t_sec": t_sec,
        "AUC": float(auc),
        "TPR@FPR=0.001": tpr_at_fpr(fpr, tpr, 0.001),
        "TPR@FPR=0.01": tpr_at_fpr(fpr, tpr, 0.01),
        "n_member": int(err_mem.shape[0]),
        "n_nonmember": int(err_non.shape[0]),
        "timestep_fixed": int(t_fixed),
        "T": int(T),
        "num_queries": int(num_queries),
        "in_chans": int(in_chans),
        "seed": int(seed),
    }


# ==========================================================
# 7) Standalone runnable demo (dummy)
# ==========================================================
class DummyDiffusion(nn.Module):
    """
    A minimal diffusion-like model:
      - has alphas_cumprod
      - forward returns eps_hat ~ linear(xt)
    """
    def __init__(self, T: int = 1000, in_chans: int = 4, dim: int = 128):
        super().__init__()
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.net = nn.Sequential(
            nn.Linear(in_chans * dim, in_chans * dim),
        )
        self.in_chans = in_chans
        self.dim = dim

    def forward(self, xt: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, D = xt.shape
        x = xt.reshape(B, C * D)
        y = self.net(x).reshape(B, C, D)
        return y


class DummyLatentDataset:
    """
    Returns sample dict with 'z' (latent), so SecMI can run with no encoder.
    """
    def __init__(self, n: int = 2000, chans: int = 4, dim: int = 128):
        self.n = n
        self.chans = chans
        self.dim = dim
        # make train and test slightly different
        self.data = torch.randn(n, chans, dim)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"z": self.data[idx]}


if __name__ == "__main__":
    # Demo: run SecMI end-to-end
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diff_config = {"nnet": {"in_chans": 4}, "z_shape": (3, 128)}
    config = {
        "SEED": 528,
        "EVAL_MIA": True,
        "SECMI_NUM_QUERIES": 4,
        "SECMI_N_MEMBER": 800,
        "SECMI_N_NONMEMBER": 800,
        "SECMI_RANDOM_T": False,
        "SECMI_T_RATIO": 1.0,
    }

    model = DummyDiffusion(T=200, in_chans=4, dim=128).to(device)
    train_ds = DummyLatentDataset(n=1500, chans=4, dim=128)
    test_ds = DummyLatentDataset(n=1500, chans=4, dim=128)

    res = run_secmi_mia_eval(
        generated_model=model,
        train_dataset=train_ds,
        test_dataset=test_ds,
        diff_config=diff_config,
        config=config,
        x0_provider=None,
    )

    print(
        f"[MIA][SecMI-stat] t_sec={res['t_sec']:.4f} "
        f"AUC={res['AUC']:.4f} "
        f"TPR@FPR=0.001={res['TPR@FPR=0.001']:.4f} "
        f"TPR@FPR=0.01={res['TPR@FPR=0.01']:.4f} "
        f"(member={res['n_member']}, nonmember={res['n_nonmember']})"
    )
