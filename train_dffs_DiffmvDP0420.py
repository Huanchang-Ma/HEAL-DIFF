import os
import torch
import tempfile
import einops
import wandb
import torch.nn.functional as F
import numpy as np
from absl import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from diffuser import diff_utils as utils
from diffuser.diff_dataset import get_dataset, collate_fn_dict
from diffuser.dpm import NoiseScheduleVP, DPM_Solver
from utils import set_random_seed
from utils import pad_list
from sklearn.metrics import roc_auc_score, roc_curve
import csv
import json
import torch.nn as nn
import shutil
import glob
import pandas as pd


def tokens_to_multihot(token_ids: torch.Tensor, vocab_dim: int) -> torch.Tensor:
    """
    token_ids: [B, L], 其中 0/1 是特殊 token, 有效 token 从 2 开始
    返回: [B, vocab_dim]
    """
    B, L = token_ids.shape
    target = torch.zeros(B, vocab_dim, device=token_ids.device, dtype=torch.float32)

    valid = token_ids > 1
    idx = (token_ids - 2).clamp(min=0)

    for b in range(B):
        cur = idx[b][valid[b]]
        if cur.numel() > 0:
            target[b, cur] = 1.0
    return target


def soft_jaccard_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """
    logits: [B, V]
    target: [B, V] multi-hot
    """
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=1)
    union = prob.sum(dim=1) + target.sum(dim=1) - inter
    loss = 1.0 - (inter + eps) / (union + eps)
    return loss.mean()


# class DPFullDiffmvWrapper(nn.Module):
#     def __init__(self, full_model, schedule, warm):
#         super().__init__()
#         self.full_model = full_model
#         self.schedule = schedule
#         self.warm = int(warm)
#
#         self.last_z = None
#         self.last_metrics = None
#         # ##
#         # self.register_buffer("ema_grad_conditions", torch.tensor(1.0))
#         # self.register_buffer("ema_grad_procedures", torch.tensor(1.0))
#         # self.register_buffer("ema_grad_drugs", torch.tensor(1.0))
#         # self.grad_ema_beta = 0.9
#         # self.balance_eps = 1e-6
#
#     def forward(self, batch_anchor, *, indices, epoch=None, **batch):
#         z, context, neigh_loss, _, _ = self.full_model.encode_everything(indices, **batch)
#
#         loss_mse, loss_round, T_loss = utils.LSimple(
#             z, self.full_model, self.schedule, context=context, **batch
#         )
#
#         loss_mse_mean = loss_mse.mean()
#         loss_round_mean = loss_round.mean()
#         T_loss_mean = T_loss.mean()
#         neigh_loss_mean = neigh_loss.mean()
#
#         # ===== 新增：视图级离散集合监督 =====
#         feature_order = ['conditions', 'procedures', 'drugs']
#         view_bce_losses = []
#         view_jacc_losses = []
#
#         for feat_idx, feature_key in enumerate(feature_order):
#             logits = self.full_model.get_logits(z[:, feat_idx, :], feature_key)   # [B, V]
#             target = tokens_to_multihot(batch[feature_key], logits.shape[1])      # [B, V]
#
#             bce_loss = F.binary_cross_entropy_with_logits(logits, target)
#             jac_loss = soft_jaccard_loss_from_logits(logits, target)
#
#             view_bce_losses.append(bce_loss)
#             view_jacc_losses.append(jac_loss)
#
#         loss_view_bce = (
#             2.0 * view_bce_losses[0] +
#             2.0 * view_bce_losses[1] +
#             1.0 * view_bce_losses[2]
#         ) / 5.0
#
#         loss_view_jacc = (
#             2.0 * view_jacc_losses[0] +
#             2.0 * view_jacc_losses[1] +
#             1.0 * view_jacc_losses[2]
#         ) / 5.0
#
#         # ===== 原损失 + 新损失 =====
#         if epoch is not None and int(epoch) > self.warm:
#             total_loss = (
#                 1.0 * loss_mse_mean
#                 + 3.0 * loss_round_mean
#                 + 0.5 * T_loss_mean
#                 + 0.3 * neigh_loss_mean
#                 + 1.0 * loss_view_bce
#                 + 1.0 * loss_view_jacc
#             )
#         else:
#             total_loss = (
#                 1.0 * loss_mse_mean
#                 + 3.0 * loss_round_mean
#                 + 0.5 * T_loss_mean
#                 + 1.0 * loss_view_bce
#                 + 1.0 * loss_view_jacc
#             )
#
#         total_loss = total_loss + 0.0 * batch_anchor.float().sum()
#
#         self.last_z = z.detach()
#         self.last_metrics = {
#             "loss_mse": loss_mse_mean.detach(),
#             "loss_c": neigh_loss_mean.detach(),
#             "loss_rounding": loss_round_mean.detach(),
#             "T_loss": T_loss_mean.detach(),
#             "loss_view_bce": loss_view_bce.detach(),
#             "loss_view_jacc": loss_view_jacc.detach(),
#             "total_loss": total_loss.detach(),
#         }
#         return total_loss
class DPFullDiffmvWrapper(nn.Module):
    def __init__(self, full_model, schedule, warm):
        super().__init__()
        self.full_model = full_model
        self.schedule = schedule
        self.warm = int(warm)

        self.last_z = None
        self.last_metrics = None
        ######
        self.register_buffer("ema_grad_conditions", torch.tensor(1.0))
        self.register_buffer("ema_grad_procedures", torch.tensor(1.0))
        self.register_buffer("ema_grad_drugs", torch.tensor(1.0))
        self.grad_ema_beta = 0.9
        self.balance_eps = 1e-6

    def forward(self, batch_anchor, *, indices, epoch=None, **batch):
        z, context, neigh_loss, _, _ = self.full_model.encode_everything(indices, **batch)

        loss_mse, loss_round, T_loss = utils.LSimple(
            z, self.full_model, self.schedule, context=context, **batch
        )

        loss_mse_mean = loss_mse.mean()
        loss_round_mean = loss_round.mean()
        T_loss_mean = T_loss.mean()
        neigh_loss_mean = neigh_loss.mean()

        # ===== 新增：视图级离散集合监督 =====
        feature_order = ['conditions', 'procedures', 'drugs']
        view_bce_losses = []
        view_jacc_losses = []

        for feat_idx, feature_key in enumerate(feature_order):
            logits = self.full_model.get_logits(z[:, feat_idx, :], feature_key)   # [B, V]
            target = tokens_to_multihot(batch[feature_key], logits.shape[1])      # [B, V]

            bce_loss = F.binary_cross_entropy_with_logits(logits, target)
            jac_loss = soft_jaccard_loss_from_logits(logits, target)

            view_bce_losses.append(bce_loss)
            view_jacc_losses.append(jac_loss)

        loss_cond_view = 1.0 * view_bce_losses[0] + 1.0 * view_jacc_losses[0]
        loss_proc_view = 1.0 * view_bce_losses[1] + 1.0 * view_jacc_losses[1]
        loss_drug_view = 1.0 * view_bce_losses[2] + 1.0 * view_jacc_losses[2]
        with torch.no_grad():
            self.ema_grad_conditions.mul_(self.grad_ema_beta).add_(
                loss_cond_view.detach() * (1.0 - self.grad_ema_beta)
            )
            self.ema_grad_procedures.mul_(self.grad_ema_beta).add_(
                loss_proc_view.detach() * (1.0 - self.grad_ema_beta)
            )
            self.ema_grad_drugs.mul_(self.grad_ema_beta).add_(
                loss_drug_view.detach() * (1.0 - self.grad_ema_beta)
            )
        inv_cond = 1.0 / (self.ema_grad_conditions + self.balance_eps)
        inv_proc = 1.0 / (self.ema_grad_procedures + self.balance_eps)
        inv_drug = 1.0 / (self.ema_grad_drugs + self.balance_eps)

        norm = (inv_cond + inv_proc + inv_drug) / 3.0

        w_cond = inv_cond / norm
        w_proc = inv_proc / norm
        w_drug = inv_drug / norm

        loss_view_balanced = (
                w_cond * loss_cond_view +
                w_proc * loss_proc_view +
                w_drug * loss_drug_view
        )

        # ===== 原损失 + 新损失 =====
        if epoch is not None and int(epoch) > self.warm:
            total_loss = (
                1.0 * loss_mse_mean
                + 3.0 * loss_round_mean
                + 0.5 * T_loss_mean
                + 0.3 * neigh_loss_mean
                + 1.0 * loss_view_balanced
            )
        else:
            total_loss = (
                1.0 * loss_mse_mean
                + 3.0 * loss_round_mean
                + 0.5 * T_loss_mean
                + 1.0 * loss_view_balanced
            )

        total_loss = total_loss + 0.0 * batch_anchor.float().sum()

        self.last_z = z.detach()
        self.last_metrics = {
            "loss_mse": loss_mse_mean.detach(),
            "loss_c": neigh_loss_mean.detach(),
            "loss_rounding": loss_round_mean.detach(),
            "T_loss": T_loss_mean.detach(),
            # "loss_view_bce": loss_view_bce.detach(),
            # "loss_view_jacc": loss_view_jacc.detach(),
            "total_loss": total_loss.detach(),
            "loss_cond_view": loss_cond_view.detach(),
            "loss_proc_view": loss_proc_view.detach(),
            "loss_drug_view": loss_drug_view.detach(),
            "w_cond": w_cond.detach(),
            "w_proc": w_proc.detach(),
            "w_drug": w_drug.detach(),
            "loss_view_balanced": loss_view_balanced.detach(),
        }
        return total_loss




def append_row_to_csv(csv_path, row_dict):
    row_df = pd.DataFrame([row_dict])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        new_df = pd.concat([old_df, row_df], ignore_index=True)
    else:
        new_df = row_df

    new_df.to_csv(csv_path, index=False)


def run_diffusion_DP(config, config_health, hp_dataset, tokenizers, exp_num='0'):
    import os
    import tempfile
    import logging
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import einops
    import wandb
    import matplotlib.pyplot as plt

    from contextlib import nullcontext
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    def _unwrap_model(m):
        return m._module if hasattr(m, "_module") else m

    from opacus.layers import DPMultiheadAttention

    def _replace_multihead_attention_for_dp(module):
        replaced = 0
        for name, child in list(module.named_children()):
            if isinstance(child, nn.MultiheadAttention):
                dp_attn = DPMultiheadAttention(
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
                dp_attn = dp_attn.to(
                    device=child.out_proj.weight.device,
                    dtype=child.out_proj.weight.dtype,
                )
                dp_attn.load_state_dict(child.state_dict())
                setattr(module, name, dp_attn)
                replaced += 1
            else:
                replaced += _replace_multihead_attention_for_dp(child)
        return replaced

    def full_centroid_refresh(epoch):
        if not hasattr(train_state.nnet, "update_train_visit_emb"):
            return
        if not hasattr(train_state.nnet, "e_step"):
            return
        if not hasattr(train_state.nnet, "update_centroid_emb"):
            return

        _disable_dp_hooks()
        try:
            model_for_estep = train_state.nnet
            model_for_estep.eval()

            total_visit_embs = []
            total_indices = []

            static_mask_epochs = int(config.get("train", {}).get("static_mask_epochs", 8))
            use_static_mask = bool(epoch < static_mask_epochs)

            old_centroids = None
            if hasattr(train_state.nnet, "visit_centroids") and train_state.nnet.visit_centroids is not None:
                old_centroids = train_state.nnet.visit_centroids.detach().clone()

            old_assign = None
            if hasattr(train_state.nnet, "visit_2cluster") and train_state.nnet.visit_2cluster is not None:
                old_assign = train_state.nnet.visit_2cluster.detach().clone()

            for indices, raw_batch in estep_loader:
                indices, batch = convert_input((indices, raw_batch), mask_static=use_static_mask)
                z, _, _, _, _ = model_for_estep.encode_everything(indices, **batch)
                total_visit_embs.append(z.view(z.shape[0], -1).detach())
                total_indices.append(indices.detach())

            total_visit_embs = torch.cat(total_visit_embs, dim=0)
            total_indices = torch.cat(total_indices, dim=0)
            total_visit_embs = total_visit_embs[total_indices.argsort()]

            train_state.nnet.update_train_visit_emb(total_visit_embs)
            visit_centroids, _ = train_state.nnet.e_step()
            train_state.nnet.update_centroid_emb(visit_centroids)

            if old_centroids is not None:
                new_centroids = train_state.nnet.visit_centroids.detach()
                centroid_delta = (new_centroids - old_centroids).abs().mean().item()
                print(f"[FULL REFRESH] centroid_delta={centroid_delta:.8f}")

            if old_assign is not None:
                new_assign = train_state.nnet.visit_2cluster.detach()
                assign_change = (new_assign != old_assign).float().mean().item()
                print(f"[FULL REFRESH] assign_change_ratio={assign_change:.8f}")

        finally:
            _enable_dp_hooks()

    def _replace_batchnorm_for_dp(module):
        replaced = 0
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm1d):
                new_layer = nn.LayerNorm(child.num_features, elementwise_affine=False)
                setattr(module, name, new_layer)
                replaced += 1
            else:
                replaced += _replace_batchnorm_for_dp(child)
        return replaced

    def _find_trainable_modules_with_buffers(model):
        bad = []
        for mod_name, mod in model.named_modules():
            direct_buffers = list(mod.named_buffers(recurse=False))
            if not direct_buffers:
                continue
            has_trainable = any(p.requires_grad for p in mod.parameters())
            if has_trainable:
                bad.append({
                    "module": mod_name if mod_name else "<root>",
                    "type": mod.__class__.__name__,
                    "buffers": [name for name, _ in direct_buffers],
                })
        return bad

    def _find_functorch_unfriendly_modules(model):
        import inspect
        bad = []
        for name, m in model.named_modules():
            try:
                sig = inspect.signature(m.forward)
            except (TypeError, ValueError):
                continue

            extra_args = [k for k in ("timesteps", "context") if k in sig.parameters]
            direct_trainable = [pn for pn, p in m.named_parameters(recurse=False) if p.requires_grad]
            if extra_args and direct_trainable:
                bad.append({
                    "module": name if name else "<root>",
                    "type": m.__class__.__name__,
                    "extra_args": extra_args,
                    "direct_trainable": direct_trainable,
                })
        return bad

    def _freeze_structurally_inactive_params(model):
        frozen = []

        if hasattr(model, "visit_encoder") and hasattr(model.visit_encoder, "embeddings_comp"):
            for key in ["conditions", "procedures", "drugs"]:
                if key in model.visit_encoder.embeddings_comp:
                    emb = model.visit_encoder.embeddings_comp[key]
                    emb.weight.requires_grad = False
                    frozen.append(f"visit_encoder.embeddings_comp.{key}.weight")

        if hasattr(model, "visit_encoder") and hasattr(model.visit_encoder, "set_encoder"):
            if getattr(model.visit_encoder, "mode", None) != "attn":
                for name, p in model.visit_encoder.set_encoder.named_parameters():
                    p.requires_grad = False
                    frozen.append(f"visit_encoder.set_encoder.{name}")

        if hasattr(model, "note_linear"):
            for name, p in model.note_linear.named_parameters():
                p.requires_grad = False
                frozen.append(f"note_linear.{name}")

        if hasattr(model, "context_encoder") and hasattr(model.context_encoder, "history_encoder"):
            he = model.context_encoder.history_encoder

            if hasattr(he, "attention_dict") and "note" in he.attention_dict:
                for name, p in he.attention_dict["note"].named_parameters():
                    p.requires_grad = False
                    frozen.append(f"context_encoder.history_encoder.attention_dict.note.{name}")

            hist_mode = he.config.get("hist_mode", None)

            if hist_mode == "individual":
                if hasattr(he, "attention_joint"):
                    for name, p in he.attention_joint.named_parameters():
                        p.requires_grad = False
                        frozen.append(f"context_encoder.history_encoder.attention_joint.{name}")

            elif hist_mode == "joint":
                for key in ["conditions", "procedures", "drugs"]:
                    if hasattr(he, "attention_dict") and key in he.attention_dict:
                        for name, p in he.attention_dict[key].named_parameters():
                            p.requires_grad = False
                            frozen.append(f"context_encoder.history_encoder.attention_dict.{key}.{name}")

        return frozen

    def debug_bad_gradsample_modules(model, bad_param_names):
        import inspect

        bad_prefixes = set()
        for n in bad_param_names:
            if "." in n:
                bad_prefixes.add(n.rsplit(".", 1)[0])

        for mod_name, mod in model.named_modules():
            if mod_name in bad_prefixes:
                print(f"[BAD MODULE] {mod_name} :: {mod.__class__.__name__}")
                try:
                    print("  forward sig:", inspect.signature(mod.forward))
                except Exception:
                    pass

    def debug_modules_with_empty_positional_inputs(model):
        handles = []

        def _hook(name):
            def fn(module, args, output):
                has_trainable = any(p.requires_grad for p in module.parameters(recurse=False))
                if not has_trainable:
                    return
                if len(args) == 0:
                    print(f"[EMPTY ARGS] {name} :: {module.__class__.__name__}")
                else:
                    last = args[-1]
                    if not torch.is_tensor(last):
                        print(f"[LAST ARG NOT TENSOR] {name} :: {module.__class__.__name__} :: {type(last)}")
            return fn

        for name, m in model.named_modules():
            handles.append(m.register_forward_hook(_hook(name)))
        return handles

    def _disable_dp_hooks():
        if use_dp and hasattr(train_state, "dp_nnet") and hasattr(train_state.dp_nnet, "disable_hooks"):
            train_state.dp_nnet.disable_hooks()

    def _enable_dp_hooks():
        if use_dp and hasattr(train_state, "dp_nnet") and hasattr(train_state.dp_nnet, "enable_hooks"):
            train_state.dp_nnet.enable_hooks()

    def _pick_batch_item(v, idx):
        if isinstance(v, list):
            return v[idx]
        if isinstance(v, tuple):
            return v[idx]
        if torch.is_tensor(v):
            item = v[idx]
            if item.dim() == 0:
                return item.item()
            return item.detach().cpu().tolist()
        if isinstance(v, np.ndarray):
            item = v[idx]
            if np.ndim(item) == 0:
                return item.item()
            return item.tolist()
        return v

    def _json_safe(x):
        if torch.is_tensor(x):
            if x.dim() == 0:
                return x.item()
            return x.detach().cpu().tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {k: _json_safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_json_safe(v) for v in x]
        return x

    def _sample_signature(cond, proc, drug):
        return (
            tuple(sorted(cond)),
            tuple(sorted(proc)),
            tuple(sorted(drug)),
        )

    def _keep_top_by_label(rows):
        label_key = config.get("sample", {}).get("label_key", None)
        if not label_key:
            return rows

        keep_ratio = float(config.get("sample", {}).get("keep_ratio", 0.35))
        min_keep_per_label = int(config.get("sample", {}).get("min_keep_per_label", 16))

        buckets = {}
        for row in rows:
            label = str(row.get(label_key, "__missing__"))
            buckets.setdefault(label, []).append(row)

        kept = []
        for _, items in buckets.items():
            items = sorted(items, key=lambda x: x["synth_score"], reverse=True)
            keep_n = max(min_keep_per_label, int(len(items) * keep_ratio))
            kept.extend(items[:keep_n])
        return kept

    def _remove_buffers_from_trainable_modules(model):
        """
        递归地把所有“有可训练参数”的模块上的 direct buffers 挪出 _buffers，
        保留成普通属性，避免 Opacus 因 trainable layer with buffers 报错。
        """
        removed = {}

        for mod_name, mod in model.named_modules():
            has_trainable = any(p.requires_grad for p in mod.parameters())
            if not has_trainable:
                continue

            for buf_name, buf in list(mod.named_buffers(recurse=False)):
                full_name = f"{mod_name}.{buf_name}" if mod_name else buf_name
                removed[full_name] = buf
                del mod._buffers[buf_name]
                object.__setattr__(mod, buf_name, buf)

        return removed

    print("run_diffusion")
    set_random_seed(config_health['SEED'])

    data_n = config_health['DATASET']
    task = config_health['TASK']

    device = torch.device('cuda:' + str(config['device']) if torch.cuda.is_available() else 'cpu')
    logging.info(f'using device: {device}')

    batch_size = config['train']['batch_size']

    eps_tag = f"eps{str(config['train']['dp_target_epsilon']).replace('.', 'p')}"
    ckpt_dir = os.path.join(config['ckpt_root'], data_n, task, eps_tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb.init(
        dir=os.path.abspath(config['workdir']),
        project=f'mmht_{data_n}_{task}',
        config=config,
        name=config['hparams'],
        job_type='train',
        mode='offline'
    )
    utils.set_logger(
        log_level='info',
        fname=os.path.join(config['workdir'], data_n + '-' + task + '-' + exp_num + '-output.log')
    )
    logging.info(config)

    step_history = {
        'step': [],
        'loss_mse': [],
        'loss_c': [],
        'loss_rounding': [],
        'T_loss': [],
        'total_loss': [],
        'lr': [],
    }
    epoch_history = {
        'epoch': [],
        'quality_mse': [],
        'quality_jaccard_conditions': [],
        'quality_jaccard_procedures': [],
        'quality_jaccard_drugs': [],
        'quality_jaccard_mean': [],
        'train_mse': [],
        'train_jaccard_conditions': [],
        'train_jaccard_procedures': [],
        'train_jaccard_drugs': [],
        'train_jaccard_mean': [],
    }

    dataset = get_dataset(data_n, aug=False, samples=hp_dataset, config=config)

    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_dict
    )
    test_dataset = dataset.get_split(split='test', labeled=True)
    test_dataset_loader = DataLoader(
        test_dataset,
        batch_size=config['sample']['mini_batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_dict
    )
    estep_loader = DataLoader(
        train_dataset,
        batch_size=config['sample']['mini_batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_dict,
    )

    print("[LOADER LEN BEFORE DP]", len(train_dataset_loader))
    logging.info(f"[LOADER LEN BEFORE DP] {len(train_dataset_loader)}")

    print("Input for diffusion", train_dataset[0])
    print("AAAAAAAA", (len(train_dataset) // batch_size + 1) * batch_size, len(train_dataset))

    _betas = utils.stable_diffusion_beta_schedule()
    _schedule = utils.Schedule(_betas)
    logging.info(f'use {_schedule}')

    train_state = utils.initialize_train_state(config, device, tokenizers, len(train_dataset))
    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = (
        train_state.nnet,
        train_state.nnet_ema,
        train_state.optimizer,
        train_dataset_loader,
        test_dataset_loader,
    )
    lr_scheduler = train_state.lr_scheduler

    if config['train'].get('resume', False):
        train_state.resume(ckpt_dir)

    loss_csv_path = os.path.join(config["workdir"], f"{data_n}-{task}-{exp_num}-loss.csv")
    is_new = not os.path.exists(loss_csv_path)
    loss_f = open(loss_csv_path, "a", newline="")
    loss_writer = csv.writer(loss_f)
    if is_new:
        loss_writer.writerow(["step", "loss_mse", "loss_c", "loss_rounding", "T_loss", "total_loss", "lr"])
        loss_f.flush()
    logging.info(f"Loss csv: {loss_csv_path}")

    privacy_engine = None
    use_dp = bool(config['train'].get('dp_sgd', False))

    if use_dp:
        from opacus import PrivacyEngine
        from opacus.utils.batch_memory_manager import BatchMemoryManager
        from opacus.validators import ModuleValidator

        privacy_engine = PrivacyEngine(
            accountant=config['train'].get('dp_accountant', 'prv'),
            secure_mode=bool(config['train'].get('dp_secure_mode', False)),
        )

        bad = _find_trainable_modules_with_buffers(nnet)
        print("[DP] trainable modules with buffers:", bad)

        replaced = _replace_multihead_attention_for_dp(nnet)
        print(f"[DP] replaced multihead attention count = {replaced}")

        bn_replaced = _replace_batchnorm_for_dp(nnet)
        print(f"[DP] replaced batchnorm count = {bn_replaced}")

        ema_replaced = _replace_multihead_attention_for_dp(train_state.nnet_ema)
        print(f"[DP] replaced multihead attention count (nnet_ema) = {ema_replaced}")

        ema_bn_replaced = _replace_batchnorm_for_dp(train_state.nnet_ema)
        print(f"[DP] replaced batchnorm count (nnet_ema) = {ema_bn_replaced}")

        nnet = train_state.nnet
        nnet_ema = train_state.nnet_ema

        train_state.nnet_ema.load_state_dict(train_state.nnet.state_dict(), strict=True)
        train_state.nnet_ema.requires_grad_(False)
        train_state.nnet_ema.eval()

        errors = ModuleValidator.validate(nnet, strict=False)
        print("[DP] validator errors after replacement:", errors)
        if errors:
            raise RuntimeError(errors)

        bad2 = _find_functorch_unfriendly_modules(nnet)
        print("[DP] functorch-unfriendly trainable modules after freezing:", bad2)

        base_nnet = train_state.nnet

        if not hasattr(base_nnet, "visit_2cluster") or base_nnet.visit_2cluster is None:
            base_nnet.visit_2cluster = torch.zeros(
                len(train_dataset),
                dtype=torch.long,
                device=device,
            )
            print("[DP] initialized dummy visit_2cluster")

        # 原始 root pos_embed 必须冻结，否则 Opacus 会在 full_model:DIT 上继续报错
        if hasattr(base_nnet, "pos_embed") and isinstance(base_nnet.pos_embed, torch.nn.Parameter):
            base_nnet.pos_embed.requires_grad = False
            print("[DP] froze root pos_embed for Opacus compatibility")

        # 结构性不活跃分支冻结
        mandatory_frozen = _freeze_structurally_inactive_params(base_nnet)
        print("[DP] mandatory frozen structurally inactive params:", mandatory_frozen)

        # 新增的位置增量模块
        use_pos_adapter = bool(config['train'].get('use_pos_adapter', False))
        if hasattr(base_nnet, "pos_embed_adapter"):
            base_nnet.pos_embed_adapter.requires_grad_(use_pos_adapter)
            print(f"[DP] pos_embed_adapter trainable = {use_pos_adapter}")

        # 先创建 wrapper，并先搬到 device
        full_dp_model = DPFullDiffmvWrapper(
            base_nnet,
            _schedule,
            warm=config.get('warm', 10),
            # warm=config.get('warm', 5000),
        ).to(device)

        # 原始 root pos_embed 必须冻结，否则 Opacus 会在 full_model:DIT 上报 trainable layer with buffers
        if hasattr(base_nnet, "pos_embed") and isinstance(base_nnet.pos_embed, torch.nn.Parameter):
            base_nnet.pos_embed.requires_grad = False
            print("[DP] froze root pos_embed for Opacus compatibility")

        bad2 = _find_functorch_unfriendly_modules(full_dp_model.full_model)
        print("[DP] functorch-unfriendly trainable modules after freezing:", bad2)

        use_pos_adapter = bool(config['train'].get('use_pos_adapter', False))
        if use_pos_adapter and hasattr(base_nnet, "pos_embed"):
            if not hasattr(base_nnet, "pos_embed_adapter"):
                base_nnet.pos_embed_adapter = PosEmbedAdapter(base_nnet.pos_embed.shape).to(device)
                print("[DP] attached trainable pos_embed_adapter")

        optimizer = torch.optim.AdamW(
            [p for p in full_dp_model.parameters() if p.requires_grad],
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer'].get('weight_decay', 0.0),
        )

        ##  'dp_clipping': 'flat'
        full_dp_model, optimizer, train_dataset_loader = privacy_engine.make_private_with_epsilon(
            module=full_dp_model,
            optimizer=optimizer,
            data_loader=train_dataset_loader,
            target_epsilon=float(config['train']['dp_target_epsilon']),
            target_delta=float(config['train']['dp_delta']),
            epochs=int(config['train']['n_epochs']),
            max_grad_norm=float(config['train']['dp_max_grad_norm']),
            batch_first=True,
            loss_reduction="mean",
            poisson_sampling=bool(config['train'].get('dp_poisson_sampling', True)),
            grad_sample_mode=str(config['train'].get('dp_grad_sample_mode', 'functorch')),
            clipping=str(config['train'].get('dp_clipping', 'flat')),
        )

        # ##  'dp_clipping': 'per_layer'
        # clipping_mode = str(config['train'].get('dp_clipping', 'flat'))
        # base_max_grad_norm = float(config['train']['dp_max_grad_norm'])
        # trainable_params = [p for p in full_dp_model.parameters() if p.requires_grad]
        # if clipping_mode == "per_layer":
        #     max_grad_norm = [base_max_grad_norm for _ in trainable_params]
        # else:
        #     max_grad_norm = base_max_grad_norm
        # full_dp_model, optimizer, train_dataset_loader = privacy_engine.make_private_with_epsilon(
        #     module=full_dp_model,
        #     optimizer=optimizer,
        #     data_loader=train_dataset_loader,
        #     target_epsilon=float(config['train']['dp_target_epsilon']),
        #     target_delta=float(config['train']['dp_delta']),
        #     epochs=int(config['train']['n_epochs']),
        #     max_grad_norm=max_grad_norm,
        #     batch_first=True,
        #     loss_reduction="mean",
        #     poisson_sampling=bool(config['train'].get('dp_poisson_sampling', True)),
        #     grad_sample_mode=str(config['train'].get('dp_grad_sample_mode', 'functorch')),
        #     clipping=clipping_mode,
        # )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=(
                train_state.lr_scheduler.lr_lambdas[0]
                if hasattr(train_state.lr_scheduler, "lr_lambdas")
                else (lambda step: 1.0)
            ),
        )

        train_state.nnet = base_nnet
        train_state.dp_nnet = full_dp_model
        train_state.optimizer = optimizer
        train_state.lr_scheduler = lr_scheduler

        debug_target = train_state.dp_nnet
        if hasattr(debug_target, "_module"):
            debug_target = debug_target._module
        debug_handles = debug_modules_with_empty_positional_inputs(debug_target)

    def _get_decode_threshold(feature_key):
        default_thr = float(config.get("threshod", 0.4))
        threshold_map = config.get("sample", {}).get("decode_thresholds", {})
        if isinstance(threshold_map, dict) and feature_key in threshold_map:
            return float(threshold_map[feature_key])
        return default_thr

    def _get_decode_topk(feature_key):
        topk_map = config.get("sample", {}).get("decode_topk", {})
        if isinstance(topk_map, dict) and feature_key in topk_map:
            val = topk_map[feature_key]
            return None if val is None else int(val)
        if feature_key == "conditions":
            return int(config.get("sample", {}).get("conditions_topk", 20))
        return None

    def decode_with_stats(_batch, get_logits=None):
        eval_model = train_state.nnet_ema if use_dp else train_state.nnet
        eval_model.eval()

        hard_samples = {}
        stats = {}
        feature_order = ['conditions', 'procedures', 'drugs']

        for feat_idx, feature_key in enumerate(feature_order):
            logits = get_logits(_batch[:, feat_idx, :], feature_key)
            probs = torch.sigmoid(logits)
            max_indices = torch.argmax(probs, dim=1)

            topk = _get_decode_topk(feature_key)
            thr = _get_decode_threshold(feature_key)

            if topk is not None and topk > 0:
                k = min(int(topk), probs.shape[1])
                top_vals, top_idx = torch.topk(probs, k=k, dim=1)
                token_rows = (top_idx + 2).detach().cpu().tolist()
                tokens = tokenizers[feature_key].batch_decode_2d(token_rows)
                conf_tensor = top_vals.mean(dim=1)
                len_tensor = torch.full_like(conf_tensor, fill_value=float(k))
            else:
                mask = probs >= thr
                token_rows = []
                conf_vals = []
                len_vals = []

                for b in range(probs.size(0)):
                    sel = torch.where(mask[b])[0]
                    if sel.numel() == 0:
                        sel = max_indices[b:b + 1]
                    token_rows.append((sel + 2).detach().cpu().tolist())
                    conf_vals.append(probs[b, sel].mean())
                    len_vals.append(float(sel.numel()))

                tokens = tokenizers[feature_key].batch_decode_2d(token_rows)
                conf_tensor = torch.stack(conf_vals)
                len_tensor = torch.tensor(len_vals, device=probs.device, dtype=torch.float32)

            hard_samples[feature_key] = tokens
            stats[f"{feature_key}_conf"] = conf_tensor
            stats[f"{feature_key}_len"] = len_tensor

        return _batch, hard_samples, stats

    def decode(_batch, get_logits=None, thred=config['threshod']):
        _, hard_samples, _ = decode_with_stats(_batch, get_logits=get_logits)
        return _batch, hard_samples

    # def build_sample_score(stats, samples=None, z_init=None):
    #     conf_score = (
    #         # 0.4 * stats['conditions_conf'] +
    #         # 0.3 * stats['procedures_conf'] +
    #         # 0.3 * stats['drugs_conf']
    #         0.6 * stats['conditions_conf'] +
    #         0.2 * stats['procedures_conf'] +
    #         0.2 * stats['drugs_conf']
    #     )
    #
    #     target_proc_len = float(config.get("sample", {}).get("target_proc_len", 8.0))
    #     target_drug_len = float(config.get("sample", {}).get("target_drug_len", 8.0))
    #
    #     proc_pen = torch.abs(stats['procedures_len'] - target_proc_len) / max(target_proc_len, 1.0)
    #     drug_pen = torch.abs(stats['drugs_len'] - target_drug_len) / max(target_drug_len, 1.0)
    #
    #     score = conf_score - 0.15 * (0.5 * proc_pen + 0.5 * drug_pen)
    #
    #     if bool(config.get("sample", {}).get("use_latent_score", True)) and z_init is not None:
    #         latent_pen = ((samples - z_init) ** 2).reshape(samples.shape[0], -1).mean(dim=1)
    #         score = score - 0.05 * latent_pen
    #
    #     return score
    def build_sample_score(stats, samples=None, z_init=None):
        conf_score = (
                0.5 * stats['conditions_conf'] +
                0.2 * stats['procedures_conf'] +
                0.3 * stats['drugs_conf']
        )
        target_cond_len = float(config.get("sample", {}).get("target_cond_len", 8.0))
        target_proc_len = float(config.get("sample", {}).get("target_proc_len", 8.0))
        target_drug_len = float(config.get("sample", {}).get("target_drug_len", 8.0))

        cond_pen = torch.abs(stats['conditions_len'] - target_cond_len) / max(target_cond_len, 1.0)
        proc_pen = torch.abs(stats['procedures_len'] - target_proc_len) / max(target_proc_len, 1.0)
        drug_pen = torch.abs(stats['drugs_len'] - target_drug_len) / max(target_drug_len, 1.0)

        score = conf_score - 0.10 * (0.4 * cond_pen + 0.3 * proc_pen + 0.3 * drug_pen)

        if bool(config.get("sample", {}).get("use_latent_score", True)) and z_init is not None:
            latent_pen = ((samples - z_init) ** 2).reshape(samples.shape[0], -1).mean(dim=1)
            score = score - 0.02 * latent_pen

        return score

    # def dpm_solver_sample(_n_samples, _sample_steps, return_stats=False, **kwargs):
    #     _z_init = torch.randn(_n_samples, *config['z_shape'], device=device)
    #     z_init2 = kwargs['z_init']
    #     mask = kwargs['mask'].unsqueeze(dim=-1).repeat(1, 1, config['z_shape'][1])
    #     _z_init = torch.where(mask, _z_init, z_init2)
    #
    #     noise_schedule = NoiseScheduleVP(
    #         schedule='discrete',
    #         betas=torch.tensor(_betas, device=device).float()
    #     )
    #
    #     def model_fn(x, t_continuous):
    #         t = t_continuous * _schedule.N
    #         return cfg_nnet(x, t, kwargs['context'])
    #
    #     dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
    #     _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1.)
    #
    #     if return_stats:
    #         _, hard_samples, stats = decode_with_stats(_z, train_state.nnet.get_logits)
    #         return _z, hard_samples, stats
    #     else:
    #         return decode(_z, train_state.nnet.get_logits)
    def dpm_solver_sample(_n_samples, _sample_steps, return_stats=False, **kwargs):
        _z_init = torch.randn(_n_samples, *config['z_shape'], device=device)
        z_init2 = kwargs['z_init']
        mask = kwargs['mask'].unsqueeze(dim=-1).repeat(1, 1, config['z_shape'][1])
        _z_init = torch.where(mask, _z_init, z_init2)

        noise_schedule = NoiseScheduleVP(
            schedule='discrete',
            betas=torch.tensor(_betas, device=device).float()
        )

        def model_fn(x, t_continuous):
            t = t_continuous * _schedule.N
            return cfg_nnet(x, t, kwargs['context'])

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1.)

        # _z = z_init2 + 0.3 * (_z - z_init2)
        _z = z_init2 + 0.1 * (_z - z_init2)  ## 较好结果

        if return_stats:
            _, hard_samples, stats = decode_with_stats(_z, train_state.nnet.get_logits)
            return _z, hard_samples, stats
        else:
            return decode(_z, train_state.nnet.get_logits)

    def best_of_k_sample(_n_samples, _sample_steps, context, z_init, mask=None, return_scores=False, return_latent_shift=False):
        best_of_k = int(config.get("sample", {}).get("best_of_k", 1))

        if best_of_k <= 1:
            samples, hard_samples, stats = dpm_solver_sample(
                _n_samples=_n_samples,
                _sample_steps=_sample_steps,
                context=context,
                z_init=z_init,
                mask=mask,
                return_stats=True,
            )
            scores = build_sample_score(stats, samples=samples, z_init=z_init)

            latent_shift = ((samples - z_init) ** 2).reshape(samples.shape[0], -1).mean()

            if return_scores and return_latent_shift:
                return samples, hard_samples, scores, float(latent_shift.item())
            if return_scores:
                return samples, hard_samples, scores
            if return_latent_shift:
                return samples, hard_samples, float(latent_shift.item())
            return samples, hard_samples

        best_samples = None
        best_hard_samples = None
        best_scores = None

        for _ in range(best_of_k):
            samples_k, hard_k, stats_k = dpm_solver_sample(
                _n_samples=_n_samples,
                _sample_steps=_sample_steps,
                context=context,
                z_init=z_init,
                mask=mask,
                return_stats=True,
            )
            score_k = build_sample_score(stats_k, samples=samples_k, z_init=z_init)

            if best_scores is None:
                best_scores = score_k
                best_samples = samples_k
                best_hard_samples = hard_k
            else:
                better = score_k > best_scores
                best_scores = torch.where(better, score_k, best_scores)

                view_shape = [samples_k.shape[0]] + [1] * (samples_k.dim() - 1)
                better_view = better.view(*view_shape)
                best_samples = torch.where(better_view, samples_k, best_samples)

                merged_hard = {}
                for fk in hard_k.keys():
                    merged_hard[fk] = [
                        hard_k[fk][b] if bool(better[b].item()) else best_hard_samples[fk][b]
                        for b in range(len(hard_k[fk]))
                    ]
                best_hard_samples = merged_hard

        latent_shift = ((best_samples - z_init) ** 2).reshape(best_samples.shape[0], -1).mean()
        if return_scores and return_latent_shift:
            return best_samples, best_hard_samples, best_scores, float(latent_shift.item())
        if return_scores:
            return best_samples, best_hard_samples, best_scores
        if return_latent_shift:
            return best_samples, best_hard_samples, float(latent_shift.item())
        return best_samples, best_hard_samples

    def _compute_sample_grad_stats(model, max_grad_norm):
        sq_norms = None
        for name, p in model.named_parameters():
            gs = getattr(p, "grad_sample", None)
            if gs is None:
                continue
            gs = gs.reshape(gs.shape[0], -1)
            cur = (gs ** 2).sum(dim=1)
            sq_norms = cur if sq_norms is None else sq_norms + cur

        if sq_norms is None:
            return None

        sample_norms = torch.sqrt(sq_norms)
        clip_ratio = (sample_norms > max_grad_norm).float().mean().item()

        return {
            "sample_grad_norm_mean": float(sample_norms.mean().item()),
            "sample_grad_norm_p90": float(torch.quantile(sample_norms, 0.90).item()),
            "sample_grad_norm_p95": float(torch.quantile(sample_norms, 0.95).item()),
            "sample_grad_norm_max": float(sample_norms.max().item()),
            "sample_grad_norm_min": float(sample_norms.min().item()),
            "clip_ratio": float(clip_ratio),
        }

    def _group_param_names_by_view(model):

        view_groups = {
            "conditions": [],
            "procedures": [],
            "drugs": [],
            "shared": [],
        }
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            lname = name.lower()
            if "condition" in lname:
                view_groups["conditions"].append(name)
            elif "procedure" in lname:
                view_groups["procedures"].append(name)
            elif "drug" in lname:
                view_groups["drugs"].append(name)
            else:
                view_groups["shared"].append(name)

        return view_groups

    def _compute_sample_grad_stats_by_name_subset(model, selected_names, max_grad_norm):
        """
        对 selected_names 对应的参数，统计 per-sample grad norm
        """
        sq_norms = None
        selected_names = set(selected_names)
        for name, p in model.named_parameters():
            if name not in selected_names:
                continue
            gs = getattr(p, "grad_sample", None)
            if gs is None:
                continue
            gs = gs.reshape(gs.shape[0], -1)  # [B, ...] -> [B, D]
            cur = (gs ** 2).sum(dim=1)  # [B]
            sq_norms = cur if sq_norms is None else sq_norms + cur
        if sq_norms is None:
            return None

        sample_norms = torch.sqrt(sq_norms)
        clip_ratio = (sample_norms > max_grad_norm).float().mean().item()

        return {
            "sample_norms": sample_norms.detach(),
            "mean": float(sample_norms.mean().item()),
            "p90": float(torch.quantile(sample_norms, 0.90).item()),
            "p95": float(torch.quantile(sample_norms, 0.95).item()),
            "max": float(sample_norms.max().item()),
            "min": float(sample_norms.min().item()),
            "clip_ratio": float(clip_ratio),
        }

    def _count_valid_tokens(x: torch.Tensor) -> torch.Tensor:
        return (x > 1).sum(dim=1)

    def _mark_rare_drug_patients(drug_tokens: torch.Tensor, rare_threshold: int = 2) -> torch.Tensor:
        """
        drug_tokens: [B, L]
        有效 drug token 数 <= rare_threshold 的 sample 视作 rare-drug patient
        返回 [B] bool
        """
        drug_cnt = _count_valid_tokens(drug_tokens)
        return drug_cnt <= rare_threshold

    def _plot_cdf(values_dict, save_path, title="CDF of per-sample gradient norms"):
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(8, 6))

        for name, vals in values_dict.items():
            if torch.is_tensor(vals):
                vals = vals.detach().cpu().numpy()
            vals = np.asarray(vals, dtype=np.float64)
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                continue

            vals = np.sort(vals)
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            plt.plot(vals, cdf, label=name)

        plt.xlabel("Per-sample gradient norm")
        plt.ylabel("CDF")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def _compute_grad_norm_from_param_subset(model, selected_names):
        sq = None
        selected_names = set(selected_names)

        for name, p in model.named_parameters():
            if name not in selected_names:
                continue
            if p.grad is None:
                continue

            g = p.grad.reshape(-1)
            cur = (g ** 2).sum()
            sq = cur if sq is None else sq + cur

        if sq is None:
            return 0.0
        return float(torch.sqrt(sq).item())

    def collect_nondp_view_grad_norms_single_batch(model, batch, indices, schedule, epoch, warm):
        """
        对一个 batch 中每个 patient 单独 backward，统计 per-view grad norm
        这里只用于诊断，不参与正式训练更新。
        """
        model.zero_grad(set_to_none=True)
        model.train()

        view_groups = _group_param_names_by_view(model)

        out = {
            "conditions": [],
            "procedures": [],
            "drugs": [],
        }

        B = indices.shape[0]

        for b in range(B):
            model.zero_grad(set_to_none=True)

            one_indices = indices[b:b + 1]

            one_batch = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    one_batch[k] = v[b:b + 1]
                elif isinstance(v, list):
                    one_batch[k] = [_pick_batch_item(v, b)]
                elif isinstance(v, tuple):
                    one_batch[k] = [_pick_batch_item(v, b)]
                elif isinstance(v, np.ndarray):
                    item = _pick_batch_item(v, b)
                    one_batch[k] = [item]
                else:
                    one_batch[k] = v

            z, context, neigh_loss, _, _ = model.encode_everything(one_indices, **one_batch)

            loss_mse, loss_round, T_loss = utils.LSimple(
                z, model, schedule, context=context, **one_batch
            )

            loss_mse_mean = loss_mse.mean()
            loss_round_mean = loss_round.mean()
            T_loss_mean = T_loss.mean()
            neigh_loss_mean = neigh_loss.mean()

            # 视图级离散集合监督
            feature_order = ['conditions', 'procedures', 'drugs']
            view_bce_losses = []
            view_jacc_losses = []

            for feat_idx, feature_key in enumerate(feature_order):
                logits = model.get_logits(z[:, feat_idx, :], feature_key)
                target = tokens_to_multihot(one_batch[feature_key], logits.shape[1])

                bce_loss = F.binary_cross_entropy_with_logits(logits, target)
                jac_loss = soft_jaccard_loss_from_logits(logits, target)

                view_bce_losses.append(bce_loss)
                view_jacc_losses.append(jac_loss)

            loss_view_bce = (
                                    2.0 * view_bce_losses[0] +
                                    2.0 * view_bce_losses[1] +
                                    1.0 * view_bce_losses[2]
                            ) / 5.0

            loss_view_jacc = (
                                     2.0 * view_jacc_losses[0] +
                                     2.0 * view_jacc_losses[1] +
                                     1.0 * view_jacc_losses[2]
                             ) / 5.0

            if epoch is not None and int(epoch) > warm:
                total_loss = (
                        1.0 * loss_mse_mean
                        + 3.0 * loss_round_mean
                        + 0.5 * T_loss_mean
                        + 0.3 * neigh_loss_mean
                        + 1.0 * loss_view_bce
                        + 1.0 * loss_view_jacc
                )
            else:
                total_loss = (
                        1.0 * loss_mse_mean
                        + 3.0 * loss_round_mean
                        + 0.5 * T_loss_mean
                        + 1.0 * loss_view_bce
                        + 1.0 * loss_view_jacc
                )

            total_loss.backward()

            for view_name in ["conditions", "procedures", "drugs"]:
                gn = _compute_grad_norm_from_param_subset(model, view_groups[view_name])
                out[view_name].append(gn)

        for k in out:
            out[k] = torch.tensor(out[k], dtype=torch.float32)

        return out

    def collect_nondp_view_grad_norms_full_epoch(model, data_loader, schedule, epoch, warm, mask_static=True):
        """
        遍历整个 data_loader，对所有 batch、所有 patient 统计 per-view grad norm
        """
        out = {
            "conditions": [],
            "procedures": [],
            "drugs": [],
        }

        for indices, raw_batch in data_loader:
            indices, batch = convert_input((indices, raw_batch), mask_static=mask_static)

            batch_stats = collect_nondp_view_grad_norms_single_batch(
                model=model,
                batch=batch,
                indices=indices,
                schedule=schedule,
                epoch=epoch,
                warm=warm,
            )

            for view_name in ["conditions", "procedures", "drugs"]:
                out[view_name].append(batch_stats[view_name].detach().cpu())

        for view_name in out:
            if len(out[view_name]) > 0:
                out[view_name] = torch.cat(out[view_name], dim=0)
            else:
                out[view_name] = torch.empty(0, dtype=torch.float32)

        return out

    def summarize_grad_tensor(x):
        if x.numel() == 0:
            return None
        return {
            "mean": float(x.mean().item()),
            "p90": float(torch.quantile(x, 0.90).item()),
            "p95": float(torch.quantile(x, 0.95).item()),
            "max": float(x.max().item()),
            "min": float(x.min().item()),
        }




    def get_test_generator():
        while True:
            for indices, data in test_dataset_loader:
                yield indices, data

    test_generator = get_test_generator()

    def cfg_nnet(x, timesteps, context):
        eval_model = train_state.nnet_ema if use_dp else train_state.nnet
        _cond = eval_model(x, timesteps, context=context)
        _empty_context = torch.as_tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = eval_model(x, timesteps, context=_empty_context)
        return _cond

    def train_step(indices, _batch, epoch=None):
        optimizer.zero_grad(set_to_none=True)

        if use_dp:
            dp_model = train_state.dp_nnet
            dp_model.train()

            if "mask_hist" in _batch:
                batch_anchor = _batch["mask_hist"].float()
            elif "mask" in _batch:
                batch_anchor = _batch["mask"].float()
            else:
                batch_anchor = torch.ones(indices.size(0), 1, device=indices.device)

            total_loss = dp_model(
                batch_anchor,
                indices=indices,
                epoch=epoch,
                **_batch,
            )
            total_loss.backward()

            target = dp_model._module if hasattr(dp_model, "_module") else dp_model

            ###########################
            # ===== 新增：per-view / per-patient gradient 诊断 =====
            base_clip = float(config['train']['dp_max_grad_norm'])
            view_groups = _group_param_names_by_view(target)

            view_grad_stats = {}
            for view_name in ["conditions", "procedures", "drugs"]:
                stats_v = _compute_sample_grad_stats_by_name_subset(
                    target,
                    view_groups[view_name],
                    max_grad_norm=base_clip,
                )
                if stats_v is not None:
                    view_grad_stats[view_name] = stats_v

            rare_drug_mask = _mark_rare_drug_patients(_batch["drugs"])

            if train_state.step % config['train']['log_interval'] == 0:
                for view_name, stats_v in view_grad_stats.items():
                    print(
                        f"[DP VIEW GRAD STATS] {view_name} "
                        f"mean={stats_v['mean']:.6f} "
                        f"p95={stats_v['p95']:.6f} "
                        f"max={stats_v['max']:.6f} "
                        f"clip_ratio={stats_v['clip_ratio']:.6f}"
                    )

                if "drugs" in view_grad_stats:
                    drug_norms = view_grad_stats["drugs"]["sample_norms"]
                    rare_vals = drug_norms[rare_drug_mask]
                    common_vals = drug_norms[~rare_drug_mask]

                    if rare_vals.numel() > 0 and common_vals.numel() > 0:
                        print(
                            f"[RARE DRUG GRAD] "
                            f"rare_mean={rare_vals.mean().item():.6f}, "
                            f"common_mean={common_vals.mean().item():.6f}, "
                            f"rare_max={rare_vals.max().item():.6f}, "
                            f"common_max={common_vals.max().item():.6f}"
                        )
            ###########################

            inactive_params = []
            missing_grad_sample = []

            for name, p in target.named_parameters():
                if not p.requires_grad:
                    continue

                has_grad = (p.grad is not None)
                has_grad_sample = (getattr(p, "grad_sample", None) is not None)

                if (not has_grad) and (not has_grad_sample):
                    inactive_params.append(name)
                elif has_grad and (not has_grad_sample):
                    missing_grad_sample.append(name)

            if inactive_params:
                print("[DP] inactive params on this batch (show first 50):", inactive_params[:50])

            if missing_grad_sample:
                print("[DP] params with grad but no grad_sample (show first 50):", missing_grad_sample[:50])
                debug_bad_gradsample_modules(target, missing_grad_sample[:50])
                raise RuntimeError(
                    "These params have grad but no grad_sample. This is a true DP compatibility problem."
                )

            grad_stats = _compute_sample_grad_stats(
                target,
                float(config['train']['dp_max_grad_norm'])
            )
            # if grad_stats is not None and train_state.step % config['train']['log_interval'] == 0:
            #     print("[DP GRAD STATS]", grad_stats)
            #     wandb.log({f"dp/{k}": v for k, v in grad_stats.items()}, step=train_state.step)

            optimizer.step()
            did_update = not getattr(optimizer, "_is_last_step_skipped", False)

            if did_update:
                lr_scheduler.step()
                train_state.ema_update(config.get('ema_rate', 0.9999))
                train_state.step += 1

            # metrics = {
            #     "loss_mse": float(target.last_metrics["loss_mse"]),
            #     "loss_c": float(target.last_metrics["loss_c"]),
            #     "loss_rounding": float(target.last_metrics["loss_rounding"]),
            #     "T_loss": float(target.last_metrics["T_loss"]),
            #     "total_loss": float(target.last_metrics["total_loss"]),
            #     "lr": float(optimizer.param_groups[0]['lr']),
            # }
            metrics = {
                "loss_mse": float(target.last_metrics["loss_mse"]),
                "loss_c": float(target.last_metrics["loss_c"]),
                "loss_rounding": float(target.last_metrics["loss_rounding"]),
                "T_loss": float(target.last_metrics["T_loss"]),
                # "loss_view_bce": float(target.last_metrics["loss_view_bce"]),
                # "loss_view_jacc": float(target.last_metrics["loss_view_jacc"]),
                "total_loss": float(target.last_metrics["total_loss"]),
                "lr": float(optimizer.param_groups[0]['lr']),
            }

            # return target.last_z, metrics, did_update
            grad_diag = {
                "view_grad_stats": view_grad_stats,
                "rare_drug_mask": rare_drug_mask.detach().cpu(),
            }
            return target.last_z, metrics, did_update, grad_diag

        else:
            _z, _context, neigh_loss, _, _ = nnet.encode_everything(indices, **_batch)

            loss_mse, loss_round, T_loss = utils.LSimple(
                _z, nnet, _schedule, context=_context, **_batch
            )

            if epoch is not None and int(epoch) > config['warm']:
                total_loss = (
                        loss_mse.mean()
                        + neigh_loss.mean()
                        + loss_round.mean()
                        + T_loss.mean()
                )
            else:
                total_loss = (
                        loss_mse.mean()
                        + loss_round.mean()
                        + T_loss.mean()
                )

            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_state.ema_update(config.get('ema_rate', 0.9999))
            train_state.step += 1

            metrics = {
                "loss_mse": float(loss_mse.detach().mean()),
                "loss_c": float(neigh_loss.detach().mean()),
                "loss_rounding": float(loss_round.detach().mean()),
                "T_loss": float(T_loss.detach().mean()),
                "total_loss": float(total_loss.detach()),
                "lr": float(optimizer.param_groups[0]['lr']),
            }

            # return _z.detach(), metrics, True
            return _z.detach(), metrics, True, None

    def eval_step(n_samples, sample_steps):
        logging.info(
            f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, '
            f'algorithm=dpm_solver, mini_batch_size={config["sample"]["mini_batch_size"]}'
        )
        _disable_dp_hooks()
        try:
            with torch.no_grad():
                def sample_fn(_n_samples):
                    indices, batch = next(test_generator)

                    assert indices.size(0) == _n_samples
                    indices, sample_input = convert_input((indices, batch), mask_static=True)

                    eval_model = train_state.nnet_ema if use_dp else train_state.nnet
                    # eval_model = train_state.nnet
                    _z, _context_emb, _, labels, hard_labels = eval_model.encode_everything_test(indices, **batch)

                    return labels, hard_labels, best_of_k_sample(
                        _n_samples,
                        sample_steps,
                        context=_context_emb,
                        z_init=_z,
                        mask=sample_input['mask'],
                    )

                with tempfile.TemporaryDirectory() as temp_path:
                    path = config['sample']['path'] + data_n + '/' + task or temp_path
                    work_dir = config['workdir'] + data_n + '/' + task
                    os.makedirs(path, exist_ok=True)
                    os.makedirs(work_dir, exist_ok=True)

                    utils.sample2dir(
                        path,
                        n_samples,
                        config['sample']['mini_batch_size'],
                        sample_fn,
                        tokenizers,
                        dataset.unpreprocess
                    )
                    _quality, jaccard_quality = utils.calculate_quality(path)
                    logging.info(f'step={train_state.step} quality-{n_samples}={_quality}')
                    logging.info(f'step={train_state.step} jaccard_quality-{n_samples}={jaccard_quality}')
                    with open(os.path.join(config['workdir'] + data_n + '/' + task, 'eval.log'), 'a') as f:
                        print(
                            f'step={train_state.step} quality-{n_samples}={_quality} '
                            f'jaccard_quality-{n_samples}={jaccard_quality}',
                            file=f
                        )
                    wandb.log({f'quality_mse-{n_samples}': _quality}, step=train_state.step)
                    _quality = torch.tensor(_quality, device=device)
            return _quality.item(), jaccard_quality
        finally:
            _enable_dp_hooks()

    logging.info(f'Start fitting, step={train_state.step}')

    def generate_fixed_view_mask(batch_size, view_name, device):
        """
        生成固定单视图缺失 mask
        约定顺序: [conditions, procedures, drugs]
        1 表示该视图需要生成/补全
        0 表示该视图保留原始输入
        """
        view2idx = {
            "conditions": 0,
            "procedures": 1,
            "drugs": 2,
        }
        mask = torch.zeros(batch_size, 3, dtype=torch.bool, device=device)
        if view_name is None:
            return mask
        mask[:, view2idx[view_name]] = True
        return mask

    # def get_curriculum_mask_view(epoch):
    #     if epoch < 10:
    #         return "conditions"
    #     elif epoch < 20:
    #         return "procedures"
    #     elif epoch < 30:
    #         return "drugs"
    #     else:
    #         return None
    def convert_input(batch_data, mask_static=True):
        indices, batch = batch_data

        feature_keys = ['conditions', 'drugs', 'procedures']
        for feature_key in feature_keys:
            batch[feature_key] = tokenizers[feature_key].batch_encode_2d(batch[feature_key])
            batch[feature_key + '_hist'] = tokenizers[feature_key].batch_encode_3d(batch[feature_key + '_hist'])
            origin = batch[feature_key + '_comp'].copy()
            batch[feature_key + '_comp'] = tokenizers[feature_key].batch_encode_2d(batch[feature_key + '_comp'])

            batch[feature_key] = torch.tensor(batch[feature_key], dtype=torch.long, device=device)
            batch[feature_key + '_hist'] = torch.tensor(batch[feature_key + '_hist'], dtype=torch.long, device=device)
            batch[feature_key + '_comp'] = torch.tensor(batch[feature_key + '_comp'], dtype=torch.long, device=device)

            batch[feature_key + '_mask'] = batch[feature_key] != 0
            batch[feature_key + '_hist_mask'] = batch[feature_key + '_hist'] != 0
            batch[feature_key + '_comp_mask'] = batch[feature_key + '_comp'] != 0

            batch[feature_key + '_comps'] = tokenizers[feature_key].batch_encode_2d(
                origin, padding=False, truncation=False
            )
            batch[feature_key + '_comps_origin'] = origin
            batch['has_note'] = False

        # if mask_static:
        #     batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
        # else:
        #     m = 3
        #     batch['mask'] = utils.generate_mask(batch['conditions'].shape[0], m=m).to(device=device)
        if mask_static:
            batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
        else:
            fixed_mask_view = config.get("train", {}).get("fixed_mask_view", None)
            if fixed_mask_view is None:
                m = 3
                batch['mask'] = utils.generate_mask(batch['conditions'].shape[0], m=m).to(device=device)
            else:
                batch['mask'] = generate_fixed_view_mask(
                    batch_size=batch['conditions'].shape[0],
                    view_name=fixed_mask_view,
                    device=device,
                )

        aligned_data = [
            batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1)
            for feature_key in feature_keys
        ]
        batch['mask_hist'] = torch.cat(aligned_data, dim=-1)

        return indices, batch
    # def convert_input(batch_data, mask_static=True, fixed_mask_view=None):
    #     indices, batch = batch_data
    #
    #     feature_keys = ['conditions', 'drugs', 'procedures']
    #     for feature_key in feature_keys:
    #         batch[feature_key] = tokenizers[feature_key].batch_encode_2d(batch[feature_key])
    #         batch[feature_key + '_hist'] = tokenizers[feature_key].batch_encode_3d(batch[feature_key + '_hist'])
    #         origin = batch[feature_key + '_comp'].copy()
    #         batch[feature_key + '_comp'] = tokenizers[feature_key].batch_encode_2d(batch[feature_key + '_comp'])
    #
    #         batch[feature_key] = torch.tensor(batch[feature_key], dtype=torch.long, device=device)
    #         batch[feature_key + '_hist'] = torch.tensor(batch[feature_key + '_hist'], dtype=torch.long, device=device)
    #         batch[feature_key + '_comp'] = torch.tensor(batch[feature_key + '_comp'], dtype=torch.long, device=device)
    #
    #         batch[feature_key + '_mask'] = batch[feature_key] != 0
    #         batch[feature_key + '_hist_mask'] = batch[feature_key + '_hist'] != 0
    #         batch[feature_key + '_comp_mask'] = batch[feature_key + '_comp'] != 0
    #
    #         batch[feature_key + '_comps'] = tokenizers[feature_key].batch_encode_2d(
    #             origin, padding=False, truncation=False
    #         )
    #         batch[feature_key + '_comps_origin'] = origin
    #         batch['has_note'] = False
    #
    #     # if mask_static:
    #     #     batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
    #     # else:
    #     #     m = 3
    #     #     batch['mask'] = utils.generate_mask(batch['conditions'].shape[0], m=m).to(device=device)
    #     if mask_static:
    #         batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
    #     else:
    #         if fixed_mask_view is None:
    #             m = 3
    #             batch['mask'] = utils.generate_mask(batch['conditions'].shape[0], m=m).to(device=device)
    #         else:
    #             batch['mask'] = generate_fixed_view_mask(
    #                 batch_size=batch['conditions'].shape[0],
    #                 view_name=fixed_mask_view,
    #                 device=device,
    #             )
    #
    #     aligned_data = [
    #         batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1)
    #         for feature_key in feature_keys
    #     ]
    #     batch['mask_hist'] = torch.cat(aligned_data, dim=-1)
    #
    #     return indices, batch


    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=config['sample']['mini_batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_dict
    )

    def get_train_eval_generator():
        while True:
            for indices, data in train_eval_loader:
                yield indices, data

    train_eval_generator = get_train_eval_generator()

    def train_epoch_metrics(sample_steps, num_batches=2, mask_static=True):
        eval_model = train_state.nnet_ema if use_dp else train_state.nnet
        eval_model.eval()

        mse_list = []
        jc_list, jp_list, jd_list = [], [], []

        with torch.no_grad():
            for _ in range(num_batches):
                idx, raw_b = next(train_eval_generator)
                idx, b = convert_input((idx, raw_b), mask_static=mask_static)

                z0, contexts, _, labels, hard_labels = eval_model.encode_everything_test(idx, **raw_b)

                samples, hard_samples = best_of_k_sample(
                    _n_samples=idx.size(0),
                    _sample_steps=sample_steps,
                    context=contexts,
                    z_init=z0,
                    mask=b['mask']
                )

                mse_list.append(F.mse_loss(samples, labels.squeeze()).item())

                gt_cond = tokenizers['conditions'].batch_decode_2d(hard_labels['conditions'])
                gt_proc = tokenizers['procedures'].batch_decode_2d(hard_labels['procedures'])
                gt_drug = tokenizers['drugs'].batch_decode_2d(hard_labels['drugs'])

                jc_list.append(utils.calculate_average_jaccard(hard_samples['conditions'], gt_cond))
                jp_list.append(utils.calculate_average_jaccard(hard_samples['procedures'], gt_proc))
                jd_list.append(utils.calculate_average_jaccard(hard_samples['drugs'], gt_drug))

        train_mse = float(np.mean(mse_list))
        train_jacc = (float(np.mean(jc_list)), float(np.mean(jp_list)), float(np.mean(jd_list)))
        return train_mse, train_jacc

    def export_filtered_synth_dataset(sample_steps):
        _disable_dp_hooks()
        synth_rows = []

        try:
            eval_model = train_state.nnet_ema if use_dp else train_state.nnet
            eval_model.eval()

            synth_loader = DataLoader(
                train_dataset,
                batch_size=config['sample']['mini_batch_size'],
                shuffle=False,
                drop_last=False,
                num_workers=1,
                pin_memory=True,
                persistent_workers=True,
                collate_fn=collate_fn_dict
            )

            with torch.no_grad():
                for indices, raw_batch in synth_loader:
                    indices, batch = convert_input((indices, raw_batch), mask_static=True)

                    z0, contexts, _, labels, hard_labels = eval_model.encode_everything_test(indices, **raw_batch)

                    samples, hard_samples, sample_scores = best_of_k_sample(
                        _n_samples=indices.size(0),
                        _sample_steps=sample_steps,
                        context=contexts,
                        z_init=z0,
                        mask=batch['mask'],
                        return_scores=True,
                    )

                    for b in range(indices.size(0)):
                        row = {}

                        for k, v in raw_batch.items():
                            if k in ['conditions', 'procedures', 'drugs']:
                                continue
                            row[k] = _json_safe(_pick_batch_item(v, b))

                        row['conditions'] = _json_safe(hard_samples['conditions'][b])
                        row['procedures'] = _json_safe(hard_samples['procedures'][b])
                        row['drugs'] = _json_safe(hard_samples['drugs'][b])
                        row['synth_score'] = float(sample_scores[b].item())
                        row['source_index'] = int(indices[b].item())
                        row['is_synth'] = 1
                        synth_rows.append(row)
        finally:
            _enable_dp_hooks()

        uniq = {}
        for row in synth_rows:
            sig = _sample_signature(row['conditions'], row['procedures'], row['drugs'])
            if sig not in uniq or row['synth_score'] > uniq[sig]['synth_score']:
                uniq[sig] = row
        synth_rows = list(uniq.values())

        if bool(config.get("sample", {}).get("balance_by_label", False)):
            synth_rows = _keep_top_by_label(synth_rows)
        else:
            synth_rows = sorted(synth_rows, key=lambda x: x['synth_score'], reverse=True)
            keep_ratio = float(config.get("sample", {}).get("keep_ratio", 0.35))
            keep_n = max(1, int(len(synth_rows) * keep_ratio))
            synth_rows = synth_rows[:keep_n]

        out_dir = os.path.join(config['workdir'], data_n, task)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"synth_only_filtered_{exp_num}.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(synth_rows, f, ensure_ascii=False)

        logging.info(f"[SYNTH] saved filtered synthetic dataset: {out_path}, n={len(synth_rows)}")
        return out_path, len(synth_rows)

    step_quality = []
    jaccard_quality = []

    best_jacc_so_far = -1.0
    best_quality_so_far = float("inf")
    best_epoch = None
    best_ckpt_dir = None
    best_quality_value = None
    best_jacc_value = None
    grad_diag_cache = {
        "conditions": [],
        "procedures": [],
        "drugs": [],
        "drug_rare": [],
        "drug_common": [],
    }
    dp_epoch_grad_cache = {
        "conditions": [],
        "procedures": [],
        "drugs": [],
    }

    for epoch in range(config['train']['n_epochs']):
        if epoch == 0:
            train_state.nnet.e_step()

        total_visit_embs = []
        total_indices = []

        # 统计：本 epoch 走了多少个物理 step、多少个真实优化 step
        physical_steps_epoch = 0
        true_updates_epoch = 0

        dp_epoch_grad_cache = {
            "conditions": [],
            "procedures": [],
            "drugs": [],
            "drug_rare": [],
            "drug_common": [],
        }

        if use_dp:
            dp_loss_meter = {
                "loss_mse": 0.0,
                "loss_c": 0.0,
                "loss_rounding": 0.0,
                "T_loss": 0.0,
                "loss_view_bce": 0.0,
                "loss_view_jacc": 0.0,
                "total_loss": 0.0,
            }
            dp_phys_cnt = 0

        # ## BatchMemoryManager 打开，在做逻辑 batch 大、物理 batch 小的 DP 训练
        # if use_dp and int(config['train'].get('dp_physical_batch_size', 0)) > 0:
        #     from opacus.utils.batch_memory_manager import BatchMemoryManager
        #     loader_cm = BatchMemoryManager(
        #         data_loader=train_dataset_loader,
        #         max_physical_batch_size=int(config['train']['dp_physical_batch_size']),
        #         optimizer=optimizer,
        #     )
        # else:
        #     loader_cm = nullcontext(train_dataset_loader)

        ## 当前
        loader_cm = nullcontext(train_dataset_loader)

        dp_loss_meter = {
            'loss_mse': 0.0,
            'loss_c': 0.0,
            'loss_rounding': 0.0,
            'T_loss': 0.0,
            'total_loss': 0.0,
        }
        dp_phys_cnt = 0

        with loader_cm as epoch_loader:
            for indices, raw_batch in epoch_loader:
                physical_steps_epoch += 1
                nnet.train()

                # static_mask_epochs = int(config.get("train", {}).get("static_mask_epochs", 8))
                # use_static_mask = bool(epoch < static_mask_epochs)
                #
                # indices, batch = convert_input((indices, raw_batch), mask_static=use_static_mask)
                static_mask_epochs = int(config.get("train", {}).get("static_mask_epochs", 0))
                force_static_mask = bool(config.get("train", {}).get("force_static_mask", False))
                use_static_mask = force_static_mask or bool(epoch < static_mask_epochs)

                indices, batch = convert_input((indices, raw_batch), mask_static=use_static_mask)
                # curr_fixed_view = get_curriculum_mask_view(epoch)
                # indices, batch = convert_input(
                #     (indices, raw_batch),
                #     mask_static=use_static_mask,
                #     fixed_mask_view=curr_fixed_view
                # )


                # _z, metrics, did_update = train_step(indices, batch, epoch)
                _z, metrics, did_update, grad_diag = train_step(indices, batch, epoch)
                if use_dp and grad_diag is not None:
                    vstats = grad_diag["view_grad_stats"]
                    for view_name in ["conditions", "procedures", "drugs"]:
                        if view_name in vstats:
                            dp_epoch_grad_cache[view_name].append(
                                vstats[view_name]["sample_norms"].detach().cpu()
                            )
                if grad_diag is not None:
                    vstats = grad_diag["view_grad_stats"]
                    rare_mask = grad_diag["rare_drug_mask"]
                    for view_name in ["conditions", "procedures", "drugs"]:
                        if view_name in vstats:
                            grad_diag_cache[view_name].append(vstats[view_name]["sample_norms"].detach().cpu())
                    if "drugs" in vstats:
                        dn = vstats["drugs"]["sample_norms"].detach().cpu()
                        grad_diag_cache["drug_rare"].append(dn[rare_mask])
                        grad_diag_cache["drug_common"].append(dn[~rare_mask])



                if did_update:
                    true_updates_epoch += 1

                if use_dp:
                    for k in dp_loss_meter:
                        dp_loss_meter[k] += float(metrics[k])
                    dp_phys_cnt += 1
                    if not did_update:
                        continue
                    metrics = {
                        k: dp_loss_meter[k] / max(dp_phys_cnt, 1)
                        for k in dp_loss_meter
                    }
                    metrics['lr'] = float(optimizer.param_groups[0]['lr'])
                    dp_loss_meter = {k: 0.0 for k in dp_loss_meter}
                    dp_phys_cnt = 0

                total_visit_embs.append(_z.view(_z.shape[0], -1))
                total_indices.append(indices)

                nnet.eval()

                if train_state.step % config['train']['log_interval'] == 0:
                    logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
                    wandb.log(metrics, step=train_state.step)

                    step_history['step'].append(int(train_state.step))
                    step_history['loss_mse'].append(float(metrics['loss_mse']))
                    step_history['loss_c'].append(float(metrics['loss_c']))
                    step_history['loss_rounding'].append(float(metrics['loss_rounding']))
                    step_history['T_loss'].append(float(metrics['T_loss']))
                    step_history['total_loss'].append(float(metrics['total_loss']))
                    step_history['lr'].append(float(metrics['lr']))

                    loss_writer.writerow([
                        int(train_state.step),
                        float(metrics["loss_mse"]),
                        float(metrics["loss_c"]),
                        float(metrics["loss_rounding"]),
                        float(metrics["T_loss"]),
                        float(metrics["total_loss"]),
                        float(metrics["lr"]),
                    ])
                    loss_f.flush()

        msg = (
            f"[UPDATE STATS] epoch={epoch} "
            f"physical_steps={physical_steps_epoch} "
            f"true_updates={true_updates_epoch} "
            f"update_ratio={true_updates_epoch / max(physical_steps_epoch, 1):.6f}"
        )

        print(msg)
        logging.info(msg)

        ##################
        if use_dp:
            dp_epoch_grad_all = {}
            for view_name in ["conditions", "procedures", "drugs"]:
                if len(dp_epoch_grad_cache[view_name]) > 0:
                    dp_epoch_grad_all[view_name] = torch.cat(dp_epoch_grad_cache[view_name], dim=0)
                else:
                    dp_epoch_grad_all[view_name] = torch.empty(0, dtype=torch.float32)

            for view_name in ["conditions", "procedures", "drugs"]:
                s = summarize_grad_tensor(dp_epoch_grad_all[view_name])
                if s is not None:
                    print(
                        f"[DP EPOCH VIEW GRAD] epoch={epoch} {view_name} "
                        f"mean={s['mean']:.6f} p95={s['p95']:.6f} max={s['max']:.6f}"
                    )

            work_dir = os.path.join(config['workdir'], data_n, task)
            os.makedirs(work_dir, exist_ok=True)
            cdf_path = os.path.join(work_dir, f"{data_n}-{task}-{exp_num}-epoch{epoch}-dp-gradnorm-cdf.png")

            _plot_cdf(
                {
                    "dp_conditions": dp_epoch_grad_all["conditions"],
                    "dp_procedures": dp_epoch_grad_all["procedures"],
                    "dp_drugs": dp_epoch_grad_all["drugs"],
                },
                cdf_path,
                title=f"Epoch {epoch} DP per-view grad norm CDF",
            )
            print(f"[DP GRAD CDF SAVED] {cdf_path}")

        if not use_dp:
            nondp_epoch_grad_all = collect_nondp_view_grad_norms_full_epoch(
                model=train_state.nnet,
                data_loader=train_dataset_loader,
                schedule=_schedule,
                epoch=epoch,
                warm=config.get("warm", 10),
                mask_static=True,
            )

            for view_name in ["conditions", "procedures", "drugs"]:
                s = summarize_grad_tensor(nondp_epoch_grad_all[view_name])
                if s is not None:
                    print(
                        f"[NONDP EPOCH VIEW GRAD] epoch={epoch} {view_name} "
                        f"mean={s['mean']:.6f} p95={s['p95']:.6f} max={s['max']:.6f}"
                    )

            work_dir = os.path.join(config['workdir'], data_n, task)
            os.makedirs(work_dir, exist_ok=True)
            cdf_path = os.path.join(work_dir, f"{data_n}-{task}-{exp_num}-epoch{epoch}-nondp-gradnorm-cdf.png")

            _plot_cdf(
                {
                    "nondp_conditions": nondp_epoch_grad_all["conditions"],
                    "nondp_procedures": nondp_epoch_grad_all["procedures"],
                    "nondp_drugs": nondp_epoch_grad_all["drugs"],
                },
                cdf_path,
                title=f"Epoch {epoch} non-DP per-view grad norm CDF",
            )
            print(f"[NON-DP GRAD CDF SAVED] {cdf_path}")

        ##################

        wandb.log({
            "dp_epoch/physical_steps": physical_steps_epoch,
            "dp_epoch/true_updates": true_updates_epoch,
            "dp_epoch/update_ratio": true_updates_epoch / max(physical_steps_epoch, 1),
        }, step=train_state.step)

        train_mse_ep, train_jacc_ep = train_epoch_metrics(
            sample_steps=config['sample']['sample_steps'],
            num_batches=config['train'].get('train_eval_batches', 2),
            mask_static=True
        )

        logging.info(
            f"[UPDATE STATS] epoch={epoch} "
            f"physical_steps={physical_steps_epoch} "
            f"true_updates={true_updates_epoch} "
            f"update_ratio={true_updates_epoch / max(physical_steps_epoch, 1):.6f}"
        )
        wandb.log({
            "dp_epoch/physical_steps": physical_steps_epoch,
            "dp_epoch/true_updates": true_updates_epoch,
            "dp_epoch/update_ratio": true_updates_epoch / max(physical_steps_epoch, 1),
        }, step=train_state.step)

        logging.info(f'[TRAIN EVAL] epoch={epoch} train_mse={train_mse_ep} train_jacc={train_jacc_ep}')
        wandb.log({
            'train_epoch/mse': train_mse_ep,
            'train_epoch/jaccard_conditions': train_jacc_ep[0],
            'train_epoch/jaccard_procedures': train_jacc_ep[1],
            'train_epoch/jaccard_drugs': train_jacc_ep[2],
            'train_epoch/jaccard_mean': (train_jacc_ep[0] + train_jacc_ep[1] + train_jacc_ep[2]) / 3.0,
        }, step=train_state.step)

        epoch_history['epoch'].append(epoch)
        epoch_history['train_mse'].append(float(train_mse_ep))
        epoch_history['train_jaccard_conditions'].append(float(train_jacc_ep[0]))
        epoch_history['train_jaccard_procedures'].append(float(train_jacc_ep[1]))
        epoch_history['train_jaccard_drugs'].append(float(train_jacc_ep[2]))
        epoch_history['train_jaccard_mean'].append(
            float((train_jacc_ep[0] + train_jacc_ep[1] + train_jacc_ep[2]) / 3.0)
        )

        # if use_dp and config['train'].get('use_full_centroid_refresh', False):
        #     full_centroid_refresh(epoch)
        if use_dp and config['train'].get('use_full_centroid_refresh', False):
            refresh_every = int(config['train'].get('centroid_refresh_every', 3))
            stage2_start = int(config.get('warm', 10)) + 10
            if epoch >= stage2_start and epoch % refresh_every == 0:
                full_centroid_refresh(epoch)
        logging.info(f'Eval checkpoint candidate {epoch}.')
        should_eval = (
            (not use_dp and (epoch >= 0 or (epoch >= 5 and epoch % config['train']['eval_epoch'] == 0)))
            or
            (use_dp and train_state.step > 0 and (epoch >= 0 or (epoch >= 5 and epoch % config['train']['eval_epoch'] == 0)))
        )
        if should_eval:
            quality, jaccard_tuple = eval_step(
                n_samples=len(test_dataset),
                sample_steps=config['sample']['sample_steps']
            )
        else:
            quality = 10000.0
            jaccard_tuple = (1.0, 1.0, 1.0)

        step_quality.append((epoch, quality))
        jaccard_quality.append((epoch, jaccard_tuple))

        epoch_history['quality_mse'].append(float(quality))
        epoch_history['quality_jaccard_conditions'].append(float(jaccard_tuple[0]))
        epoch_history['quality_jaccard_procedures'].append(float(jaccard_tuple[1]))
        epoch_history['quality_jaccard_drugs'].append(float(jaccard_tuple[2]))
        epoch_history['quality_jaccard_mean'].append(float(np.mean(jaccard_tuple)))

        jacc_mean = float(np.mean(jaccard_tuple))
        is_better = (
            (jacc_mean > best_jacc_so_far + 1e-8) or
            (
                abs(jacc_mean - best_jacc_so_far) <= 1e-8
                and float(quality) < best_quality_so_far
            )
        )
        if is_better:
            best_jacc_so_far = jacc_mean
            best_quality_so_far = float(quality)
            best_epoch = int(epoch)
            best_quality_value = float(quality)
            best_jacc_value = tuple(float(x) for x in jaccard_tuple)
            new_best_ckpt_dir = os.path.join(ckpt_dir, f'{epoch}.ckpt')
            if os.path.exists(new_best_ckpt_dir):
                shutil.rmtree(new_best_ckpt_dir)
            logging.info(
                f'[BEST] save new best checkpoint at epoch={epoch}, '
                f'jacc_mean={jacc_mean:.6f}, quality={quality:.6f}'
            )
            train_state.save(new_best_ckpt_dir)
            for path in glob.glob(os.path.join(ckpt_dir, '*.ckpt')):
                if os.path.abspath(path) != os.path.abspath(new_best_ckpt_dir):
                    shutil.rmtree(path)
            best_ckpt_dir = new_best_ckpt_dir
        # jc, jp, jd = [float(x) for x in jaccard_tuple]
        #
        # # 弱路优先：conditions / procedures 权重大一些
        # jacc_balanced = 0.4 * jc + 0.4 * jp + 0.2 * jd
        #
        # is_better = (
        #         (jacc_balanced > best_jacc_so_far + 1e-8) or
        #         (
        #                 abs(jacc_balanced - best_jacc_so_far) <= 1e-8
        #                 and float(quality) < best_quality_so_far
        #         )
        # )
        #
        # if is_better:
        #     best_jacc_so_far = jacc_balanced
        #     best_quality_so_far = float(quality)
        #     best_epoch = int(epoch)
        #     best_quality_value = float(quality)
        #     best_jacc_value = (jc, jp, jd)
        #
        #     new_best_ckpt_dir = os.path.join(ckpt_dir, f'{epoch}.ckpt')
        #     if os.path.exists(new_best_ckpt_dir):
        #         shutil.rmtree(new_best_ckpt_dir)
        #
        #     logging.info(
        #         f'[BEST] save new best checkpoint at epoch={epoch}, '
        #         f'jacc_balanced={jacc_balanced:.6f}, quality={quality:.6f}, '
        #         f'jc={jc:.6f}, jp={jp:.6f}, jd={jd:.6f}'
        #     )
        #
        #     train_state.save(new_best_ckpt_dir)
        #
        #     for path in glob.glob(os.path.join(ckpt_dir, '*.ckpt')):
        #         if os.path.abspath(path) != os.path.abspath(new_best_ckpt_dir):
        #             shutil.rmtree(path)
        #
        #     best_ckpt_dir = new_best_ckpt_dir



        # if len(total_visit_embs) > 0:
        #     total_visit_embs = torch.cat(total_visit_embs, dim=0)
        #     total_indices = torch.cat(total_indices, dim=0)
        #     total_visit_embs = total_visit_embs[total_indices.argsort()]
        #     print("CCCCCCCC", total_visit_embs.shape)
        #
        #     train_state.nnet.update_train_visit_emb(total_visit_embs)
        #     visit_centroids, _ = train_state.nnet.e_step()
        #     train_state.nnet.update_centroid_emb(visit_centroids)
        if use_dp:
            visit_centroids, _ = train_state.nnet.e_step()

            if hasattr(train_state.nnet, "visit_centroids") and train_state.nnet.visit_centroids is not None:
                old_centroids = train_state.nnet.visit_centroids.detach()
                centroid_ema = float(config['train'].get('centroid_ema', 0.8))
                visit_centroids = centroid_ema * old_centroids + (1.0 - centroid_ema) * visit_centroids

            train_state.nnet.update_centroid_emb(visit_centroids)

        if not use_dp:
            total_visit_embs = torch.cat(total_visit_embs, dim=0)
            total_indices = torch.cat(total_indices, dim=0)
            total_visit_embs = total_visit_embs[total_indices.argsort()]
            print("CCCCCCCC", total_visit_embs.shape)

            train_state.nnet.update_train_visit_emb(total_visit_embs)
            visit_centroids, _ = train_state.nnet.e_step()
            train_state.nnet.update_centroid_emb(visit_centroids)


        if use_dp and privacy_engine is not None:
            eps = privacy_engine.get_epsilon(float(config['train']['dp_delta']))
            logging.info(
                f"[DP] epoch={epoch} epsilon={eps:.6f} delta={config['train']['dp_delta']}"
            )
            wandb.log({"dp/epsilon": eps}, step=train_state.step)

    def save_history_csv():
        work_dir = os.path.join(config['workdir'], data_n, task)
        os.makedirs(work_dir, exist_ok=True)

        step_csv = os.path.join(work_dir, f'{data_n}-{task}-{exp_num}-step_history.csv')
        epoch_csv = os.path.join(work_dir, f'{data_n}-{task}-{exp_num}-epoch_history.csv')

        with open(step_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(step_history.keys())
            rows = zip(*step_history.values())
            writer.writerows(rows)

        with open(epoch_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(epoch_history.keys())
            rows = zip(*epoch_history.values())
            writer.writerows(rows)

    def plot_history():
        work_dir = os.path.join(config['workdir'], data_n, task)
        os.makedirs(work_dir, exist_ok=True)

        def clean_xy(x, y):
            x2, y2 = [], []
            for a, b in zip(x, y):
                if isinstance(a, torch.Tensor):
                    a = a.detach().cpu().item() if a.numel() == 1 else None
                if isinstance(b, torch.Tensor):
                    b = b.detach().cpu().item() if b.numel() == 1 else None
                if a is None or b is None:
                    continue
                try:
                    a = float(a)
                    b = float(b)
                except Exception:
                    continue
                if np.isfinite(a) and np.isfinite(b):
                    x2.append(a)
                    y2.append(b)
            return np.asarray(x2, dtype=np.float64), np.asarray(y2, dtype=np.float64)

        plt.figure(figsize=(10, 6))
        for key in ['loss_mse', 'loss_c', 'loss_rounding', 'T_loss', 'total_loss']:
            x, y = clean_xy(step_history['step'], step_history[key])
            if len(x) > 0:
                plt.plot(x, y, label=key)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, f'{data_n}-{task}-{exp_num}-step_losses.png'), dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        x, y = clean_xy(step_history['step'], step_history['lr'])
        if len(x) > 0:
            plt.plot(x, y, label='lr')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, f'{data_n}-{task}-{exp_num}-lr.png'), dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        for key in ['quality_mse', 'train_mse']:
            x, y = clean_xy(epoch_history['epoch'], epoch_history[key])
            if len(x) > 0:
                plt.plot(x, y, label=key)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, f'{data_n}-{task}-{exp_num}-epoch_mse.png'), dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        for key in [
            'train_jaccard_conditions',
            'train_jaccard_procedures',
            'train_jaccard_drugs',
            'train_jaccard_mean',
            'quality_jaccard_conditions',
            'quality_jaccard_procedures',
            'quality_jaccard_drugs',
            'quality_jaccard_mean',
        ]:
            x, y = clean_xy(epoch_history['epoch'], epoch_history[key])
            if len(x) > 0:
                plt.plot(x, y, label=key)
        plt.xlabel('Epoch')
        plt.ylabel('Jaccard')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, f'{data_n}-{task}-{exp_num}-epoch_jaccard.png'), dpi=150)
        plt.close()

    save_history_csv()
    try:
        plot_history()
    except Exception as e:
        logging.info(f"[plot_history failed] {e}")

    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_quality: {step_quality}')
    logging.info(f'jaccard_quality: {jaccard_quality}')

    if best_epoch is None or best_ckpt_dir is None:
        loss_f.close()
        raise RuntimeError("No best checkpoint was saved during training.")
    logging.info(f'step_best: {best_epoch}')

    if best_quality_value is None or best_jacc_value is None:
        loss_f.close()
        raise RuntimeError("Best metric values were not recorded during training.")

    best_quality = best_quality_value
    best_jacc = best_jacc_value

    if use_dp and privacy_engine is not None:
        final_epsilon = float(privacy_engine.get_epsilon(float(config['train']['dp_delta'])))
    else:
        final_epsilon = float("nan")

    train_state.load(best_ckpt_dir)
    eval_step(
        n_samples=len(test_dataset),
        sample_steps=config['sample']['sample_steps']
    )

    synth_dataset_path, synth_dataset_count = export_filtered_synth_dataset(
        sample_steps=config['sample']['sample_steps']
    )

    nnet_path = os.path.join(best_ckpt_dir, 'nnet.pth')
    generation_csv = os.path.join(
        config['workdir'],
        data_n,
        task,
        "generation_dp_summary.csv"
    )
    append_row_to_csv(
        generation_csv,
        {
            "exp_num": str(exp_num),
            "target_epsilon": float(config['train']['dp_target_epsilon']),
            "actual_epsilon": final_epsilon,
            "best_epoch": int(best_epoch),
            "best_quality_mse": float(best_quality),
            "best_jaccard_conditions": float(best_jacc[0]),
            "best_jaccard_procedures": float(best_jacc[1]),
            "best_jaccard_drugs": float(best_jacc[2]),
            "best_jaccard_mean": float(np.mean(best_jacc)),
            "ckpt_dir": ckpt_dir,
            "best_ckpt_dir": best_ckpt_dir,
            "nnet_path": nnet_path,
            "synth_dataset_path": synth_dataset_path,
            "synth_dataset_count": int(synth_dataset_count),
            "keep_ratio": float(config.get("sample", {}).get("keep_ratio", 0.35)),
            "best_of_k": int(config.get("sample", {}).get("best_of_k", 1)),
        }
    )

    train_state.synth_dataset_path = synth_dataset_path
    train_state.synth_dataset_count = synth_dataset_count

    loss_f.close()

    return nnet_path, train_state.nnet, train_state.nnet.total_visit_emb.shape[0]