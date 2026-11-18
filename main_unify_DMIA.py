# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : main.py
# Time       ：23/10/2024 4:55 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：# 为HyperHealth, diag
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import time
import torch

from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset, eICUDataset, OMOPDataset # SampleEHRDataset
from pyhealth.datasets import SampleEHRDataset as SampleEHRDatasets
# from data import SampleEHRDatasetSIMPLE as SampleEHRDataset #
# from pic_parse import PICDataset
# from omix_parse import OMIXDataset
from utils import split_by_patient, load_pickle, save_pickle, set_random_seed, get_tokenizers
from loader import get_dataloader, get_special_input
from data import convert_dataset
from diffuser.diff_config import config as diff_config
from config import config
from train_dffs import run_diffusion, run_diffusion1
# from eval_dffs import augment_dataset, load_diffusion, evaluate_diff
from eval_dffs import augment_dataset, load_diffusion, evaluate_diff, eval_diffusion_latent_mae_rmse, eval_diffusion_latent_mae_rmse20, eval_diffusion_latent_mae_rmse20all, eval_diffusion_latent_mae_rmse_allviews, plot_latent_sq_error_density, plot_sq_error_hist_for_scan
from train_hp import run_pcf, run_pretrain, load_pretrain_model, get_task_fn, add_imp_fields
from baseline_utils import missing_statistics
from data import generate_patient_group
set_random_seed(config['SEED'])
# from shot_parse import SHOTDataset
from diffuser.diff_dataset import get_dataset, collate_fn_dict

import copy
import numpy as np
import csv
import os, time, json
import pandas as pd


import math
import random
from typing import Any, Dict, Optional, Tuple, Union, Sequence
from sklearn.metrics import roc_auc_score, roc_curve



############################# ALL Views ###############################
FEATURE_ORDER = ('conditions', 'procedures', 'drugs')  # 与三视图顺序一致

############################# Diffusion MIA ###############################
# =========================
# SecMI-stat (MMHealth / MMHealth-main compatible)
# =========================
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def convert_input(batch_data, mask_static=True):
    # padding到相同的长度便于转为tensor
    indices, batch = batch_data
    # print("batch_data 0", batch['mask'])

    feature_keys = ['conditions', 'drugs', 'procedures']
    # batch_new = {}
    for feature_key in feature_keys:
        # tokenizers
        batch[feature_key] = tokenizers[feature_key].batch_encode_2d(batch[feature_key])  # B,M,  m——feature没有用。
        batch[feature_key + '_hist'] = tokenizers[feature_key].batch_encode_3d(batch[feature_key + '_hist'])  # B,V,M
        origin = batch[feature_key + '_comp'].copy()
        batch[feature_key + '_comp'] = tokenizers[feature_key].batch_encode_2d(batch[feature_key + '_comp'])  # B,M
        # tensor
        batch[feature_key] = torch.tensor(batch[feature_key], dtype=torch.long, device=device)
        batch[feature_key + '_hist'] = torch.tensor(batch[feature_key + '_hist'], dtype=torch.long, device=device)
        batch[feature_key + '_comp'] = torch.tensor(batch[feature_key + '_comp'], dtype=torch.long, device=device)
        # mask
        batch[feature_key + '_mask'] = batch[feature_key] != 0  # 为True的就是mask掉的, # B,M
        batch[feature_key + '_hist_mask'] = batch[feature_key + '_hist'] != 0  # 这个是为了padding, # B,V,M
        batch[feature_key + '_comp_mask'] = batch[feature_key + '_comp'] != 0  # 这个是为了padding # B,M

        # others
        batch[feature_key + '_comps'] = tokenizers[feature_key].batch_encode_2d(origin, padding=False,
                                                                                truncation=False)  # 不需要减去2,只需要能看清楚是什么就行，因为decode的时候会加上2
        batch[feature_key + '_comps_origin'] = origin  # origin ID

        # note special
        batch['has_note'] = False
        ## 20251210注释掉"if config_health['DATASET'] == 'MIV-Note':"
        # if config_health['DATASET'] == 'MIV-Note': # LOS MIV修改下。
        #     batch['note'] = torch.tensor(batch['note'], device=device) # cur, B,D; 这里不需要对他进行重建，因为他可能变为pad字符串
        #     batch['note' + '_hist'] = pad_list(batch['note' + '_hist'], device=device)# hist, B,T,D
        #     batch['note' + '_hist_mask'] = batch['note' + '_hist'].sum(dim=-1) !=0 # B,T
        #
        #     # 作为context,需要合并
        #     batch['note' + '_hist'] = torch.cat([batch['note' + '_hist'], batch['note'].unsqueeze(dim=1)], dim=1)  # B,T+1,D
        #     batch['note' + '_mask'] = batch['note'].sum(dim=-1) != 0  # B,
        #     batch['note' + '_hist_mask'] = torch.cat(
        #         [batch['note' + '_hist_mask'], batch['note' + '_mask'].unsqueeze(dim=1)], dim=1)  # B,T+1
        #
        #     batch['has_note'] = True

    # mask设置为固定
    if mask_static:
        batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
    # mask不固定, evaldffs固定就好, diverse
    else:
        m = 3  # 4 if batch['has_note'] else 3
        batch['mask'] = utils.generate_mask(batch['conditions'].shape[0], m=m).to(device=device)

    aligned_data = [batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1) for feature_key in
                    feature_keys]  # B,V,1
    batch['mask_hist'] = torch.cat(aligned_data, dim=-1)  # B, V, 3，记录history mask状态

    return indices, batch

# secmi_eval.py
# -*- coding: utf-8 -*-

import random
from typing import Any, Dict, Optional, Tuple, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


# =========================================================
# 0) small utils
# =========================================================
def set_seed(seed: int = 528) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_tensor(x: Any) -> bool:
    return torch.is_tensor(x)


def _to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    return x


def _extract_first_tensor(out: Any) -> torch.Tensor:
    """兼容 model 输出为 Tensor / (Tensor, ...) / dict{...:Tensor}"""
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for v in out:
            if torch.is_tensor(v):
                return v
        raise TypeError("[SecMI] model output tuple/list but no Tensor found")
    if isinstance(out, dict):
        # 优先常见 key
        for k in ("x0", "pred_x0", "eps", "epsilon", "noise", "pred_noise", "model_output", "out"):
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        for v in out.values():
            if torch.is_tensor(v):
                return v
        raise TypeError("[SecMI] model output dict but no Tensor found")
    raise TypeError(f"[SecMI] Unsupported model output type: {type(out)}")


def _list_depth(x: Any, max_check: int = 4) -> int:
    """粗略判断嵌套 list 深度：用于判断 2d/3d tokenization"""
    d = 0
    cur = x
    while isinstance(cur, list) and len(cur) > 0 and d < max_check:
        d += 1
        cur = cur[0]
    return d


# =========================================================
# 1) schedule (alphas_cumprod)
# =========================================================
def _linear_betas(T: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, steps=T, dtype=torch.float32)


def get_alphas_cumprod_fallback(
    generated_model: torch.nn.Module,
    diff_config: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """
    1) 尝试从 model/diffusion 中拿 alphas_cumprod 或 betas
    2) 尝试从你项目的 utils.stable_diffusion_beta_schedule() 拿 betas（若能 import）
    3) 最后 fallback linear betas
    """
    # --- 1) model buffers/attrs
    # 常见字段名
    for name in ("alphas_cumprod", "alpha_cumprod", "alpha_bars", "alpha_bar"):
        if hasattr(generated_model, name) and torch.is_tensor(getattr(generated_model, name)):
            return getattr(generated_model, name).detach().to(device)

    if hasattr(generated_model, "diffusion"):
        diff = generated_model.diffusion
        for name in ("alphas_cumprod", "alpha_cumprod", "alpha_bars", "alpha_bar"):
            if hasattr(diff, name) and torch.is_tensor(getattr(diff, name)):
                return getattr(diff, name).detach().to(device)

    # --- betas from model?
    betas = None
    for name in ("betas", "beta"):
        if hasattr(generated_model, name) and torch.is_tensor(getattr(generated_model, name)):
            betas = getattr(generated_model, name).detach()
            break
    if betas is None and hasattr(generated_model, "diffusion") and hasattr(generated_model.diffusion, "betas"):
        if torch.is_tensor(generated_model.diffusion.betas):
            betas = generated_model.diffusion.betas.detach()

    # --- 2) try project utils schedule (optional)
    if betas is None:
        try:
            import utils as _utils  # 你的项目里大概率有
            if hasattr(_utils, "stable_diffusion_beta_schedule"):
                b = _utils.stable_diffusion_beta_schedule()
                betas = torch.tensor(b, dtype=torch.float32) if not torch.is_tensor(b) else b.float()
        except Exception:
            betas = None

    # --- 3) fallback linear
    if betas is None:
        T = int(diff_config.get("SECMI_T_STEPS", 1000))
        beta_start = float(diff_config.get("SECMI_BETA_START", 1e-4))
        beta_end = float(diff_config.get("SECMI_BETA_END", 2e-2))
        print(f"[SecMI][WARN] schedule not found in model/ckpt. Use fallback linear betas: "
              f"T={T}, beta_start={beta_start}, beta_end={beta_end}")
        betas = _linear_betas(T=T, beta_start=beta_start, beta_end=beta_end)

    betas = betas.to(device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod


def q_sample(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    x0: [B,C,D]
    t:  [B] long
    eps:[B,C,D]
    """
    a = alphas_cumprod[t].view(-1, 1, 1)  # [B,1,1]
    return torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * eps


# =========================================================
# 2) call model: your nnet signature is forward(x, timesteps, context=...)
# =========================================================
@torch.no_grad()
def predict_model_auto(
    generated_model: torch.nn.Module,
    xt: torch.Tensor,
    t_model: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    """
    只针对你项目最常见形式：
        nnet(x, timesteps, context=context)
    同时做少量 fallback 以免封装层不同（nnet/model）
    """
    candidates = []
    if hasattr(generated_model, "nnet"):
        candidates.append(getattr(generated_model, "nnet"))
    if hasattr(generated_model, "model"):
        candidates.append(getattr(generated_model, "model"))
    candidates.append(generated_model)

    last_err = None
    for fn in candidates:
        try:
            out = fn(xt, t_model, context=context)
            return _extract_first_tensor(out)
        except Exception as e:
            last_err = e

        # 有些实现用第三个位置参数，不用 keyword
        try:
            out = fn(xt, t_model, context)
            return _extract_first_tensor(out)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"[SecMI] cannot call diffusion model. last error: {repr(last_err)}")


# =========================================================
# 3) ROC / AUC (no sklearn)
# =========================================================
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


# =========================================================
# 4) dataset wrapper & batch normalize
# =========================================================
class _IndexWrapper(torch.utils.data.Dataset):
    """
    保证 __getitem__ 返回 (index, sample_dict)
    - 如果原 dataset 已经返回 (index, dict)，就直接透传
    - 如果原 dataset 返回 dict，就 wrap 成 (i, dict)
    """
    def __init__(self, ds):
        self.ds = ds

        # 试探第一个样本形态
        try:
            s0 = ds[0]
        except Exception:
            s0 = None
        self._already_pair = (
            isinstance(s0, (tuple, list)) and len(s0) == 2 and isinstance(s0[1], dict)
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        if self._already_pair:
            return item  # (index, dict)
        if isinstance(item, dict):
            return idx, item
        # 如果是别的形态，也尽量包一层
        return idx, {"_raw": item}


def _default_collate_keep_dict(batch):
    """
    batch 是 list of (idx, dict)
    输出:
      indices: LongTensor [B]
      merged: dict，value 变成 list[B] 方便下游自己处理
    """
    indices, samples = zip(*batch)
    indices = torch.tensor(indices, dtype=torch.long)

    merged = {}
    # union keys
    keys = set()
    for s in samples:
        if isinstance(s, dict):
            keys |= set(s.keys())
    for k in keys:
        merged[k] = [s.get(k) for s in samples]
    return indices, merged


def _move_batch_tensors_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# =========================================================
# 5) SecMI-stat main
# =========================================================
@torch.no_grad()
def run_secmi_mia_eval(
    generated_model: torch.nn.Module,
    train_dataset: Any,
    test_dataset: Any,
    tokenizers: Any,
    diff_config: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    你外面只需要这样调用（保持不变）：
        mia_res = run_secmi_mia_eval(generated_model, train_dataset, test_dataset, tokenizers, diff_config, config)

    核心思想（SecMI-stat）：
    - 先用 generated_model.encode_everything_test 得到 x0(z) 和 context
    - 做 q_sample 得到 xt
    - 用模型预测 (默认预测 x0，因为你 DPM_Solver 用 predict_x0=True)
    - score = - MSE(pred, target)  (越大越像 member)

    可控参数（在 config 里加）：
      SECMI_N_MEMBER / SECMI_N_NONMEMBER
      SECMI_NUM_QUERIES
      SECMI_BATCH_SIZE
      SECMI_RANDOM_T  (True/False)
      SECMI_T          (固定 t，默认 T-1)
      SECMI_PRED_TYPE  ("x0" 或 "eps")  默认 "x0"
    """
    seed = int(config.get("SECMI_SEED", config.get("SEED", 528)))
    set_seed(seed)

    device = next(generated_model.parameters()).device
    generated_model.eval()

    # ---------- schedule ----------
    alphas_cumprod = get_alphas_cumprod_fallback(generated_model, diff_config, device)
    T = int(alphas_cumprod.numel())

    # ---------- params ----------
    num_queries = int(config.get("SECMI_NUM_QUERIES", 8))
    use_random_t = bool(config.get("SECMI_RANDOM_T", False))
    max_member = int(config.get("SECMI_N_MEMBER", 2000))
    max_nonmem = int(config.get("SECMI_N_NONMEMBER", 2000))
    batch_size = int(config.get("SECMI_BATCH_SIZE", 16))

    pred_type = str(config.get("SECMI_PRED_TYPE", "x0")).lower().strip()  # "x0" or "eps"
    if pred_type not in ("x0", "eps"):
        pred_type = "x0"

    t_fixed = int(config.get("SECMI_T", T - 1))
    t_fixed = max(1, min(T - 1, t_fixed))

    # ---------- loaders (完全不依赖你项目 collate_fn_dict) ----------
    train_loader = DataLoader(
        _IndexWrapper(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=_default_collate_keep_dict,
    )
    test_loader = DataLoader(
        _IndexWrapper(test_dataset),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=_default_collate_keep_dict,
    )

    # ---------- helper: prepare batch for encode_everything_test ----------
    def _prepare_for_encode(indices: torch.Tensor, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        尽量少假设：
        - 如果 batch 本身已经是 tensor（很多数据管线是这样），只 move to device
        - 如果是 list，就尽量保持原样，让 encode_everything_test 自己处理
        """
        indices = indices.to(device)

        # 一些数据集返回 merged dict: value 是 list[B]，而你的 encode_everything_test 可能期望 list[B] 形式
        # 所以这里不强行 tokenization，只在必要时你再手动打开（建议先跑通）
        batch = _move_batch_tensors_to_device(batch, device)

        return indices, batch

    # ---------- helper: encode to (x0, context) ----------
    def _encode_x0_context(indices: torch.Tensor, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        最关键：用你 diffusion nnet 自带的 encode_everything_test 得到:
          x0(z): [B,C,D]
          context: [B,L,D]
        """
        if not hasattr(generated_model, "encode_everything_test"):
            raise AttributeError("[SecMI] generated_model has no encode_everything_test(). "
                                 "请确认 load_diffusion 返回的是 nnet，并且 nnet 提供 encode_everything_test。")

        out = generated_model.encode_everything_test(indices, **batch)

        # 你项目里常见: (_z, _context_emb, _, labels, hard_labels)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            z = out[0]
            context = out[1]
        elif isinstance(out, dict):
            # 兜底
            z = out.get("z", out.get("x0", None))
            context = out.get("context", out.get("cond", None))
            if z is None or context is None:
                raise RuntimeError(f"[SecMI] encode_everything_test dict keys={list(out.keys())} "
                                   f"but cannot find z/context")
        else:
            raise RuntimeError(f"[SecMI] unexpected encode_everything_test output type={type(out)}")

        if not torch.is_tensor(z) or not torch.is_tensor(context):
            raise RuntimeError("[SecMI] encode_everything_test did not return tensors for z/context")

        # 统一 float32
        z = z.to(device=device, dtype=torch.float32)
        context = context.to(device=device, dtype=torch.float32)

        # 统一成 [B,C,D]
        if z.dim() == 2:
            z = z.unsqueeze(0)
        if z.dim() != 3:
            raise RuntimeError(f"[SecMI] z shape expected [B,C,D], got {tuple(z.shape)}")

        return z, context

    # ---------- helper: collect errs ----------
    def _collect_errs(loader: DataLoader, max_n: int) -> np.ndarray:
        errs = []
        collected = 0

        for indices, batch in loader:
            if collected >= max_n:
                break

            indices, batch = _prepare_for_encode(indices, batch)

            # 1) encode -> x0, context
            x0, context = _encode_x0_context(indices, batch)
            B = x0.size(0)

            # 截断到 max_n
            if collected + B > max_n:
                keep = max_n - collected
                x0 = x0[:keep]
                context = context[:keep]
                indices = indices[:keep]
                B = keep

            # 2) 多次 query 平均误差
            e_acc = torch.zeros(B, device=device, dtype=torch.float32)

            for _ in range(num_queries):
                if use_random_t:
                    t = torch.randint(low=1, high=T, size=(B,), device=device, dtype=torch.long)
                else:
                    t = torch.full((B,), t_fixed, device=device, dtype=torch.long)

                eps = torch.randn_like(x0)
                xt = q_sample(x0, t, eps, alphas_cumprod)

                # 注意：你的 nnet timesteps 很可能允许 float（DPM solver 用连续 t）
                t_model = t.float()

                pred = predict_model_auto(generated_model, xt, t_model, context)

                # pred 可能是 x0_hat（推荐）或 eps_hat
                pred = pred.to(dtype=torch.float32)

                if pred.shape != x0.shape:
                    # 有些实现输出 [B,D] 或 [B,C,D] 轻微差异，这里尽量对齐
                    if pred.dim() == 2 and x0.dim() == 3 and pred.size(0) == x0.size(0) and pred.size(1) == x0.size(2):
                        pred = pred.unsqueeze(1).repeat(1, x0.size(1), 1)
                    else:
                        raise RuntimeError(f"[SecMI] pred shape {tuple(pred.shape)} != x0 shape {tuple(x0.shape)}. "
                                           f"请确认 nnet 输出是 x0 或 eps 且维度一致。")

                if pred_type == "x0":
                    diff = (pred - x0).reshape(B, -1)
                else:  # "eps"
                    diff = (pred - eps).reshape(B, -1)

                e = (diff * diff).mean(dim=1)
                e_acc += e

            e_acc = e_acc / float(num_queries)
            errs.append(e_acc.detach().cpu().numpy())
            collected += B

        if len(errs) == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(errs, axis=0)

    # ---------- run ----------
    n_mem = min(max_member, len(train_dataset))
    n_non = min(max_nonmem, len(test_dataset))

    err_mem = _collect_errs(train_loader, n_mem)   # 越小越像 member
    err_non = _collect_errs(test_loader, n_non)

    scores = np.concatenate([-err_mem, -err_non], axis=0)  # 越大越像 member
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
        "timestep": int(t_fixed),
        "num_queries": int(num_queries),
        "pred_type": pred_type,
        "seed": int(seed),
        "T": int(T),
    }


###########################################################################

def _extract_incomplete_matrix(sample, keys=('incomplete', 'incomplete_raw')):
    inc = None
    for k in keys:
        if k in sample:
            inc = sample[k]
            break
    if inc is None:
        return None

    inc = np.array(inc, dtype=object)
    while inc.ndim > 2:
        inc = inc[0]

    if inc.ndim == 1:
        if inc.shape[0] != 3:
            return None
        inc = inc.reshape(1, 3)

    if inc.ndim != 2 or inc.shape[1] != 3:
        return None

    def to01(x):
        try:
            return 0 if int(x) == 0 else 1
        except Exception:
            return 1

    out = np.zeros((inc.shape[0], 3), dtype=int)
    for i in range(inc.shape[0]):
        for j in range(3):
            out[i, j] = to01(inc[i, j])
    return out

def is_complete_sample(sample):
    inc = _extract_incomplete_matrix(sample)
    if inc is not None:
        return int(inc.max()) == 0
    # fallback：弱判断
    for feat in FEATURE_ORDER:
        if feat not in sample:
            return False
        v = sample[feat]
        if v is None:
            return False
        if isinstance(v, list) and len(v) == 0:
            return False
    return True

def filter_complete_samples(samples):
    return [s for s in samples if is_complete_sample(s)]

def _make_empty_like(orig_val, sample):
    # list-of-list(按visit) => [[],[],...]
    if isinstance(orig_val, list) and len(orig_val) > 0 and isinstance(orig_val[0], list):
        V = len(sample.get('visit_id_hist', orig_val))
        return [[] for _ in range(V)]
    if isinstance(orig_val, list):
        return []
    return orig_val

def apply_view_mask_ratio(samples, ratio, seed=0, mode='bernoulli'):
    """
    人工按视图掩码，并把被掩码视图置空（不再使用真实值）
    同时备份真实到 orig_*，并写入 incomplete/incomplete_raw
    """
    rng = np.random.RandomState(seed)
    out = copy.deepcopy(samples)
    N = len(out)

    if mode == 'exact':
        miss_mat = np.zeros((N, 3), dtype=int)
        k = int(round(ratio * N))
        for j in range(3):
            idx = rng.permutation(N)[:k]
            miss_mat[idx, j] = 1
    else:
        miss_mat = (rng.rand(N, 3) < ratio).astype(int)

    for i, s in enumerate(out):
        miss = miss_mat[i].tolist()     # [cond, proc, drug]
        s['incomplete'] = [miss]        # [[...]]
        s['incomplete_raw'] = miss      # [...]

        # 备份真实值（仅用于融合/评估）
        for j, feat in enumerate(FEATURE_ORDER):
            if feat in s:
                s[f'orig_{feat}'] = copy.deepcopy(s[feat])

        # 掩码视图置空：关键，避免补全偷看真实
        for j, feat in enumerate(FEATURE_ORDER):
            if miss[j] == 1 and feat in s:
                s[feat] = _make_empty_like(s[feat], s)

    return out

def _is_list_of_list(x):
    return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)

def _match_structure(new_val, orig_val):
    # 让 new 尽量与 orig 嵌套结构一致
    if _is_list_of_list(orig_val):
        if _is_list_of_list(new_val):
            return new_val
        return [new_val]
    else:
        if _is_list_of_list(new_val):
            if len(new_val) == 1:
                return new_val[0]
            flat = []
            for vv in new_val:
                flat.extend(vv)
            return flat
        return new_val

def build_imp_fields(samples, new_prefix='new_', out_prefix='imp_'):
    """
    缺失视图 -> 用 new_*
    非缺失视图 -> 用 orig_*（真实）
    输出 imp_conditions/imp_procedures/imp_drugs
    """
    for s in samples:
        inc = _extract_incomplete_matrix(s, keys=('incomplete', 'incomplete_raw'))
        miss_vec = [0, 0, 0] if inc is None else inc.max(axis=0).tolist()

        for j, feat in enumerate(FEATURE_ORDER):
            orig_val = s.get(f'orig_{feat}', s.get(feat, None))
            new_val  = s.get(f'{new_prefix}{feat}', None)

            if orig_val is None:
                continue

            if miss_vec[j] == 1 and new_val is not None:
                s[f'{out_prefix}{feat}'] = _match_structure(new_val, orig_val)
            else:
                s[f'{out_prefix}{feat}'] = orig_val

    return samples


############################### One View ###############################
VIEW2IDX = {"conditions": 0, "procedures": 1, "drugs": 2}
ALL_VIEWS = ("conditions", "procedures", "drugs")

def _is_list_of_list(x):
    return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)

def _normalize_incomplete_raw(sample):
    """
    返回长度=3的 incomplete_raw。
    兼容：
      - incomplete_raw: [3]
      - incomplete: [[3]] 或 Vx3 -> 做 OR 合并成 [3]
      - 都没有 -> [0,0,0]
    """
    inc_raw = sample.get("incomplete_raw", None)
    if isinstance(inc_raw, list) and len(inc_raw) == 3 and not isinstance(inc_raw[0], list):
        return [int(x) for x in inc_raw]

    inc = sample.get("incomplete", None)
    if isinstance(inc, list) and len(inc) > 0 and isinstance(inc[0], list):
        # inc 可能是 [[3]] 或 Vx3，统一 OR 合并成 [3]
        out = [0, 0, 0]
        for row in inc:
            if isinstance(row, list) and len(row) == 3:
                out = [max(out[i], int(row[i])) for i in range(3)]
        return out

    return [0, 0, 0]

def _sample_all_views_complete(sample):
    """
    判断样本是否“三视图都完整”：
      1) incomplete_raw 全 0
      2) conditions/procedures/drugs 三个字段都有内容（若是 list-of-list，则每个visit非空）
    """
    inc_raw = _normalize_incomplete_raw(sample)
    if any(x != 0 for x in inc_raw):
        return False

    for k in ALL_VIEWS:
        v = sample.get(k, None)
        if v is None:
            return False
        if _is_list_of_list(v):
            # 每个 visit 都要非空
            for vv in v:
                if vv is None or len(vv) == 0:
                    return False
        else:
            if not isinstance(v, list) or len(v) == 0:
                return False
    return True

def mask_one_view_on_complete_samplewise(samples, view_to_mask, ratio, seed=123, mode="bernoulli"):
    """
    只对 view_to_mask 做额外缺失：
      - 只在“三视图都完整”的样本上，以概率 ratio 置缺失
      - 其它视图不做人工掩码
      - 输出保证：incomplete_raw 一定是 [3]，incomplete 一定是 [[3]]
    """
    assert view_to_mask in VIEW2IDX
    rng = np.random.RandomState(seed)
    vidx = VIEW2IDX[view_to_mask]

    out = []
    for s in samples:
        s2 = copy.deepcopy(s)
        inc_raw = _normalize_incomplete_raw(s2)

        # 只对三视图完整样本施加人工缺失
        eligible = _sample_all_views_complete(s2)
        do_mask = eligible and ((rng.rand() < ratio) if mode == "bernoulli" else False)

        if do_mask:
            inc_raw[vidx] = 1  # 标记该视图缺失

            # 把该视图置空：如果是 visit-wise list-of-list，置空每个 visit
            v = s2.get(view_to_mask, [])
            if _is_list_of_list(v):
                s2[view_to_mask] = [[] for _ in range(len(v))]
            else:
                s2[view_to_mask] = []

        # 关键：保证格式一致
        s2["incomplete_raw"] = inc_raw
        s2["incomplete"] = [inc_raw]

        out.append(s2)

    return out




def _pick_device(cfg_gpu):
    s = str(cfg_gpu).strip() if cfg_gpu is not None else "0"
    if s.lower() in ("cpu", "-1", ""):
        return torch.device("cpu")
    if "," in s:  # 形如 "0,1" 只取第一张
        s = s.split(",")[0].strip()
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        print("[warn] CUDA 不可用，回退 CPU")
        return torch.device("cpu")
    try:
        idx = int(s)
    except Exception:
        idx = 0
    n = torch.cuda.device_count()
    if idx < 0 or idx >= n:
        print(f"[warn] 请求 GPU {idx} 越界；当前可见 {n} 张卡，改用 0")
        idx = 0
    dev = torch.device(f"cuda:{idx}")
    try:
        torch.cuda.set_device(dev)  # 绑定当前设备，避免后续库内部用默认0号
    except Exception as e:
        print(f"[warn] set_device 失败：{e}；继续使用 {dev}")
    return dev


class SampleEHRDataset(SampleEHRDatasets):
    # 在制造数据时候可以避免繁琐的检查,
    def _validate(self):
        return True

def get_warm_cold_split(samples):
    from pyhealth.datasets import split_by_patient
    sample_dataset = SampleEHRDatasets(samples)
    _,_,warm_test_dataset = split_by_patient(
        sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
        train_ratio=1.0,  # Train test split
        warm_cold='warm',
        seed=config['SEED']
    )
    _, _, cold_test_dataset = split_by_patient(
        sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
        train_ratio=1.0,  # Train test split
        warm_cold='cold',
        seed=config['SEED']
    )
    print("Success split warm - cold !")
    return warm_test_dataset, cold_test_dataset





def run_single_config(augment=False, retrain=False, exp_num='3', pretrain=True):
    """
    :param augment: whether augmented dataset
    :param retrain: whether retrain the diffusion model
    :param exp_num:
    :return:
    """
    # load datasets
    # STEP 1: load data
    # root_to = '/home/exp1/mhc/MMHealth-main/data/{}/{}/processed/'.format(config['TASK'], config['DATASET'])
    root_to = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/{}/{}/processed/'.format(config['TASK'], config['DATASET'])

    task_fn, mode = get_task_fn(config)
    if not os.path.exists(root_to + 'datasets_pre_stand.pkl'):
        print("Prepare dataset!")
        if config['DATASET'] == 'MIII':
            base_dataset = MIMIC3Dataset(
                # root="/home/exp1/mhc/MMHealth-main/data/physionet.org/files/mimiciii/1.4",
                root='/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/mimic-iii-clinical-database-1.4/',
                tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
                code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": config['ATCLEVEL']}})},
                # 这里graphcare的ATC-level是3；和我们在data阶段有差别
                dev=False,
                refresh_cache=False,
            )
        elif config['DATASET'] == 'eICU':
            base_dataset = eICUDataset(
                # root="//home/exp1/mhc/MMHealth-main/data/physionet.org/files/eicu-crd/2.0",
                root='/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/eicu-collaborative-research-database-2.0/',
                tables=["diagnosis", "medication", "physicalExam", "treatment", "admissionDx"],
                dev=False,
                refresh_cache=False,
            )
        elif config['DATASET'] == 'MIV':
            base_dataset = MIMIC4Dataset(
                # root="/home/exp1/mhc/MMHealth-main/data/physionet.org/files/mimiciv/2.0/hosp",
                root='/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/mimic-iv-3.1/hosp/',
                tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": config['ATCLEVEL']}})},  # 4
                dev=False,
                refresh_cache=False,
            )
        elif config['DATASET'] == 'MIV-Note':  # 这里可以通过subject_id进行联系。因为有一部分数据被MIMIC官方雪藏了。所有用2.2以上的版本
            from pandarallel import pandarallel
            from pandas.core.groupby.generic import DataFrameGroupBy
            _real_initialize = pandarallel.initialize

            def safe_initialize(*args, **kwargs):
                # 不管别人怎么传，强制只开 1 个 worker，关掉内存文件系统
                kwargs["nb_workers"] = 1
                kwargs["use_memory_fs"] = False
                print("[INFO] Override pandarallel.initialize: nb_workers=1, use_memory_fs=False")
                _real_initialize(*args, **kwargs)

                # 把 GroupBy.parallel_apply 改回普通的 apply（单进程，稳定）
                def _safe_parallel_apply(self, func=None, *a, **k):
                    return self.apply(func, *a, **k)

                if hasattr(DataFrameGroupBy, "parallel_apply"):
                    DataFrameGroupBy.parallel_apply = _safe_parallel_apply
                    print("[INFO] Patch GroupBy.parallel_apply -> apply (disable pandarallel)")

            # 用我们的 safe_initialize 替换原函数
            pandarallel.initialize = safe_initialize

            base_dataset = MIMIC4Dataset(
                # root="/home/exp1/mhc/MMHealth-main/data/physionet.org/files/mimiciv/2.0/hosp",
                root='/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/mimic-iv-3.1/hosp/',
                # root='/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/',
                tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": config['ATCLEVEL']}})},  # 4
                dev=False,  ## PHE任务 原始状态
                # dev=True,  ## LOS任务 子集
                refresh_cache=False,
            )

        else:
            print("No such dataset!")
            return
        base_dataset.stat()
        # set task
        print("ABCDEF")
        sample_dataset = base_dataset.set_task(task_fn)
        print("ABCD")
        sample_dataset.stat()
        print("AB")
        samples = sample_dataset.samples
        save_pickle(samples, root_to + 'datasets_pre_stand.pkl')


        print("initial dataset done!")
        print("Please run again!")
        return
    else:
        start = time.time()
        samples = load_pickle(root_to + 'datasets_pre_stand.pkl')
        # load_code_convert(dataset=config['DATASET'], samples=samples) # 如果这里要弄，必须要重新生成rare disease

        # group
        generate_patient_group(samples, root_to)
        p_grouped = load_pickle(root_to + 'group_patient.pkl')

        end = time.time()
        print("Load data done! Cost time {} s".format(end-start))

        if config['DEV']:
            print("DEV train mode: 1000 patient")
            samples = samples[:3000]
            sample_dataset = SampleEHRDataset(# 这个贼耗时
                samples,
                dataset_name=config['DATASET'],
                task_name=config['TASK'],
            )
            train_dataset, val_dataset, test_dataset = split_by_patient(
                sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
                train_ratio=1.0,  # Train test split
                seed=528
            )
            del samples

            endt = time.time()
            print('Dataset done!, Cost {} s'.format(endt - end))
        else:
            device = _pick_device(config.get('GPU', 0))
            print('Using device:', device)

            reserved_tensor = None
            try:
                # reserved_tensor = torch.ones((10, 10)).to('cuda:' + config['GPU']) # 占据GPU
                # print("GPU Memory Usage", torch.cuda.memory_allocated('cuda:' + config['GPU']) / 1024 / 1024 / 1024,
                #       "GB")
                if device.type == 'cuda':
                    reserved_tensor = torch.ones((10, 10), device=device)
                    gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                    print(f"GPU Memory Usage on {device}: {gb:.3f} GB")
                if os.path.exists(root_to + 'train_dataset.pkl'):
                    sample_dataset = convert_dataset(samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False, all=True) # 需要查看是否需要load_code_convert
                    # warm_test, cold_test = get_warm_cold_split(samples)
                    train_samples = load_pickle(root_to + 'train_dataset.pkl')
                    val_samples = load_pickle(root_to + 'val_dataset.pkl')
                    test_samples = load_pickle(root_to + 'test_dataset.pkl')
                    train_dataset = convert_dataset(train_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
                    val_dataset = convert_dataset(val_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
                    test_dataset = convert_dataset(test_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
                else:
                    sample_dataset = convert_dataset(samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=True)

                    train_dataset, val_dataset, test_dataset = split_by_patient(
                        sample_dataset, [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
                        train_ratio=1.0,  # Train test split
                        warm_cold=False, # 所有的warm一次性看完。
                        seed=config['SEED']
                    )  # 这样似乎更快，固定随机种子的时候是一样的；
                    # train_samples, _, test_samples = achieve_samples(train_dataset), _, achieve_samples(test_dataset)
                    save_pickle(train_dataset, root_to + 'train_dataset.pkl')
                    save_pickle(val_dataset, root_to + 'val_dataset.pkl')
                    save_pickle(test_dataset, root_to + 'test_dataset.pkl')
            finally:
                del reserved_tensor
                # torch.cuda.empty_cache() # 会突然爆oom

            endt = time.time()
            print('Train Dataset done!, Cost {} s'.format(endt - end))


    # STEP 2: load dataloader
    if augment:
        diff_config['device'] = int(config['GPU'])
        if config['DATASET'] == "MIV-Note": # 要改train_diff, eval_diff
            # diff_config['nnet']['num_contx_token'] = 10
            diff_config['nnet']['num_contx_token'] = 9
        tokenizers = get_tokenizers(sample_dataset)
        if retrain or not os.path.exists(diff_config['ckpt_root']+ config['DATASET']):
            print('start retrain diffusion model!')
            print('Sample data of 1 patient, input for diffusion dataset ', train_dataset[0])
            best_model_path, generated_model, num_train_visit = run_diffusion1(diff_config, config, [train_dataset, val_dataset], tokenizers, exp_num=exp_num) # 在val上测试防止泄漏
            assert False
            # best_model_path, generated_model, num_train_visit = run_diffusion(diff_config, config, [train_dataset, val_dataset], tokenizers, exp_num=exp_num) # 在val上测试防止泄漏
            with open(diff_config['ckpt_root'] + config['DATASET'] + '/' + config['TASK'] + "/best_model_path.txt", "w") as file:
                print(best_model_path, file=file)
                print(num_train_visit, file=file)

            diff_config["nnet_path"] = best_model_path
            diff_config["num_train_visit"] = num_train_visit

            train_samples, val_samples = augment_dataset(generated_model,diff_config, config,  [train_dataset, val_dataset], tokenizers, root_to=root_to)
            test_samples = augment_dataset1(generated_model, diff_config, config, [train_dataset, test_dataset], tokenizers, root_to=root_to)
            print('retrain diffusion model done!')
        else:
            if not os.path.exists(root_to + 'train_samples.pkl'):
                with open(diff_config['ckpt_root'] + config['DATASET'] + '/' + config['TASK']+ "/best_model_path.txt", "r") as file:
                    lines = file.readlines()  # 逐行读取内容
                # 获取内容
                best_model_path = lines[0].strip()  # 第一个内容
                num_train_visit = int(lines[1].strip())  # 第二个内容，转换为整数

                # 输出读取的内容
                print("Best model path:", best_model_path)
                print("Number of training visits:", num_train_visit)

                diff_config["nnet_path"] = best_model_path
                diff_config["num_train_visit"] = num_train_visit

                print('Load diffusion model!')
                generated_model = load_diffusion(diff_config, tokenizers)

                print('Load diffusion model done!')
                train_samples, val_samples = augment_dataset(generated_model,diff_config, config,  [train_dataset, val_dataset], tokenizers, root_to=root_to)
                test_samples = augment_dataset1(generated_model, diff_config, config, [train_dataset, test_dataset], tokenizers, root_to=root_to)
                # train_samples, test_samples = augment_dataset(generated_model,diff_config, config,  [train_dataset, test_dataset], tokenizers, root_to=root_to)

            else:
                print("Have been augmented!")
                # 1) 先读取之前保存的 best_model_path 和 num_train_visit
                best_info_path = diff_config['ckpt_root'] + config['DATASET'] + '/' + config[
                    'TASK'] + "/best_model_path.txt"
                with open(best_info_path, "r") as f:
                    lines = f.readlines()
                best_model_path = lines[0].strip()
                num_train_visit = int(lines[1].strip())
                diff_config["nnet_path"] = best_model_path
                diff_config["num_train_visit"] = num_train_visit
                # 2) 加载已经训练好的 diffusion 模型（不会重新训练）
                print("Load diffusion model for latent evaluation...")
                generated_model = load_diffusion(diff_config, tokenizers)
                print("Load diffusion model done!")
                # 4) 评估完再释放 val/test dataset，加载已经生成好的增强样本

                # if config.get("EVAL_MIA", True):
                #     mia_res = run_secmi_mia_eval(
                #         generated_model=generated_model,
                #         train_dataset=train_dataset,
                #         test_dataset=test_dataset,
                #         tokenizers=tokenizers,
                #         diff_config=diff_config,
                #         config=config,
                #     )
                #     print(
                #         f"[MIA][SecMI-stat] t_sec={mia_res['t_sec']} "
                #         f"AUC={mia_res['AUC']:.4f} "
                #         f"TPR@FPR=0.001={mia_res['TPR@FPR=0.001']:.4f} "
                #         f"TPR@FPR=0.01={mia_res['TPR@FPR=0.01']:.4f} "
                #         f"(member={mia_res['n_member']}, nonmember={mia_res['n_nonmember']})"
                #     )
                # assert False
                del val_dataset, test_dataset
                train_samples = load_pickle(root_to + 'train_samples.pkl')
                val_samples = load_pickle(root_to + 'val_samples.pkl')
                test_samples = load_pickle(root_to + 'test_samples.pkl')

        train_dataset = convert_dataset(train_samples, dataset_name=config['DATASET'], task_name=config['TASK'],
                                            valid=False)
        val_dataset = convert_dataset(val_samples, dataset_name=config['DATASET'], task_name=config['TASK'],
                                          valid=False)
        test_dataset = convert_dataset(test_samples, dataset_name=config['DATASET'], task_name=config['TASK'],
                                           valid=False)
        print("Augment Done!")

    collate_fn = get_special_input(config)
    # missing_statistics(train_dataset, config)
    train_dataloader = get_dataloader(train_dataset, batch_size=config['BATCH'], shuffle=True, drop_last=True,
                                      collate_fn=collate_fn)
    # val_dataloader = get_dataloader(val_dataset, batch_size=config['BATCH']*5, shuffle=False, drop_last=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=config['BATCH'], shuffle=False, drop_last=False,
                                    collate_fn=collate_fn)
    test_dataloader = get_dataloader(test_dataset, batch_size=config['BATCH'], shuffle=False, drop_last=False,
                                     collate_fn=collate_fn)
    load_dataloader = time.time()
    print('Dataloader done!, Cost {} s'.format(load_dataloader - endt))
    # cache clear
    # torch.cuda.empty_cache() # 占用内存清理
    gc.collect()

    print("===============Train Healthcare Prediction Model!===============")
    # assert False
    existing_model = None
    if pretrain:
        print("Pretrain on the baseline models")
        start = time.time()
        run_pretrain(sample_dataset, config, exp_num=exp_num,
                     special_input={'train': train_dataset, 'test': test_dataset})
        end_time = time.time()

        model = load_pretrain_model(sample_dataset, config, special_input=None, exp_num=exp_num, mode=mode)
        print("Pretrain done! Cost {} s".format(end_time - start))
        existing_model = model

    start = time.time()
    run_pcf(sample_dataset, train_dataloader, val_dataloader, test_dataloader, config, exp_num=exp_num,
            existing_model=existing_model, mode=mode, aux_data={'p_grouped': p_grouped})
    end = time.time()
    print("Training done! Cost {} s".format(end - start))





# # PHE\LOS任务
# if __name__ == '__main__':
#     retrain = False
#     augment = True # ours要开
#     pretrain = False # 只有在一些二次训练的baseline上面开启。SMART, CORGAN
#     # pretrain = True
#     if config['MODEL'] in  ['ours', 'MedDiff']:
#         retrain = False
#         # retrain = True
#         augment = True
#     elif config['MODEL'] == 'SMART' or config['MODEL'] == 'CORGAN':
#         pretrain = True
#     else:
#         print('No need to retrain or augment!')
#
#     exp_num = '20' # 1 为aba；20 为rebutall
#     print("Hi, This is Diffrm Health!")
#     print("You are running on", config['DATASET'], "dataset! ", config['TASK'], "task!", config['MODEL'], "model!")
#     run_single_config(augment=augment, retrain=retrain, exp_num=exp_num, pretrain=pretrain)
#     print("All Done!")
#     # du -sh ~/* | sort -h ; disk


## Pretrain Diffmv
if __name__ == '__main__':
    retrain = False
    augment = True # ours要开
    # pretrain = False # 只有在一些二次训练的baseline上面开启。SMART, CORGAN
    pretrain = True
    if config['MODEL'] in  ['ours', 'MedDiff']:
        # retrain = False
        retrain = True
        augment = True
    elif config['MODEL'] == 'SMART' or config['MODEL'] == 'CORGAN':
        pretrain = True
    else:
        print('No need to retrain or augment!')

    exp_num = '20' # 1 为aba；20 为rebutall
    print("Hi, This is Diffrm Health!")
    print("You are running on", config['DATASET'], "dataset! ", config['TASK'], "task!", config['MODEL'], "model!")
    run_single_config(augment=augment, retrain=retrain, exp_num=exp_num, pretrain=pretrain)
    print("All Done!")
    # du -sh ~/* | sort -h ; disk
