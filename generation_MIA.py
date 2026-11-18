# -*- coding: utf-8 -*-
"""
seed528_privacy_audit_full.py

在 seed528 / DPM-loss 审计代码基础上：
(1) 加权Jaccard/wrec/wprec（IDF权重 + topk稀有token过滤）
(2) 支持 mask_mode:
    - random_one: 每样本随机mask一个view
    - all_single: 每样本枚举3个单view mask，再聚合
    - all_nonempty: 每样本枚举7个非空mask组合，再聚合
    - all_views: 每样本固定mask=[1,1,1]（三视图全掩码）  <-- NEW
(3) 支持返回“分别mask结果 + 聚合结果”（return_per_mask=True） <-- NEW
(4) 单个测试样本 多步steps loss曲线诊断

依赖你工程已有：
- utils.stable_diffusion_beta_schedule()
- NoiseScheduleVP, DPM_Solver
- get_dataset, collate_fn_dict
"""

import os
import math
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from diffuser import diff_utils as utils
from diffuser.diff_dataset import get_dataset, collate_fn_dict
from diffuser.dpm import NoiseScheduleVP, DPM_Solver


# ============================================================
# (0) 简单的load_pickle（你工程里有就用你工程的）
# ============================================================
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# (0) convert_input_fn（保持你给出的版本）
# ============================================================
def build_convert_input_fn_health_v1(tokenizers, device, config_health):
    feature_keys = ['conditions', 'drugs', 'procedures']

    def convert_input_fn(batch_data, mask_static=True):
        indices, batch = batch_data

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

        # ---- safe fallback for missing 'mask' field ----
        if mask_static:
            if 'mask' in batch:
                batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
            else:
                B = batch['conditions'].shape[0]
                batch['mask'] = torch.zeros(B, 3, dtype=torch.bool, device=device)
        else:
            m = 3
            batch['mask'] = utils.generate_mask(batch['conditions'].shape[0], m=m).to(device=device)

        aligned_data = [
            batch[fk + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1) for fk in feature_keys
        ]
        batch['mask_hist'] = torch.cat(aligned_data, dim=-1)

        return indices, batch

    return convert_input_fn


# ============================================================
# (1) 完整三视图子集
# ============================================================
def seed528_complete3v_subset_v1(ds_split, min_len=1, keys_map=None):
    if keys_map is None:
        keys_map = {
            "conditions": ["conditions_comp", "conditions_comps", "conditions"],
            "procedures": ["procedures_comp", "procedures_comps", "procedures"],
            "drugs":      ["drugs_comp", "drugs_comps", "drugs"],
        }

    def _unwrap(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, tuple):
            for t in reversed(x):
                if isinstance(t, dict):
                    return t
        return x

    def _pick_list(s, cands):
        if not isinstance(s, dict):
            return []
        for k in cands:
            v = s.get(k, None)
            if v is not None:
                return v
        return []

    keep = []
    for i in range(len(ds_split)):
        s = _unwrap(ds_split[i])

        c = _pick_list(s, keys_map["conditions"])
        p = _pick_list(s, keys_map["procedures"])
        d = _pick_list(s, keys_map["drugs"])

        if (len(c) >= min_len) and (len(p) >= min_len) and (len(d) >= min_len):
            keep.append(i)

    return Subset(ds_split, keep)


# ============================================================
# (2) random mask 1 view
# ============================================================
def seed528_viewmask_one_v1(B, device, seed):
    cuda_devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        ridx = torch.randint(0, 3, (B,), device=device)
        m = torch.zeros(B, 3, dtype=torch.bool, device=device)
        m[torch.arange(B, device=device), ridx] = True
        return m
#
# def seed528_enumerate_viewmasks_v1(mode: str, device):
#     """
#     返回 patterns: [K,3] bool
#     """
#     if mode == "all_single":
#         return torch.eye(3, dtype=torch.bool, device=device)  # [3,3]
#     elif mode == "all_nonempty":
#         masks = []
#         for a in [0, 1]:
#             for b in [0, 1]:
#                 for c in [0, 1]:
#                     if a == 0 and b == 0 and c == 0:
#                         continue
#                     masks.append([a, b, c])
#         return torch.tensor(masks, dtype=torch.bool, device=device)  # [7,3]
#     elif mode == "all_three":
#         return torch.tensor([[1, 1, 1]], dtype=torch.bool, device=device)  # [1,3] 只跑一次
#     else:
#         raise ValueError(f"Unknown mask mode: {mode}")


# ============================================================
# (2.1) NEW：枚举mask patterns（增加 all_views）
# ============================================================
def seed528_enumerate_viewmasks_v2(mode: str, device):
    """
    返回 patterns: [K,3] bool
      - all_single: 3个单掩码
      - all_nonempty: 7个非空组合掩码
      - all_views: 1个全掩码 [1,1,1]   <-- NEW
    """
    if mode == "all_single":
        return torch.eye(3, dtype=torch.bool, device=device)  # [3,3]
    elif mode == "all_nonempty":
        masks = []
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    if a == 0 and b == 0 and c == 0:
                        continue
                    masks.append([a, b, c])
        return torch.tensor(masks, dtype=torch.bool, device=device)  # [7,3]
    elif mode == "all_views":
        return torch.tensor([[1, 1, 1]], dtype=torch.bool, device=device)  # [1,3]
    else:
        raise ValueError(f"Unknown mask mode: {mode}")


# ============================================================
# (3) DPM masked latent MSE
# ============================================================
@torch.no_grad()
def seed528_dpm_masked_latent_mse_v1(
    nnet,
    z0_true, context,
    view_mask_b3,
    sample_steps=20,
    repeats=4,
    agg="mean",   # "mean" or "min"
    seed=528
):
    device = z0_true.device
    B, M, D = z0_true.shape

    mask_b3 = view_mask_b3.to(device)
    mask_b3d = mask_b3.unsqueeze(-1).expand(B, M, D)
    denom = mask_b3d.float().sum(dim=(1, 2)).clamp_min(1.0)

    betas = utils.stable_diffusion_beta_schedule()
    betas_t = torch.tensor(betas, device=device).float()
    N = betas_t.numel()
    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=betas_t)

    def model_fn(x, t_cont):
        t = t_cont * N
        return nnet(x, t, context=context)

    dpm = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)

    scores = []
    for r in range(repeats):
        cuda_devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(seed + 10007 * r)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + 10007 * r)
            noise = torch.randn_like(z0_true)

        z_init = torch.where(mask_b3d, noise, z0_true)
        z_hat = dpm.sample(z_init, steps=sample_steps, eps=1.0 / N, T=1.0)

        diff2 = (z_hat - z0_true) ** 2
        mse_masked = (diff2 * mask_b3d.float()).sum(dim=(1, 2)) / denom
        scores.append(mse_masked)

    scores = torch.stack(scores, dim=0)  # [R,B]
    return scores.min(dim=0).values if agg == "min" else scores.mean(dim=0)


# ============================================================
# (4) 收集loss（增强版：支持 random_one / all_single / all_nonempty / all_views）
#     NEW: return_per_mask=True -> 返回分别mask的loss矩阵
# ============================================================
@torch.no_grad()
def seed528_collect_losses_complete3v_dpm_v2(
    loader,
    nnet,
    convert_input_fn,
    sample_steps=20,
    repeats=4,
    agg="mean",                 # repeats聚合：mean/min
    strict_context=True,
    max_batches=None,
    seed_base=528,
    mask_mode="random_one",     # random_one / all_single / all_nonempty / all_views
    mask_agg="min",             # 对K个mask聚合：min/mean（all_views时K=1无意义）
    verbose_every=0,
    return_per_mask=False       # NEW
):
    feature_keys = ["conditions", "drugs", "procedures"]
    all_final = []
    all_per_mask = []   # list of [K,B]

    patterns_cache = None  # [K,3]

    for bi, (indices, batch) in enumerate(loader):
        if (max_batches is not None) and (bi >= max_batches):
            break

        indices, batch = convert_input_fn((indices, batch), mask_static=True)

        # 1) GT latent + full context
        z0_true, ctx_full, *_ = nnet.encode_everything_test(indices, **batch)
        B = z0_true.size(0)
        device = z0_true.device

        # 2) view_masks: [K,B,3]
        if mask_mode == "random_one":
            patterns_cache = None
            view_masks = seed528_viewmask_one_v1(
                B=B, device=device, seed=seed_base + bi * 1000 + 17
            ).unsqueeze(0)  # [1,B,3]
        else:
            patterns = seed528_enumerate_viewmasks_v2(mask_mode, device=device)  # [K,3]
            patterns_cache = patterns.detach().cpu()
            view_masks = patterns.unsqueeze(1).expand(-1, B, -1).contiguous()    # [K,B,3]

        K = view_masks.size(0)
        per_mask_scores = []

        for k in range(K):
            view_mask = view_masks[k]  # [B,3]

            # 3) strict_context
            if strict_context:
                batch2 = {}
                for kk, vv in batch.items():
                    batch2[kk] = vv.clone() if torch.is_tensor(vv) else vv

                for j, fk in enumerate(feature_keys):
                    m = view_mask[:, j]
                    if m.any():
                        batch2[fk][m] = 0
                        batch2[fk + "_mask"] = (batch2[fk] != 0)

                batch2["mask"] = view_mask
                _, ctx, *_ = nnet.encode_everything_test(indices, **batch2)
            else:
                ctx = ctx_full

            # 4) DPM masked loss
            s = seed528_dpm_masked_latent_mse_v1(
                nnet=nnet,
                z0_true=z0_true,
                context=ctx,
                view_mask_b3=view_mask,
                sample_steps=sample_steps,
                repeats=repeats,
                agg=agg,
                seed=seed_base + bi * 1000 + 10007 * k
            )  # [B]
            per_mask_scores.append(s)

        per_mask_scores = torch.stack(per_mask_scores, dim=0)  # [K,B]

        # 5) 聚合成每样本一个分数
        if mask_agg == "min":
            s_final = per_mask_scores.min(dim=0).values
        elif mask_agg == "mean":
            s_final = per_mask_scores.mean(dim=0)
        else:
            raise ValueError(f"mask_agg must be min/mean, got {mask_agg}")

        all_final.append(s_final.detach().cpu())
        if return_per_mask:
            all_per_mask.append(per_mask_scores.detach().cpu())

        if verbose_every and (bi % verbose_every == 0):
            print(f"[seed528-audit-v2] batch={bi} B={B} K={K} "
                  f"mean={float(s_final.mean()):.6f} mask_mode={mask_mode} mask_agg={mask_agg} strict_context={strict_context}")

    if len(all_final) == 0:
        if return_per_mask:
            return np.array([], dtype=np.float32), None, patterns_cache
        return np.array([], dtype=np.float32)

    final_losses = torch.cat(all_final, dim=0).numpy()

    if not return_per_mask:
        return final_losses

    # per_mask_losses: [K, N]
    # 拼接batch维：all_per_mask is list of [K,B]
    per_mask_losses = torch.cat(all_per_mask, dim=1).numpy()  # [K, N]
    return final_losses, per_mask_losses, patterns_cache


# ============================================================
# (5) End-to-end：跑 train/test losses
#     NEW: return_per_mask（分别mask + 聚合）
# ============================================================
def seed528_run_complete3v_dpm_loss_v2(
    nnet,
    diff_config,
    config_health,
    hp_dataset,
    tokenizers,
    convert_input_fn,
    keys_map=None,
    min_len=1,
    sample_steps=20,
    repeats=4,
    agg="mean",
    strict_context=True,
    max_batches_train=None,
    max_batches_test=None,
    seed_base=528,
    mask_mode="random_one",
    mask_agg="min",
    quantiles=(0.50, 0.90, 0.99, 0.999),
    return_per_mask=False
):
    device = torch.device("cuda:" + str(diff_config["device"]) if torch.cuda.is_available() else "cpu")
    nnet.eval().to(device)

    dataset = get_dataset(config_health["DATASET"], aug=True, samples=hp_dataset, config=diff_config)
    train_ds = dataset.get_split(split="train", labeled=True)
    test_ds  = dataset.get_split(split="test",  labeled=True)

    train_c = seed528_complete3v_subset_v1(train_ds, min_len=min_len, keys_map=keys_map)
    test_c  = seed528_complete3v_subset_v1(test_ds,  min_len=min_len, keys_map=keys_map)

    train_loader = DataLoader(
        train_c,
        batch_size=diff_config["train"]["batch_size"],
        shuffle=False, drop_last=False,
        num_workers=8, pin_memory=True, persistent_workers=True,
        collate_fn=collate_fn_dict
    )
    test_loader = DataLoader(
        test_c,
        batch_size=diff_config["sample"]["mini_batch_size"],
        shuffle=False, drop_last=False,
        num_workers=8, pin_memory=True, persistent_workers=True,
        collate_fn=collate_fn_dict
    )

    if return_per_mask:
        train_losses, train_permask, patterns = seed528_collect_losses_complete3v_dpm_v2(
            train_loader, nnet, convert_input_fn,
            sample_steps=sample_steps, repeats=repeats, agg=agg,
            strict_context=strict_context,
            max_batches=max_batches_train,
            seed_base=seed_base,
            mask_mode=mask_mode,
            mask_agg=mask_agg,
            return_per_mask=True
        )
        test_losses, test_permask, _ = seed528_collect_losses_complete3v_dpm_v2(
            test_loader, nnet, convert_input_fn,
            sample_steps=sample_steps, repeats=repeats, agg=agg,
            strict_context=strict_context,
            max_batches=max_batches_test,
            seed_base=seed_base + 100000,
            mask_mode=mask_mode,
            mask_agg=mask_agg,
            return_per_mask=True
        )
    else:
        train_losses = seed528_collect_losses_complete3v_dpm_v2(
            train_loader, nnet, convert_input_fn,
            sample_steps=sample_steps, repeats=repeats, agg=agg,
            strict_context=strict_context,
            max_batches=max_batches_train,
            seed_base=seed_base,
            mask_mode=mask_mode,
            mask_agg=mask_agg,
            return_per_mask=False
        )
        test_losses = seed528_collect_losses_complete3v_dpm_v2(
            test_loader, nnet, convert_input_fn,
            sample_steps=sample_steps, repeats=repeats, agg=agg,
            strict_context=strict_context,
            max_batches=max_batches_test,
            seed_base=seed_base + 100000,
            mask_mode=mask_mode,
            mask_agg=mask_agg,
            return_per_mask=False
        )
        train_permask, test_permask, patterns = None, None, None

    def _stats(arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return {"mean": 0.0, **{f"q{q}": 0.0 for q in quantiles}}
        out = {"mean": float(arr.mean())}
        for q in quantiles:
            out[f"q{q}"] = float(np.quantile(arr, q))
        return out

    summary = {
        "N_train": int(train_losses.size),
        "N_test": int(test_losses.size),
        "seed": int(seed_base),
        "steps": int(sample_steps),
        "repeats": int(repeats),
        "agg": str(agg),
        "strict_context": bool(strict_context),
        "mask_mode": str(mask_mode),
        "mask_agg": str(mask_agg),
        "train": _stats(train_losses),
        "test": _stats(test_losses),
    }

    print("\n===== seed528 audit: complete3v -> mask -> DPM -> masked-loss (v2) =====")
    print("N_train_complete =", summary["N_train"], " N_test_complete =", summary["N_test"])
    print("seed =", summary["seed"], "steps =", summary["steps"], "repeats =", summary["repeats"],
          "agg =", summary["agg"], "strict_context =", summary["strict_context"],
          "mask_mode =", summary["mask_mode"], "mask_agg =", summary["mask_agg"])
    print("train mean =", summary["train"]["mean"], "test mean =", summary["test"]["mean"])
    for q in quantiles:
        print(f"train q{q} = {summary['train'][f'q{q}']:.6f} | test q{q} = {summary['test'][f'q{q}']:.6f}")

    out = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "summary": summary,
        "train_loader": train_loader,
        "test_loader": test_loader,
    }
    if return_per_mask:
        out.update({
            "train_permask": train_permask,  # [K,Ntr]
            "test_permask": test_permask,    # [K,Nts]
            "mask_patterns": patterns,       # [K,3]
        })
    return out


# ============================================================
# (6) Attack-P
# ============================================================
def seed528_attackP_from_maskedloss_v1(train_losses, test_losses, alphas=(0.2, 0.1, 0.05, 0.01, 0.001)):
    train_losses = np.asarray(train_losses)
    test_losses = np.asarray(test_losses)
    res = {}
    if train_losses.size == 0 or test_losses.size == 0:
        return res

    for a in alphas:
        thr = float(np.quantile(test_losses, a))
        tpr = float((train_losses <= thr).mean())
        fpr = float((test_losses  <= thr).mean())
        res[a] = {
            "threshold": thr,
            "TPR": tpr,
            "FPR": fpr,
            "ADV": float(tpr - fpr),
            "N_train": int(train_losses.size),
            "N_test": int(test_losses.size),
        }
    return res


def seed528_run_complete3v_dpm_then_attackP_v2(
    nnet,
    diff_config,
    config_health,
    hp_dataset,
    tokenizers,
    convert_input_fn,
    keys_map=None,
    min_len=1,
    sample_steps=20,
    repeats=4,
    agg="mean",
    strict_context=True,
    max_batches_train=None,
    max_batches_test=None,
    seed_base=528,
    mask_mode="random_one",
    mask_agg="min",
    alphas=(0.2, 0.1, 0.05, 0.01, 0.001),
    return_losses=False,
    return_per_mask=False
):
    out = seed528_run_complete3v_dpm_loss_v2(
        nnet=nnet,
        diff_config=diff_config,
        config_health=config_health,
        hp_dataset=hp_dataset,
        tokenizers=tokenizers,
        convert_input_fn=convert_input_fn,
        keys_map=keys_map,
        min_len=min_len,
        sample_steps=sample_steps,
        repeats=repeats,
        agg=agg,
        strict_context=strict_context,
        max_batches_train=max_batches_train,
        max_batches_test=max_batches_test,
        seed_base=seed_base,
        mask_mode=mask_mode,
        mask_agg=mask_agg,
        return_per_mask=return_per_mask
    )

    mia_res = seed528_attackP_from_maskedloss_v1(
        train_losses=out["train_losses"],
        test_losses=out["test_losses"],
        alphas=alphas
    )

    print("\n===== Attack-P results (seed528 / DPM-loss) =====")
    print("N_train_scores:", int(out["train_losses"].size), "N_test_scores:", int(out["test_losses"].size))
    print("MIA results:", mia_res)

    if return_losses:
        return mia_res, out
    return mia_res


def seed528_print_attackP_table_v1(mia_res):
    if not mia_res:
        print("MIA results: {}")
        return
    for a in sorted(mia_res.keys(), reverse=True):
        r = mia_res[a]
        print(
            f"alpha={a:>6} | thr={r['threshold']:.6f} | "
            f"TPR={r['TPR']:.6f} | FPR={r['FPR']:.6f} | ADV={r['ADV']:.6f} | "
            f"Ntr={r['N_train']} Nts={r['N_test']}"
        )


# ============================================================
# (7) 单条测试样本：loss vs steps
# ============================================================
def _slice_batch_one(batch: Dict[str, Any], i: int) -> Dict[str, Any]:
    one = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            one[k] = v[i:i+1]
        elif isinstance(v, list):
            one[k] = v[i:i+1]
        else:
            one[k] = v
    return one


@torch.no_grad()
def seed528_one_sample_loss_vs_steps(
    nnet,
    indices_one,
    batch_one,
    convert_input_fn,
    steps_list=(5, 10, 20, 50),
    repeats=16,
    agg="min",
    strict_context=True,
    mask_mode="all_single",   # random_one/all_single/all_nonempty/all_views 都支持
    mask_agg="min",
    seed_base=528
):
    indices_one, batch_one = convert_input_fn((indices_one, batch_one), mask_static=True)
    z0_true, ctx_full, *_ = nnet.encode_everything_test(indices_one, **batch_one)
    B = z0_true.size(0)
    assert B == 1, "seed528_one_sample_loss_vs_steps 建议一次只喂 1 条样本(B=1)"

    device = z0_true.device

    # patterns: [K,3]
    if mask_mode == "random_one":
        patterns = seed528_viewmask_one_v1(B=B, device=device, seed=seed_base)  # [1,3]
        view_masks = patterns.unsqueeze(0)  # [1,1,3]
    else:
        patterns = seed528_enumerate_viewmasks_v2(mask_mode, device=device)  # [K,3]
        view_masks = patterns.unsqueeze(1).expand(-1, B, -1).contiguous()    # [K,1,3]

    K = view_masks.size(0)
    feature_keys = ["conditions", "drugs", "procedures"]

    out_curve = {}
    for steps in steps_list:
        per_mask_scores = []

        for k in range(K):
            view_mask = view_masks[k]  # [1,3]

            if strict_context:
                batch2 = {}
                for kk, vv in batch_one.items():
                    batch2[kk] = vv.clone() if torch.is_tensor(vv) else vv

                for j, fk in enumerate(feature_keys):
                    m = view_mask[:, j]
                    if m.any():
                        batch2[fk][m] = 0
                        batch2[fk + "_mask"] = (batch2[fk] != 0)

                batch2["mask"] = view_mask
                _, ctx, *_ = nnet.encode_everything_test(indices_one, **batch2)
            else:
                ctx = ctx_full

            s = seed528_dpm_masked_latent_mse_v1(
                nnet=nnet,
                z0_true=z0_true,
                context=ctx,
                view_mask_b3=view_mask,
                sample_steps=int(steps),
                repeats=int(repeats),
                agg=agg,
                seed=seed_base + 99991 * int(steps) + 10007 * k
            )  # [1]
            per_mask_scores.append(s)

        per_mask_scores = torch.stack(per_mask_scores, dim=0)  # [K,1]
        if mask_agg == "min":
            s_final = per_mask_scores.min(dim=0).values
        elif mask_agg == "mean":
            s_final = per_mask_scores.mean(dim=0)
        else:
            raise ValueError(f"mask_agg must be min/mean, got {mask_agg}")

        out_curve[int(steps)] = float(s_final.item())

    return out_curve


# ============================================================
# (8) 加权Jaccard / wrec / wprec 审计（Gen(new_*) vs Train(real)）
# ============================================================
def _flatten_tokens(v):
    if v is None:
        return []
    if isinstance(v, (tuple, set)):
        v = list(v)
    if not isinstance(v, list):
        return [v]
    if len(v) == 0:
        return []
    # 2D / 3D flatten
    if isinstance(v[0], list):
        out = []
        for row in v:
            if isinstance(row, list) and len(row) > 0 and isinstance(row[0], list):
                # 3D
                for rr in row:
                    out.extend(rr)
            else:
                out.extend(row)
        return out
    return list(v)


def _get_tokens(sample: Dict[str, Any], feat: str, use: str):
    if use == "raw":
        v = sample.get(feat + "_raw", None)
        if v is None:
            v = sample.get(feat, None)
    elif use == "new":
        v = sample.get("new_" + feat, None)
        if v is None:
            v = sample.get(feat, None)
    else:
        v = sample.get(feat, None)
    return set(_flatten_tokens(v))


def build_idf_weights(train_samples: List[Dict[str, Any]],
                      feats=("conditions", "procedures", "drugs"),
                      train_use="raw"):
    df = defaultdict(int)
    N = len(train_samples)
    for s in train_samples:
        toks = set()
        for f in feats:
            toks |= _get_tokens(s, f, use=train_use)
        for t in toks:
            df[t] += 1

    w = {}
    for t, c in df.items():
        w[t] = math.log((N + 1.0) / (c + 1.0)) + 1.0
    return w


def audit_weighted_nn_v1(
    train_samples: List[Dict[str, Any]],
    query_samples: List[Dict[str, Any]],
    feats=("conditions", "procedures", "drugs"),
    query_use="new",
    train_use="raw",
    score_key="wrec",
    topk_tokens: Optional[int] = None,
    topn=30,
    overlap_head=20,
    th_wrec=0.9,
    th_wj=0.9,
    th_wprec=0.9,
    print_details=True
):
    w = build_idf_weights(train_samples, feats=feats, train_use=train_use)

    posting = defaultdict(list)
    train_sets = []
    train_wsum = []
    for i, s in enumerate(train_samples):
        tset = set()
        for f in feats:
            tset |= _get_tokens(s, f, use=train_use)
        train_sets.append(tset)
        tw = sum(w.get(x, 1.0) for x in tset)
        train_wsum.append(tw)
        for t in tset:
            posting[t].append(i)

    rows = []
    for qi, qs in enumerate(query_samples):
        qset = set()
        for f in feats:
            qset |= _get_tokens(qs, f, use=query_use)

        if topk_tokens is not None and topk_tokens > 0 and len(qset) > topk_tokens:
            q_sorted = sorted(qset, key=lambda x: w.get(x, 1.0), reverse=True)
            qset = set(q_sorted[:topk_tokens])

        qw = sum(w.get(x, 1.0) for x in qset)

        inter_w = defaultdict(float)
        for t in qset:
            wt = w.get(t, 1.0)
            for ti in posting.get(t, []):
                inter_w[ti] += wt

        best = None
        best_inter = set()
        for ti, wi in inter_w.items():
            tw = train_wsum[ti]
            wu = (qw + tw - wi)
            wj = wi / (wu + 1e-12)
            wrec = wi / (qw + 1e-12)
            wprec = wi / (tw + 1e-12)

            cand = {"q": qi, "nn_train": ti, "wj": float(wj), "wrec": float(wrec), "wprec": float(wprec)}
            if best is None or cand[score_key] > best[score_key]:
                best = cand
                best_inter = qset & train_sets[ti]

        if best is None:
            best = {"q": qi, "nn_train": -1, "wj": 0.0, "wrec": 0.0, "wprec": 0.0}
            best_inter = set()

        overlap_sorted = sorted(list(best_inter), key=lambda x: w.get(x, 1.0), reverse=True)
        best["overlap"] = overlap_sorted
        rows.append(best)

    wj_list = [r["wj"] for r in rows]
    wrec_list = [r["wrec"] for r in rows]
    wprec_list = [r["wprec"] for r in rows]

    summary = {
        "N_query": len(query_samples),
        "N_train": len(train_samples),
        "feats": feats,
        "query_use": query_use,
        "train_use": train_use,
        "score_key": score_key,
        "wj_mean": float(np.mean(wj_list)) if wj_list else 0.0,
        "wj_q50": float(np.quantile(wj_list, 0.50)) if wj_list else 0.0,
        "wj_q90": float(np.quantile(wj_list, 0.90)) if wj_list else 0.0,
        "wj_q99": float(np.quantile(wj_list, 0.99)) if wj_list else 0.0,
        "wrec_mean": float(np.mean(wrec_list)) if wrec_list else 0.0,
        "wrec_q99": float(np.quantile(wrec_list, 0.99)) if wrec_list else 0.0,
        "wprec_mean": float(np.mean(wprec_list)) if wprec_list else 0.0,
        "wprec_q99": float(np.quantile(wprec_list, 0.99)) if wprec_list else 0.0,
        "rate_wj_ge_th": float(np.mean([x >= th_wj for x in wj_list])) if wj_list else 0.0,
        "rate_wrec_ge_th": float(np.mean([x >= th_wrec for x in wrec_list])) if wrec_list else 0.0,
        "rate_wprec_ge_th": float(np.mean([x >= th_wprec for x in wprec_list])) if wprec_list else 0.0,
        "th_wj": float(th_wj),
        "th_wrec": float(th_wrec),
        "th_wprec": float(th_wprec),
        "topk_tokens": int(topk_tokens) if topk_tokens is not None else None,
    }

    top = sorted(rows, key=lambda r: r[score_key], reverse=True)[:topn]

    if print_details:
        print("\n===== Gen(new_*) vs Train(real) leakage audit (WEIGHTED) =====")
        print("summary =", summary)
        print(f"\n--- TOP-{topn} by {score_key} (train coverage /复刻程度) ---")
        for rank, r in enumerate(top):
            oh = r["overlap"][:overlap_head]
            print(f"[{rank}] q={r['q']} {score_key}={r[score_key]:.6f} wprec={r['wprec']:.6f} wj={r['wj']:.6f} nn_train={r['nn_train']}")
            print("  overlap(head):", oh)

    return summary, top, rows


# ============================================================
import random
import torch

@torch.no_grad()
def pick_random_test_sample_and_curve(
    nnet,
    out_dict,                 # seed528_run_complete3v_dpm_loss_v2(...) 的返回 out
    convert_input_fn,
    steps_list=(5, 10, 20, 50),
    repeats=16,
    agg="min",
    strict_context=True,
    mask_mode="all_single",   # 或 all_views / all_nonempty / random_one
    mask_agg="min",
    seed=528
):
    """
    从 out_dict["test_loader"] 随机取一个样本，返回：
      - chosen_global: 该样本在“遍历到的测试样本流”里的序号（近似global index）
      - chosen_in_batch: 在batch里的index
      - curve: {steps: loss}
    """
    test_loader = out_dict["test_loader"]

    # 1) 随机挑一个 batch（简单做法：随机跳过前 n 个 batch）
    rng = random.Random(seed)
    n_skip = rng.randint(0, max(0, len(test_loader) - 1))

    it = iter(test_loader)
    for _ in range(n_skip):
        next(it)

    indices, batch = next(it)  # 一个随机batch

    # 2) 在这个 batch 里随机挑一条样本
    B = len(indices) if isinstance(indices, (list, tuple)) else int(indices.shape[0])
    i = rng.randint(0, B - 1)

    batch_one = _slice_batch_one(batch, i)

    # indices_one：兼容 list / tensor
    if isinstance(indices, (list, tuple)):
        indices_one = indices[i:i+1]
    else:
        indices_one = indices[i:i+1]

    # 3) 跑单样本 steps 曲线
    curve = seed528_one_sample_loss_vs_steps(
        nnet=nnet,
        indices_one=indices_one,
        batch_one=batch_one,
        convert_input_fn=convert_input_fn,
        steps_list=steps_list,
        repeats=repeats,
        agg=agg,
        strict_context=strict_context,
        mask_mode=mask_mode,
        mask_agg=mask_agg,
        seed_base=seed
    )

    chosen_global = n_skip * (test_loader.batch_size or B) + i
    return chosen_global, i, curve


############################# 在训练样本上选出5个极端样本情况 #############################
# -*- coding: utf-8 -*-

import os
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from diffuser import diff_utils as utils
from diffuser.diff_dataset import get_dataset, collate_fn_dict
from diffuser.dpm import NoiseScheduleVP, DPM_Solver


# ============================================================
# 0) 你给出的 convert_input_fn（原样保留）
# ============================================================
import copy

def build_convert_input_fn_health_v11(tokenizers, device, config_health):
    feature_keys = ['conditions', 'drugs', 'procedures']

    def _maybe_encode_2d(x, tok):
        # x: List[List[str]] OR torch.Tensor (already ids)
        if torch.is_tensor(x):
            return x.to(device)
        # pyhealth tokenizer expects list[list[token]]
        return torch.tensor(tok.batch_encode_2d(x), dtype=torch.long, device=device)

    def _maybe_encode_3d(x, tok):
        if torch.is_tensor(x):
            return x.to(device)
        return torch.tensor(tok.batch_encode_3d(x), dtype=torch.long, device=device)

    def convert_input_fn(batch_data, mask_static=True):
        indices, batch = batch_data

        for fk in feature_keys:
            # 兼容：batch[fk] 可能已经是 Tensor（否则就 encode）
            batch[fk] = _maybe_encode_2d(batch[fk], tokenizers[fk])
            batch[fk + '_hist'] = _maybe_encode_3d(batch[fk + '_hist'], tokenizers[fk])

            # comp：保留 origin（原始 tokens）用于后续 GT set
            origin = batch[fk + '_comp']
            batch[fk + '_comp'] = _maybe_encode_2d(origin, tokenizers[fk])

            # masks
            batch[fk + '_mask'] = batch[fk] != 0
            batch[fk + '_hist_mask'] = batch[fk + '_hist'] != 0
            batch[fk + '_comp_mask'] = batch[fk + '_comp'] != 0

            # 下面这两个字段如果后续不用，也没关系；如果用，就尽量保持为 python list
            # 如果 origin 已经是 Tensor，就无法再 batch_encode_2d(origin)
            if isinstance(origin, list):
                batch[fk + '_comps_origin'] = origin
                batch[fk + '_comps'] = tokenizers[fk].batch_encode_2d(origin, padding=False, truncation=False)
            else:
                # fallback
                batch[fk + '_comps_origin'] = [[] for _ in range(batch[fk].shape[0])]
                batch[fk + '_comps'] = [[] for _ in range(batch[fk].shape[0])]

            batch['has_note'] = False

        # mask
        if mask_static:
            if 'mask' in batch and torch.is_tensor(batch['mask']):
                batch['mask'] = batch['mask'].to(device=device, dtype=torch.bool)
            elif 'mask' in batch:
                batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
            else:
                B = batch['conditions'].shape[0]
                batch['mask'] = torch.zeros(B, 3, dtype=torch.bool, device=device)
        else:
            m = 3
            batch['mask'] = utils.generate_mask(batch['conditions'].shape[0], m=m).to(device=device)

        # mask_hist
        aligned_data = [
            batch[fk + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1) for fk in feature_keys
        ]
        batch['mask_hist'] = torch.cat(aligned_data, dim=-1)

        return indices, batch

    return convert_input_fn


# ============================================================
# 1) complete3v 子集（原样保留）
# ============================================================
def seed528_complete3v_subset_v1(ds_split, min_len=1, keys_map=None):
    if keys_map is None:
        keys_map = {
            "conditions": ["conditions_comp", "conditions_comps", "conditions"],
            "procedures": ["procedures_comp", "procedures_comps", "procedures"],
            "drugs":      ["drugs_comp", "drugs_comps", "drugs"],
        }

    def _unwrap(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, tuple):
            for t in reversed(x):
                if isinstance(t, dict):
                    return t
        return x

    def _pick_list(s, cands):
        if not isinstance(s, dict):
            return []
        for k in cands:
            v = s.get(k, None)
            if v is not None:
                return v
        return []

    keep = []
    for i in range(len(ds_split)):
        s = _unwrap(ds_split[i])

        c = _pick_list(s, keys_map["conditions"])
        p = _pick_list(s, keys_map["procedures"])
        d = _pick_list(s, keys_map["drugs"])

        if (len(c) >= min_len) and (len(p) >= min_len) and (len(d) >= min_len):
            keep.append(i)

    return Subset(ds_split, keep)


# ============================================================
# 2) mask patterns（含 all_views）
# ============================================================
def seed528_enumerate_viewmasks_v2(mode: str, device):
    """
    返回 patterns: [K,3] bool
      - all_single: 3个单掩码
      - all_nonempty: 7个非空组合掩码
      - all_views: 1个全掩码 [1,1,1]
    """
    if mode == "all_single":
        return torch.eye(3, dtype=torch.bool, device=device)  # [3,3]
    elif mode == "all_nonempty":
        masks = []
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    if a == 0 and b == 0 and c == 0:
                        continue
                    masks.append([a, b, c])
        return torch.tensor(masks, dtype=torch.bool, device=device)  # [7,3]
    elif mode == "all_views":
        return torch.tensor([[1, 1, 1]], dtype=torch.bool, device=device)  # [1,3]
    else:
        raise ValueError(f"Unknown mask mode: {mode}")


# ============================================================
# 3) 小工具：从 dataset sample 抽 tokens（用于统计/选极端样本）
# ============================================================
def _unwrap_sample(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, tuple):
        for t in reversed(x):
            if isinstance(t, dict):
                return t
    return x

def _pick_list(s, cands):
    for k in cands:
        v = s.get(k, None)
        if v is not None:
            return v
    return []

def _flatten_tokens(v):
    if v is None:
        return []
    if not isinstance(v, list):
        return [v]
    out = []
    for x in v:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out

def get_tokens_from_sample(sample_dict, view, keys_map):
    v = _pick_list(sample_dict, keys_map[view])
    return _flatten_tokens(v)  # list (可重复)


# ============================================================
# 4) TRAIN complete3v 上做 code 统计：DF/TF/IDF
# ============================================================
def build_code_stats_complete3v(ds_subset, keys_map):
    tf_view = {v: Counter() for v in ["conditions", "drugs", "procedures"]}
    df_view = {v: Counter() for v in ["conditions", "drugs", "procedures"]}
    df_all = Counter()

    N = len(ds_subset)
    for j in range(N):
        s = _unwrap_sample(ds_subset[j])
        union_all = set()

        for v in ["conditions", "drugs", "procedures"]:
            toks = get_tokens_from_sample(s, v, keys_map)
            tf_view[v].update(toks)
            df_view[v].update(set(toks))
            union_all |= set(toks)

        df_all.update(union_all)

    idf = {t: math.log((N + 1.0) / (c + 1.0)) + 1.0 for t, c in df_all.items()}

    return {"N": N, "tf_view": tf_view, "df_view": df_view, "df_all": df_all, "idf": idf}

def print_code_stats(stats, topk=30, rare_df_le=2):
    print("\n===== [1] TRAIN complete3v code stats =====")
    print("N_complete3v =", stats["N"])
    for v in ["conditions", "drugs", "procedures"]:
        common = stats["df_view"][v].most_common(topk)
        print(f"\n--- view={v} DF top-{topk} (common codes) ---")
        print(common[:topk])

    # rare tokens（全局 df <= rare_df_le）
    rare = [(t, c) for t, c in stats["df_all"].items() if c <= rare_df_le]
    rare.sort(key=lambda x: x[1])
    print(f"\n--- global rare codes: DF <= {rare_df_le}  (show head 50) ---")
    print(rare[:50])

    # top-idf tokens
    idf_sorted = sorted(stats["idf"].items(), key=lambda kv: kv[1], reverse=True)
    print("\n--- global top-idf codes (rarest) head 50 ---")
    print(idf_sorted[:50])


# ============================================================
# 5) 给每个训练样本打分 + 选 5 条极端样本
# ============================================================
def score_samples(ds_subset, idf, keys_map):
    rows = []
    for j in range(len(ds_subset)):
        s = _unwrap_sample(ds_subset[j])

        union = set()
        lens = {}
        for v in ["conditions", "drugs", "procedures"]:
            toks = get_tokens_from_sample(s, v, keys_map)
            union |= set(toks)
            lens[v] = len(toks)

        rarity_sum = float(sum(idf.get(t, 1.0) for t in union))
        avg_idf = rarity_sum / max(len(union), 1)
        total_len = sum(lens.values())

        rows.append({
            "subset_j": j,
            "rarity_sum": rarity_sum,
            "avg_idf": avg_idf,
            "total_len": total_len,
            "lens": lens,
            "uniq": len(union),
        })
    return rows

def pick_5_extremes(rows):
    rare_sum = max(rows, key=lambda r: r["rarity_sum"])
    rare_avg = max(rows, key=lambda r: r["avg_idf"])
    common   = min(rows, key=lambda r: r["avg_idf"])
    long_    = max(rows, key=lambda r: r["total_len"])
    short_   = min([r for r in rows if r["total_len"] > 0], key=lambda r: r["total_len"])

    picked = []
    seen = set()
    for r in [rare_sum, rare_avg, common, long_, short_]:
        if r["subset_j"] not in seen:
            picked.append(r); seen.add(r["subset_j"])

    # 不足 5 则补齐（按 rarity_sum 再排）
    if len(picked) < 5:
        for r in sorted(rows, key=lambda r: r["rarity_sum"], reverse=True):
            if r["subset_j"] not in seen:
                picked.append(r); seen.add(r["subset_j"])
            if len(picked) == 5:
                break
    return picked[:5]

def print_picked(picked):
    print("\n===== [2] Picked 5 extreme TRAIN samples (complete3v) =====")
    for i, r in enumerate(picked):
        print(f"[{i}] subset_j={r['subset_j']}  rarity_sum={r['rarity_sum']:.3f}  "
              f"avg_idf={r['avg_idf']:.3f}  total_len={r['total_len']}  lens={r['lens']}  uniq={r['uniq']}")


# ============================================================
# 6) 取 subset 中单条样本 batch (B=1)
# ============================================================
def get_one_batch_from_subset(ds_subset, subset_j, collate_fn):
    one_ds = Subset(ds_subset, [subset_j])
    loader = DataLoader(one_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    indices, batch = next(iter(loader))
    return indices, batch


# ============================================================
# 7) strict_context: 真正把被 mask 的 view 从输入里抹掉（防止“mask 仍泄露GT”）
# ============================================================
def _recompute_view_masks(batch: Dict[str, Any], feature_keys: List[str]):
    for fk in feature_keys:
        if fk in batch and torch.is_tensor(batch[fk]):
            batch[fk + "_mask"] = (batch[fk] != 0)
        if fk + "_hist" in batch and torch.is_tensor(batch[fk + "_hist"]):
            batch[fk + "_hist_mask"] = (batch[fk + "_hist"] != 0)
        if fk + "_comp" in batch and torch.is_tensor(batch[fk + "_comp"]):
            batch[fk + "_comp_mask"] = (batch[fk + "_comp"] != 0)

    # mask_hist
    aligned = []
    for fk in feature_keys:
        if fk + "_hist_mask" in batch and torch.is_tensor(batch[fk + "_hist_mask"]):
            aligned.append(batch[fk + "_hist_mask"].sum(dim=-1).unsqueeze(-1))
    if len(aligned) == 3:
        batch["mask_hist"] = torch.cat(aligned, dim=-1)

def _apply_strict_view_mask(batch: Dict[str, Any], view_mask_b3: torch.Tensor,
                            feature_keys=("conditions","drugs","procedures")):
    """
    把被mask的view在 batch 里尽可能清干净：
    - fk / fk_hist / fk_comp (tensor) -> 置0
    - fk_comps_origin (list) -> 置空（防止 python list 泄露）
    """
    B = view_mask_b3.shape[0]
    device = view_mask_b3.device

    for j, fk in enumerate(feature_keys):
        m = view_mask_b3[:, j]  # [B] bool
        if not bool(m.any()):
            continue

        # 1) 主输入 fk
        if fk in batch and torch.is_tensor(batch[fk]):
            batch[fk][m] = 0

        # 2) hist
        hk = fk + "_hist"
        if hk in batch and torch.is_tensor(batch[hk]):
            batch[hk][m] = 0

        # 3) comp
        ck = fk + "_comp"
        if ck in batch and torch.is_tensor(batch[ck]):
            batch[ck][m] = 0

        # 4) 可能的 python list 原始 tokens（如果 encoder 不用它也无害）
        ok = fk + "_comps_origin"
        if ok in batch and isinstance(batch[ok], list):
            for bi in torch.where(m)[0].tolist():
                batch[ok][bi] = []  # 清空

        # 5) 也可能有 fk_comps (list of list)（如果存在）
        ok2 = fk + "_comps"
        if ok2 in batch and isinstance(batch[ok2], list):
            for bi in torch.where(m)[0].tolist():
                batch[ok2][bi] = []

    batch["mask"] = view_mask_b3.to(device)
    _recompute_view_masks(batch, list(feature_keys))


# ============================================================
# 8) DPM 反向补全：repeats 次随机初始化（只在 mask 的 view 位置注入噪声）
# ============================================================
@torch.no_grad()
def dpm_inpaint_repeats_get_mse_and_zhat(
    nnet,
    z0_true: torch.Tensor,        # [B,3,D]
    context: torch.Tensor,
    view_mask_b3: torch.Tensor,   # [B,3] bool
    sample_steps=20,
    repeats=16,
    seed=528
):
    device = z0_true.device
    B, M, D = z0_true.shape

    mask_b3 = view_mask_b3.to(device)
    mask_b3d = mask_b3.unsqueeze(-1).expand(B, M, D)
    denom = mask_b3d.float().sum(dim=(1, 2)).clamp_min(1.0)

    betas = utils.stable_diffusion_beta_schedule()
    betas_t = torch.tensor(betas, device=device).float()
    N = betas_t.numel()
    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=betas_t)

    def model_fn(x, t_cont):
        t = t_cont * N
        return nnet(x, t, context=context)

    dpm = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)

    mses = []
    zhats = []

    for r in range(repeats):
        cuda_devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(seed + 10007 * r)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + 10007 * r)
            noise = torch.randn_like(z0_true)

        z_init = torch.where(mask_b3d, noise, z0_true)  # 只替换 mask view 的 latent 为噪声
        z_hat = dpm.sample(z_init, steps=sample_steps, eps=1.0 / N, T=1.0)

        diff2 = (z_hat - z0_true) ** 2
        mse_masked = (diff2 * mask_b3d.float()).sum(dim=(1, 2)) / denom  # [B]

        mses.append(mse_masked.detach().cpu())
        zhats.append(z_hat.detach().cpu())

    mses = torch.stack(mses, dim=0)     # [R,B]
    zhats = torch.stack(zhats, dim=0)   # [R,B,3,D]
    return mses, zhats


# ============================================================
# 9) decode latent -> tokens(set) + IDF加权指标（仅评估被mask的view）
# ============================================================
def _wsum(S, idf):
    return float(sum(idf.get(t, 1.0) for t in S))

def weighted_metrics(pred_set, gt_set, idf):
    inter = pred_set & gt_set
    u = pred_set | gt_set
    wi = _wsum(inter, idf)
    wg = _wsum(gt_set, idf)
    wp = _wsum(pred_set, idf)
    wu = _wsum(u, idf)
    return {
        "wrec": wi / (wg + 1e-12),
        "wprec": wi / (wp + 1e-12),
        "wj": wi / (wu + 1e-12),
    }

@torch.no_grad()
def decode_latent_to_token_sets(z_b3d, nnet, tokenizers, threshold=0.5):
    """
    z_b3d: [B,3,D] -> 返回 dict: {feature_key: List[set(tokens)]}  (长度 B)
    注意：feature_keys 顺序必须与你的 latent view 顺序一致
    """
    feature_keys = ["conditions", "drugs", "procedures"]  # 与你 MIA/DPM-loss 代码一致
    out = {}

    for idx, fk in enumerate(feature_keys):
        logits = nnet.get_logits(z_b3d[:, idx, :], fk)  # [B,V] or similar
        prob = torch.sigmoid(logits).detach().cpu().numpy()

        max_idx = np.argmax(prob, axis=1)
        bin_ = (prob >= threshold).astype(np.int64)

        indices = []
        for b in range(bin_.shape[0]):
            on = (np.where(bin_[b] == 1)[0] + 2).tolist()
            if len(on) < 1:
                on = [int(max_idx[b] + 2)]
            indices.append(on)

        toks_2d = tokenizers[fk].batch_decode_2d(indices)  # List[List[token]]
        out[fk] = [set(row) for row in toks_2d]
    return out

def get_gt_token_sets_from_batch(batch_conv, tokenizers):
    """
    从 convert_input_fn 之后的 batch 里取 GT tokens(set)
    优先用 fk_comps_origin（编码前原始tokens）
    """
    feature_keys = ["conditions", "drugs", "procedures"]
    out = {}

    for fk in feature_keys:
        ok = fk + "_comps_origin"
        if ok in batch_conv and isinstance(batch_conv[ok], list) and len(batch_conv[ok]) > 0:
            # batch_conv[ok][0] 是该样本的原始 tokens(list)
            out[fk] = set(_flatten_tokens(batch_conv[ok][0]))
            continue

        # fallback：从 comp tensor decode
        ck = fk + "_comp"
        if ck in batch_conv and torch.is_tensor(batch_conv[ck]):
            ids = [int(x) for x in batch_conv[ck][0].detach().cpu().tolist() if int(x) != 0]
            if len(ids) == 0:
                out[fk] = set()
            else:
                toks = tokenizers[fk].batch_decode_2d([ids])[0]
                out[fk] = set(toks)
            continue

        out[fk] = set()

    return out

def overlap_top_idf_tokens(pred_set, gt_set, idf, topk=15):
    inter = list(pred_set & gt_set)
    inter.sort(key=lambda x: idf.get(x, 1.0), reverse=True)
    return inter[:topk]


# ============================================================
# 10) 单样本：固定 mask -> repeats 补全 -> MSE + 离散指标（仅mask view）
# ============================================================
@torch.no_grad()
def eval_one_sample_impute(
    nnet,
    indices_one,
    batch_one_raw,
    convert_input_fn,
    tokenizers,
    idf,
    sample_steps=20,
    repeats=16,
    repeat_agg="mean",            # mean 或 min（min：取 repeats 中 MSE 最小那次）
    mask_mode="all_single",
    strict_context_mask=True,
    decode_threshold=0.5,
    seed_base=528,
    print_overlap_topk=15,
):
    """
    返回 per_pattern: List[dict]
      每个 dict 对应一个 view_mask pattern:
        - pattern: [3] (0/1)
        - mse_by_repeat: [R] float
        - mse_agg: float
        - disc_by_repeat: {fk: [R] metrics}
        - disc_agg: {fk: metrics}  (按 repeat_agg 聚合)
        - overlap_top: {fk: top-idf overlap tokens} (只对mask view)
    """
    feature_keys = ["conditions", "drugs", "procedures"]

    indices_one, batch_conv = convert_input_fn((indices_one, batch_one_raw), mask_static=True)
    z0_true, ctx_full, *_ = nnet.encode_everything_test(indices_one, **batch_conv)

    B = z0_true.size(0)
    assert B == 1, "建议一次只跑 1 条样本(B=1)"
    device = z0_true.device

    patterns = seed528_enumerate_viewmasks_v2(mask_mode, device=device)  # [K,3]
    K = patterns.size(0)

    gt_sets = get_gt_token_sets_from_batch(batch_conv, tokenizers)

    per_pattern = []
    for k in range(K):
        view_mask = patterns[k].unsqueeze(0).expand(B, -1).contiguous()  # [1,3]

        # strict_context：把被mask view 从输入里清掉，然后重编码 context
        if strict_context_mask:
            batch2 = {}
            for kk, vv in batch_conv.items():
                if torch.is_tensor(vv):
                    batch2[kk] = vv.clone()
                elif isinstance(vv, list):
                    batch2[kk] = list(vv)
                else:
                    batch2[kk] = vv
            _apply_strict_view_mask(batch2, view_mask, feature_keys=feature_keys)
            _, ctx, *_ = nnet.encode_everything_test(indices_one, **batch2)
        else:
            ctx = ctx_full

        # repeats 补全
        mses_rb, zhats_rb3d = dpm_inpaint_repeats_get_mse_and_zhat(
            nnet=nnet,
            z0_true=z0_true,
            context=ctx,
            view_mask_b3=view_mask,
            sample_steps=sample_steps,
            repeats=repeats,
            seed=seed_base + 10007 * k
        )
        # [R,1] -> [R]
        mse_by_repeat = mses_rb[:, 0].numpy().astype(np.float32)

        # 逐 repeat 做离散指标（只对 mask view）
        disc_by_repeat = {fk: [] for fk in feature_keys}
        overlap_top = {}

        for r in range(repeats):
            z_hat = zhats_rb3d[r].to(device=device, dtype=z0_true.dtype)  # [1,3,D]
            pred_sets_b = decode_latent_to_token_sets(z_hat, nnet, tokenizers, threshold=decode_threshold)
            # pred_sets_b[fk] 是 List[set] 长度1
            for j, fk in enumerate(feature_keys):
                if bool(view_mask[0, j].item()) is False:
                    continue  # 只评估被 mask 的 view
                pred_set = pred_sets_b[fk][0]
                gt_set = gt_sets[fk]
                m = weighted_metrics(pred_set, gt_set, idf)
                disc_by_repeat[fk].append(m)

        # 聚合 repeats：mean 或 min（以 MSE 为准选 repeat）
        if repeat_agg == "mean":
            mse_agg = float(np.mean(mse_by_repeat))
            disc_agg = {}
            for j, fk in enumerate(feature_keys):
                if not bool(view_mask[0, j].item()):
                    continue
                # 平均每个指标
                ws = disc_by_repeat[fk]
                disc_agg[fk] = {
                    "wrec": float(np.mean([x["wrec"] for x in ws])) if ws else 0.0,
                    "wprec": float(np.mean([x["wprec"] for x in ws])) if ws else 0.0,
                    "wj": float(np.mean([x["wj"] for x in ws])) if ws else 0.0,
                }
                # overlap 用“最后一次 repeat”不太合理，这里用“均值repeat”也没定义；改用“MSE最小的repeat”更稳
            best_r = int(np.argmin(mse_by_repeat))
        elif repeat_agg == "min":
            best_r = int(np.argmin(mse_by_repeat))
            mse_agg = float(mse_by_repeat[best_r])
            disc_agg = {}
            for j, fk in enumerate(feature_keys):
                if not bool(view_mask[0, j].item()):
                    continue
                if len(disc_by_repeat[fk]) == 0:
                    disc_agg[fk] = {"wrec": 0.0, "wprec": 0.0, "wj": 0.0}
                else:
                    # 注意：disc_by_repeat[fk] 的长度 == repeats（只要该view被mask）
                    disc_agg[fk] = disc_by_repeat[fk][best_r]
        else:
            raise ValueError("repeat_agg must be 'mean' or 'min'")

        # 用 best_r 的 z_hat 输出 overlap(top-idf) 作为可读证据
        z_best = zhats_rb3d[best_r].to(device=device, dtype=z0_true.dtype)
        pred_best = decode_latent_to_token_sets(z_best, nnet, tokenizers, threshold=decode_threshold)
        for j, fk in enumerate(feature_keys):
            if not bool(view_mask[0, j].item()):
                continue
            overlap_top[fk] = overlap_top_idf_tokens(pred_best[fk][0], gt_sets[fk], idf, topk=print_overlap_topk)

        per_pattern.append({
            "pattern": patterns[k].detach().cpu().tolist(),
            "mse_by_repeat": mse_by_repeat,
            "mse_agg": mse_agg,
            "best_repeat": best_r,
            "disc_by_repeat": disc_by_repeat,
            "disc_agg": disc_agg,
            "overlap_top": overlap_top,
        })

    return per_pattern


# ============================================================
# 11) 单样本：loss vs steps（对每个 pattern 都算一次）
# ============================================================
@torch.no_grad()
def eval_one_sample_loss_vs_steps(
    nnet,
    indices_one,
    batch_one_raw,
    convert_input_fn,
    sample_steps_list=(5, 10, 20, 50, 100),
    repeats=16,
    repeat_agg="mean",
    mask_mode="all_single",
    strict_context_mask=True,
    seed_base=528
):
    feature_keys = ["conditions", "drugs", "procedures"]

    indices_one, batch_conv = convert_input_fn((indices_one, batch_one_raw), mask_static=True)
    z0_true, ctx_full, *_ = nnet.encode_everything_test(indices_one, **batch_conv)
    device = z0_true.device

    patterns = seed528_enumerate_viewmasks_v2(mask_mode, device=device)  # [K,3]
    B = z0_true.size(0)
    assert B == 1

    out = {}
    for steps in sample_steps_list:
        per_pattern = []
        for k in range(patterns.size(0)):
            view_mask = patterns[k].unsqueeze(0).expand(B, -1).contiguous()

            if strict_context_mask:
                batch2 = {}
                for kk, vv in batch_conv.items():
                    if torch.is_tensor(vv):
                        batch2[kk] = vv.clone()
                    elif isinstance(vv, list):
                        batch2[kk] = list(vv)
                    else:
                        batch2[kk] = vv
                _apply_strict_view_mask(batch2, view_mask, feature_keys=feature_keys)
                _, ctx, *_ = nnet.encode_everything_test(indices_one, **batch2)
            else:
                ctx = ctx_full

            mses_rb, _ = dpm_inpaint_repeats_get_mse_and_zhat(
                nnet=nnet,
                z0_true=z0_true,
                context=ctx,
                view_mask_b3=view_mask,
                sample_steps=int(steps),
                repeats=int(repeats),
                seed=seed_base + 99991 * int(steps) + 10007 * k
            )
            mse_by_repeat = mses_rb[:, 0].numpy().astype(np.float32)
            mse_agg = float(np.mean(mse_by_repeat)) if repeat_agg == "mean" else float(np.min(mse_by_repeat))

            per_pattern.append({
                "pattern": patterns[k].detach().cpu().tolist(),
                "mse_agg": mse_agg
            })
        out[int(steps)] = per_pattern
    return out


# ============================================================
# 12) TRAIN-ONLY 主函数：统计 -> 选 5 条 -> 全部补全评估
# ============================================================
def run_extreme5_train_only_impute_eval(
    nnet,
    diff_config,
    config_health,
    hp_dataset,
    tokenizers,
    convert_input_fn,
    keys_map=None,
    min_len=1,

    sample_steps=20,
    repeats=16,
    repeat_agg="mean",                    # mean: 看补全能力；min: 更敏感/偏“最容易补出来的那次”
    mask_modes=("all_single",),           # 建议先 ("all_single",) 再加 ("all_views",)
    run_loss_vs_steps=True,
    steps_list=(5, 10, 20, 50, 100),
    decode_threshold=0.5,
    strict_context_mask=True,
    seed_base=528,
    topk_common=30,
    rare_df_le=2,
    overlap_topk=15,
):
    device = torch.device("cuda:" + str(diff_config["device"]) if torch.cuda.is_available() else "cpu")
    nnet.eval().to(device)

    if keys_map is None:
        keys_map = {
            "conditions": ["conditions_comp", "conditions_comps", "conditions"],
            "procedures": ["procedures_comp", "procedures_comps", "procedures"],
            "drugs":      ["drugs_comp", "drugs_comps", "drugs"],
        }

    # 1) 只拿训练集（仍沿用你 get_dataset(...) 的写法）
    dataset = get_dataset(config_health["DATASET"], aug=True, samples=hp_dataset, config=diff_config)
    train_ds = dataset.get_split(split="train", labeled=True)

    # 2) 只保留三视图都有的训练样本
    train_c = seed528_complete3v_subset_v1(train_ds, min_len=min_len, keys_map=keys_map)
    print("\n===== [0] TRAIN-ONLY size =====")
    print("train total =", len(train_ds), "train complete3v =", len(train_c))

    # 3) code 统计（只在 train complete3v）
    stats = build_code_stats_complete3v(train_c, keys_map=keys_map)
    print_code_stats(stats, topk=topk_common, rare_df_le=rare_df_le)
    idf = stats["idf"]

    # 4) 选 5 条极端训练样本
    rows = score_samples(train_c, idf=idf, keys_map=keys_map)
    picked = pick_5_extremes(rows)
    print_picked(picked)

    results = {"stats": stats, "picked": picked, "per_sample": []}

    # 5) 对 picked-5 做补全评估
    for si, info in enumerate(picked):
        subset_j = info["subset_j"]
        indices_one, batch_one = get_one_batch_from_subset(train_c, subset_j, collate_fn_dict)

        print(f"\n==================== TRAIN Sample#{si} subset_j={subset_j} ====================")
        print(f"score: rarity_sum={info['rarity_sum']:.3f} avg_idf={info['avg_idf']:.3f} "
              f"total_len={info['total_len']} lens={info['lens']} uniq={info['uniq']}")

        sample_out = {
            "subset_j": subset_j,
            "score": info,
            "mask_mode_results": {},
            "loss_vs_steps": {}
        }

        for mm in mask_modes:
            per_pattern = eval_one_sample_impute(
                nnet=nnet,
                indices_one=indices_one,
                batch_one_raw=batch_one,
                convert_input_fn=convert_input_fn,
                tokenizers=tokenizers,
                idf=idf,
                sample_steps=sample_steps,
                repeats=repeats,
                repeat_agg=repeat_agg,
                mask_mode=mm,
                strict_context_mask=strict_context_mask,
                decode_threshold=decode_threshold,
                seed_base=seed_base + 1000 * si,
                print_overlap_topk=overlap_topk,
            )
            sample_out["mask_mode_results"][mm] = per_pattern

            print(f"\n--- TRAIN | mask_mode={mm} | steps={sample_steps} repeats={repeats} "
                  f"repeat_agg={repeat_agg} strict_context={strict_context_mask} ---")
            for pi, pr in enumerate(per_pattern):
                print(f"[pattern#{pi}] mask={pr['pattern']}  maskedMSE({repeat_agg})={pr['mse_agg']:.6f}  best_r={pr['best_repeat']}")
                for fk, m in pr["disc_agg"].items():
                    print(f"   view={fk}: wrec={m['wrec']:.4f} wprec={m['wprec']:.4f} wj={m['wj']:.4f}")
                    if fk in pr["overlap_top"]:
                        print(f"      overlap(top-idf): {pr['overlap_top'][fk]}")

        if run_loss_vs_steps:
            for mm in mask_modes:
                lvs = eval_one_sample_loss_vs_steps(
                    nnet=nnet,
                    indices_one=indices_one,
                    batch_one_raw=batch_one,
                    convert_input_fn=convert_input_fn,
                    sample_steps_list=steps_list,
                    repeats=repeats,
                    repeat_agg=repeat_agg,
                    mask_mode=mm,
                    strict_context_mask=strict_context_mask,
                    seed_base=seed_base + 1000 * si + 55555,
                )
                sample_out["loss_vs_steps"][mm] = lvs
                print(f"\n   TRAIN loss-vs-steps | mask_mode={mm} | repeat_agg={repeat_agg}")
                for st in steps_list:
                    arr = [(p["pattern"], p["mse_agg"]) for p in lvs[int(st)]]
                    print(f"    steps={st}: {arr}")

        results["per_sample"].append(sample_out)

    print("\n===== DONE: TRAIN-ONLY extreme5 complete3v impute eval =====")
    return results

# ============================================================
# 13) TRAIN memorization / duplication audit module
#     (generated hard tokens vs TRAIN complete3v)
# ============================================================
import hashlib
from collections import defaultdict

AUDIT_FEATURE_KEYS = ["conditions", "drugs", "procedures"]

def _norm_token(t):
    # 保证训练集 token 与 decode token 格式一致（都转 str）
    return str(t)

def _viewsets_from_train_sample(ds_subset, j, keys_map):
    s = _unwrap_sample(ds_subset[j])
    vs = {}
    for fk in AUDIT_FEATURE_KEYS:
        toks = get_tokens_from_sample(s, fk, keys_map)  # list
        vs[fk] = set(_norm_token(x) for x in toks)
    return vs

def _viewsets_from_pred_sets(pred_sets):
    """
    pred_sets: dict {fk: set(tokens)}  or {fk: list_of_sets}
    这里统一成 {fk: set(str)}
    """
    vs = {}
    for fk in AUDIT_FEATURE_KEYS:
        v = pred_sets.get(fk, set())
        if isinstance(v, list):
            # 如果是 List[set] 取第0个
            v = v[0] if len(v) > 0 else set()
        vs[fk] = set(_norm_token(x) for x in v)
    return vs

def _union_all(vs):
    u = set()
    for fk in AUDIT_FEATURE_KEYS:
        u |= set(vs.get(fk, set()))
    return u

def _fingerprint_viewsets(vs):
    """
    用 set（无序）做稳定 fingerprint
    注意：这忽略 token 的重复次数，只看集合一致性（与你的 wj/wrec 一致）
    """
    rep = []
    for fk in AUDIT_FEATURE_KEYS:
        rep.append(tuple(sorted(vs.get(fk, set()))))
    b = repr(tuple(rep)).encode("utf-8")
    return hashlib.md5(b).hexdigest()

def _overlap_top_idf(inter_set, idf, topk=15):
    xs = list(inter_set)
    xs.sort(key=lambda x: idf.get(x, 1.0), reverse=True)
    return xs[:topk]

def build_train_mem_index(train_subset, keys_map, idf=None, use_inverted_index=True):
    """
    返回一个 dict index:
      - train_viewsets_list: [N] 每条训练样本的 {fk:set(str)}
      - fp_to_trainjs: fingerprint -> [train_j,...]（精确复刻）
      - inv_index: token -> set(train_j)（候选加速，可选）
    """
    if idf is None:
        idf = {}

    train_viewsets_list = []
    fp_to_trainjs = defaultdict(list)
    inv_index = defaultdict(set) if use_inverted_index else None

    for j in range(len(train_subset)):
        vs = _viewsets_from_train_sample(train_subset, j, keys_map)
        train_viewsets_list.append(vs)

        fp = _fingerprint_viewsets(vs)
        fp_to_trainjs[fp].append(j)

        if use_inverted_index:
            u = _union_all(vs)
            for t in u:
                inv_index[t].add(j)

    return {
        "train_viewsets_list": train_viewsets_list,
        "fp_to_trainjs": fp_to_trainjs,
        "inv_index": inv_index,
        "idf": idf,
        "N_train": len(train_subset),
    }

def query_train_nn_one(gen_vs, mem_index, topk=5, rare_k=40, cap_candidates=3000):
    """
    对单条生成样本 gen_vs 查训练集最近邻（按 union 的 weighted Jaccard wj 排序）
    返回 [(wj,wrec,wprec,train_j, overlap_top_idf_tokens), ...] topk
    """
    idf = mem_index["idf"]
    inv = mem_index["inv_index"]
    train_list = mem_index["train_viewsets_list"]

    q_union = list(_union_all(gen_vs))
    if len(q_union) == 0:
        return []

    # 取 IDF 最高的 rare_k 个 token 去倒排找候选
    q_union.sort(key=lambda t: idf.get(t, 1.0), reverse=True)
    probe = q_union[:min(rare_k, len(q_union))]

    if inv is None:
        cand = list(range(len(train_list)))
    else:
        cand_set = set()
        for t in probe:
            cand_set |= inv.get(t, set())
        cand = list(cand_set)

    if len(cand) > cap_candidates:
        cand = cand[:cap_candidates]

    qU = set(q_union)
    scored = []
    for j in cand:
        tvs = train_list[j]
        tU = _union_all(tvs)

        m = weighted_metrics(qU, tU, idf)  # 复用你上面定义的 weighted_metrics
        inter = qU & tU
        ov = _overlap_top_idf(inter, idf, topk=15)
        scored.append((m["wj"], m["wrec"], m["wprec"], j, ov))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topk]

def audit_one_generated_sample(
    pred_sets_all_views,     # {fk: set(tokens)}  (建议三视图都传)
    mem_index,
    wj_thr=0.95,
    topk=5
):
    """
    返回 dict:
      - exact_match: True/False
      - exact_train_js: [train_j,...] (若 exact)
      - nn_topk: list of tuples (wj,wrec,wprec,train_j,overlap_topidf)
      - is_near_dup: True/False (top1 wj>=wj_thr)
    """
    gen_vs = _viewsets_from_pred_sets(pred_sets_all_views)
    fp = _fingerprint_viewsets(gen_vs)

    fp_to = mem_index["fp_to_trainjs"]
    exact_train_js = fp_to.get(fp, [])

    nn = query_train_nn_one(gen_vs, mem_index, topk=topk)
    is_near = (len(nn) > 0 and nn[0][0] >= wj_thr)

    return {
        "exact_match": len(exact_train_js) > 0,
        "exact_train_js": exact_train_js,
        "nn_topk": nn,
        "is_near_dup": is_near,
        "top1": nn[0] if len(nn) > 0 else None,
    }

def print_audit_result(tag, audit_res, wj_thr=0.95):
    """
    打印友好日志
    """
    if audit_res["exact_match"]:
        print(f"[AUDIT-{tag}] EXACT match! train_js={audit_res['exact_train_js'][:10]}")
    else:
        print(f"[AUDIT-{tag}] no exact match.")

    if audit_res["top1"] is None:
        print(f"[AUDIT-{tag}] no NN candidates.")
        return

    wj, wrec, wprec, tj, ov = audit_res["top1"]
    print(f"[AUDIT-{tag}] top1 NN: train_j={tj}  wj={wj:.4f} wrec={wrec:.4f} wprec={wprec:.4f}  "
          f"{'NEAR-DUP' if wj>=wj_thr else ''}")
    print(f"[AUDIT-{tag}] top1 overlap(top-idf): {ov}")


import math
from collections import Counter
from torch.utils.data import Subset

def _unwrap_sample(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, tuple):
        for t in reversed(x):
            if isinstance(t, dict):
                return t
    return x

def _pick_list(s, cands):
    if not isinstance(s, dict):
        return []
    for k in cands:
        v = s.get(k, None)
        if v is not None:
            return v
    return []

def _flatten_tokens(v):
    if v is None:
        return []
    if not isinstance(v, list):
        return [v]
    out = []
    for x in v:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out

def get_conditions_tokens_from_sample(sample_dict, keys_map):
    v = _pick_list(sample_dict, keys_map["conditions"])
    return _flatten_tokens(v)  # list (可重复)

def build_conditions_df_idf(train_subset, keys_map):
    """
    只统计 conditions（诊断）视图的 DF/IDF
    DF: 在多少个样本中出现
    IDF: log((N+1)/(df+1))+1
    """
    df_cond = Counter()
    N = len(train_subset)
    for j in range(N):
        s = _unwrap_sample(train_subset[j])
        toks = get_conditions_tokens_from_sample(s, keys_map)
        df_cond.update(set(toks))
    idf_cond = {t: math.log((N + 1.0) / (c + 1.0)) + 1.0 for t, c in df_cond.items()}
    return N, df_cond, idf_cond

def score_samples_by_rare_conditions(train_subset, df_cond, idf_cond, keys_map, rare_df_le=2):
    """
    给每条样本打“罕见诊断程度”分：
      - rare_cnt: 罕见诊断 code 的数量（unique）
      - rare_idf_sum: 罕见诊断 code 的 IDF 加权和
      - avg_rare_idf: 罕见诊断 code 平均 IDF
      - cond_uniq: 诊断 unique 数量
      - cond_len: 诊断长度（含重复）
      - rare_codes_sorted: 该样本罕见诊断 code（按 IDF 降序）
    """
    rows = []
    for j in range(len(train_subset)):
        s = _unwrap_sample(train_subset[j])
        toks = get_conditions_tokens_from_sample(s, keys_map)
        uniq = set(toks)

        rare = [t for t in uniq if df_cond.get(t, 0) <= rare_df_le]
        rare.sort(key=lambda x: idf_cond.get(x, 1.0), reverse=True)

        rare_cnt = len(rare)
        rare_idf_sum = float(sum(idf_cond.get(t, 1.0) for t in rare))
        avg_rare_idf = rare_idf_sum / max(rare_cnt, 1)

        rows.append({
            "subset_j": j,
            "rare_cnt": rare_cnt,
            "rare_idf_sum": rare_idf_sum,
            "avg_rare_idf": avg_rare_idf,
            "cond_uniq": len(uniq),
            "cond_len": len(toks),
            "rare_codes_sorted": rare,
        })
    return rows

def pick_5_rare_condition_samples(rows, topk=5, require_at_least_one_rare=True):
    """
    选 5 条“诊断罕见”样本：
    默认按 (rare_cnt desc, rare_idf_sum desc, cond_uniq desc) 排序
    """
    cand = rows
    if require_at_least_one_rare:
        cand = [r for r in rows if r["rare_cnt"] > 0]
    cand = sorted(cand, key=lambda r: (r["rare_cnt"], r["rare_idf_sum"], r["cond_uniq"]), reverse=True)
    return cand[:topk]

def print_rare5(picked, rare_show_top=20):
    print("\n===== Picked 5 samples with rare CONDITIONS codes =====")
    for i, r in enumerate(picked):
        rare_top = r["rare_codes_sorted"][:rare_show_top]
        print(f"[{i}] subset_j={r['subset_j']}  rare_cnt={r['rare_cnt']}  "
              f"rare_idf_sum={r['rare_idf_sum']:.3f}  avg_rare_idf={r['avg_rare_idf']:.3f}  "
              f"cond_uniq={r['cond_uniq']}  cond_len={r['cond_len']}")
        print(f"     rare_conditions(top{rare_show_top}) = {rare_top}")


def pick_rare5_conditions_from_train_only(diff_config, config_health, hp_dataset, keys_map=None, min_len=1, rare_df_le=2):
    if keys_map is None:
        keys_map = {
            "conditions": ["conditions_comp", "conditions_comps", "conditions"],
            "procedures": ["procedures_comp", "procedures_comps", "procedures"],
            "drugs":      ["drugs_comp", "drugs_comps", "drugs"],
        }

    dataset = get_dataset(config_health["DATASET"], aug=True, samples=hp_dataset, config=diff_config)
    train_ds = dataset.get_split(split="train", labeled=True)

    # 只保留三视图都有
    train_c = seed528_complete3v_subset_v1(train_ds, min_len=min_len, keys_map=keys_map)
    print("\n===== TRAIN size =====")
    print("train total =", len(train_ds), "train complete3v =", len(train_c))

    # 只统计 conditions DF/IDF
    N, df_cond, idf_cond = build_conditions_df_idf(train_c, keys_map=keys_map)
    print("\n===== CONDITIONS DF stats =====")
    print("N_complete3v =", N)
    # 打印一些全局最罕见（df<=rare_df_le）的 examples
    rare_global = [(t, c) for t, c in df_cond.items() if c <= rare_df_le]
    rare_global.sort(key=lambda x: x[1])
    print(f"global rare conditions codes: DF <= {rare_df_le}  head50 =", rare_global[:50])

    # 给样本打分并挑 5 条
    rows = score_samples_by_rare_conditions(train_c, df_cond, idf_cond, keys_map=keys_map, rare_df_le=rare_df_le)
    picked = pick_5_rare_condition_samples(rows, topk=5, require_at_least_one_rare=True)
    print_rare5(picked, rare_show_top=20)

    # 返回：subset + 5条的 subset_j（注意是 train_c 的索引，不是原始 train_ds 索引）
    picked_js = [r["subset_j"] for r in picked]
    return train_c, picked, picked_js

########################
# -*- coding: utf-8 -*-
# Save as: src/generation_MIA_rare5.py

import os
import math
import copy
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# === your project imports (keep same as your existing generation_MIA.py) ===
from diffuser import diff_utils as utils
from diffuser.diff_dataset import get_dataset, collate_fn_dict
from diffuser.dpm import NoiseScheduleVP, DPM_Solver


# ============================================================
# 0) helpers
# ============================================================
def deepcopy_batch(batch):
    # IMPORTANT: avoid "Tensor + list" error by preventing double-encoding on the same batch object
    return copy.deepcopy(batch)

def _unwrap_sample(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, tuple):
        for t in reversed(x):
            if isinstance(t, dict):
                return t
    return x

def _pick_list(s, cands):
    if not isinstance(s, dict):
        return []
    for k in cands:
        v = s.get(k, None)
        if v is not None:
            return v
    return []

def _flatten_tokens(v):
    if v is None:
        return []
    if not isinstance(v, list):
        return [v]
    out = []
    for x in v:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


# ============================================================
# 1) your convert_input_fn (same logic as you pasted)
# ============================================================
def build_convert_input_fn_health_v1(tokenizers, device, config_health):
    feature_keys = ['conditions', 'drugs', 'procedures']

    def convert_input_fn(batch_data, mask_static=True):
        indices, batch = batch_data

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

        # ---- safe fallback for missing 'mask' field ----
        if mask_static:
            if 'mask' in batch:
                batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
            else:
                B = batch['conditions'].shape[0]
                batch['mask'] = torch.zeros(B, 3, dtype=torch.bool, device=device)
        else:
            m = 3
            batch['mask'] = utils.generate_mask(batch['conditions'].shape[0], m=m).to(device=device)

        aligned_data = [
            batch[fk + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1) for fk in feature_keys
        ]
        batch['mask_hist'] = torch.cat(aligned_data, dim=-1)

        return indices, batch

    return convert_input_fn


# ============================================================
# 2) keep only complete3v
# ============================================================
def seed528_complete3v_subset_v1(ds_split, min_len=1, keys_map=None):
    if keys_map is None:
        keys_map = {
            "conditions": ["conditions_comp", "conditions_comps", "conditions"],
            "procedures": ["procedures_comp", "procedures_comps", "procedures"],
            "drugs":      ["drugs_comp", "drugs_comps", "drugs"],
        }

    def _unwrap(x):
        return _unwrap_sample(x)

    def _pick_list_local(s, cands):
        return _pick_list(s, cands)

    keep = []
    for i in range(len(ds_split)):
        s = _unwrap(ds_split[i])

        c = _pick_list_local(s, keys_map["conditions"])
        p = _pick_list_local(s, keys_map["procedures"])
        d = _pick_list_local(s, keys_map["drugs"])

        if (len(c) >= min_len) and (len(p) >= min_len) and (len(d) >= min_len):
            keep.append(i)

    return Subset(ds_split, keep)


# ============================================================
# 3) mask patterns
# ============================================================
def seed528_enumerate_viewmasks_v2(mode: str, device):
    """
    patterns: [K,3] bool
      - all_single: 3 single masks
      - all_nonempty: 7 non-empty combos
      - all_views: 1 full mask
    """
    if mode == "all_single":
        return torch.eye(3, dtype=torch.bool, device=device)
    elif mode == "all_nonempty":
        masks = []
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    if a == 0 and b == 0 and c == 0:
                        continue
                    masks.append([a, b, c])
        return torch.tensor(masks, dtype=torch.bool, device=device)
    elif mode == "all_views":
        return torch.tensor([[1, 1, 1]], dtype=torch.bool, device=device)
    else:
        raise ValueError(f"Unknown mask mode: {mode}")


# ============================================================
# 4) get one sample batch (B=1)
# ============================================================
def get_one_batch_from_subset(ds_subset, subset_j, collate_fn):
    one_ds = Subset(ds_subset, [subset_j])
    loader = DataLoader(one_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    indices, batch = next(iter(loader))
    return indices, batch


# ============================================================
# 5) strict_context: clear masked views from inputs (avoid GT leakage)
# ============================================================
def _recompute_view_masks(batch: Dict[str, Any], feature_keys: List[str]):
    for fk in feature_keys:
        if fk in batch and torch.is_tensor(batch[fk]):
            batch[fk + "_mask"] = (batch[fk] != 0)
        if fk + "_hist" in batch and torch.is_tensor(batch[fk + "_hist"]):
            batch[fk + "_hist_mask"] = (batch[fk + "_hist"] != 0)
        if fk + "_comp" in batch and torch.is_tensor(batch[fk + "_comp"]):
            batch[fk + "_comp_mask"] = (batch[fk + "_comp"] != 0)

    aligned = []
    for fk in feature_keys:
        if fk + "_hist_mask" in batch and torch.is_tensor(batch[fk + "_hist_mask"]):
            aligned.append(batch[fk + "_hist_mask"].sum(dim=-1).unsqueeze(-1))
    if len(aligned) == 3:
        batch["mask_hist"] = torch.cat(aligned, dim=-1)

def _apply_strict_view_mask(batch: Dict[str, Any], view_mask_b3: torch.Tensor,
                            feature_keys=("conditions","drugs","procedures")):
    """
    Clear masked views from batch:
    - fk / fk_hist / fk_comp -> zeros
    - fk_comps_origin / fk_comps (python list) -> empty
    """
    B = view_mask_b3.shape[0]
    device = view_mask_b3.device

    for j, fk in enumerate(feature_keys):
        m = view_mask_b3[:, j]  # [B] bool
        if not bool(m.any()):
            continue

        if fk in batch and torch.is_tensor(batch[fk]):
            batch[fk][m] = 0
        hk = fk + "_hist"
        if hk in batch and torch.is_tensor(batch[hk]):
            batch[hk][m] = 0
        ck = fk + "_comp"
        if ck in batch and torch.is_tensor(batch[ck]):
            batch[ck][m] = 0

        ok = fk + "_comps_origin"
        if ok in batch and isinstance(batch[ok], list):
            for bi in torch.where(m)[0].tolist():
                batch[ok][bi] = []

        ok2 = fk + "_comps"
        if ok2 in batch and isinstance(batch[ok2], list):
            for bi in torch.where(m)[0].tolist():
                batch[ok2][bi] = []

    batch["mask"] = view_mask_b3.to(device)
    _recompute_view_masks(batch, list(feature_keys))


# ============================================================
# 6) DPM inpaint repeats: only inject noise on masked views, then reverse sample
# ============================================================
@torch.no_grad()
def dpm_inpaint_repeats_get_mse_and_zhat(
    nnet,
    z0_true: torch.Tensor,        # [B,3,D]
    context: torch.Tensor,
    view_mask_b3: torch.Tensor,   # [B,3] bool
    sample_steps=20,
    repeats=16,
    seed=528
):
    device = z0_true.device
    B, M, D = z0_true.shape

    mask_b3 = view_mask_b3.to(device)
    mask_b3d = mask_b3.unsqueeze(-1).expand(B, M, D)
    denom = mask_b3d.float().sum(dim=(1, 2)).clamp_min(1.0)

    betas = utils.stable_diffusion_beta_schedule()
    betas_t = torch.tensor(betas, device=device).float()
    N = betas_t.numel()
    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=betas_t)

    def model_fn(x, t_cont):
        t = t_cont * N
        return nnet(x, t, context=context)

    dpm = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)

    mses = []
    zhats = []

    for r in range(repeats):
        cuda_devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(seed + 10007 * r)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + 10007 * r)
            noise = torch.randn_like(z0_true)

        z_init = torch.where(mask_b3d, noise, z0_true)
        z_hat = dpm.sample(z_init, steps=sample_steps, eps=1.0 / N, T=1.0)

        diff2 = (z_hat - z0_true) ** 2
        mse_masked = (diff2 * mask_b3d.float()).sum(dim=(1, 2)) / denom

        mses.append(mse_masked.detach().cpu())
        zhats.append(z_hat.detach().cpu())

    mses = torch.stack(mses, dim=0)     # [R,B]
    zhats = torch.stack(zhats, dim=0)   # [R,B,3,D]
    return mses, zhats


# ============================================================
# 7) decode latent -> token sets; gt sets from batch
# ============================================================
def _wsum(S, idf):
    return float(sum(idf.get(t, 1.0) for t in S))

def weighted_metrics(pred_set, gt_set, idf):
    inter = pred_set & gt_set
    u = pred_set | gt_set
    wi = _wsum(inter, idf)
    wg = _wsum(gt_set, idf)
    wp = _wsum(pred_set, idf)
    wu = _wsum(u, idf)
    return {
        "wrec": wi / (wg + 1e-12),
        "wprec": wi / (wp + 1e-12),
        "wj": wi / (wu + 1e-12),
    }

@torch.no_grad()
def decode_latent_to_token_sets(z_b3d, nnet, tokenizers, threshold=0.5):
    """
    z_b3d: [B,3,D] -> dict {fk: List[set(tokens)]}
    """
    feature_keys = ["conditions", "drugs", "procedures"]
    out = {}

    for idx, fk in enumerate(feature_keys):
        logits = nnet.get_logits(z_b3d[:, idx, :], fk)  # must exist in your nnet
        prob = torch.sigmoid(logits).detach().cpu().numpy()

        max_idx = np.argmax(prob, axis=1)
        bin_ = (prob >= threshold).astype(np.int64)

        indices = []
        for b in range(bin_.shape[0]):
            on = (np.where(bin_[b] == 1)[0] + 2).tolist()  # +2 for tokenizer offset
            if len(on) < 1:
                on = [int(max_idx[b] + 2)]
            indices.append(on)

        toks_2d = tokenizers[fk].batch_decode_2d(indices)
        out[fk] = [set(row) for row in toks_2d]
    return out

def get_gt_token_sets_from_batch(batch_conv, tokenizers):
    feature_keys = ["conditions", "drugs", "procedures"]
    out = {}

    for fk in feature_keys:
        ok = fk + "_comps_origin"
        if ok in batch_conv and isinstance(batch_conv[ok], list) and len(batch_conv[ok]) > 0:
            out[fk] = set(_flatten_tokens(batch_conv[ok][0]))
            continue

        ck = fk + "_comp"
        if ck in batch_conv and torch.is_tensor(batch_conv[ck]):
            ids = [int(x) for x in batch_conv[ck][0].detach().cpu().tolist() if int(x) != 0]
            if len(ids) == 0:
                out[fk] = set()
            else:
                toks = tokenizers[fk].batch_decode_2d([ids])[0]
                out[fk] = set(toks)
            continue

        out[fk] = set()

    return out

def overlap_top_idf_tokens(pred_set, gt_set, idf, topk=15):
    inter = list(pred_set & gt_set)
    inter.sort(key=lambda x: idf.get(x, 1.0), reverse=True)
    return inter[:topk]


# ============================================================
# 8) eval one sample under mask_mode (modified: deepcopy + return pred/gt sets)
# ============================================================
@torch.no_grad()
def eval_one_sample_impute(
    nnet,
    indices_one,
    batch_one_raw,
    convert_input_fn,
    tokenizers,
    idf,
    sample_steps=20,
    repeats=16,
    repeat_agg="mean",
    mask_mode="all_views",
    strict_context_mask=True,
    decode_threshold=0.5,
    seed_base=528,
    print_overlap_topk=15,
):
    # IMPORTANT: prevent double-encoding pollution
    batch_one_raw = deepcopy_batch(batch_one_raw)

    feature_keys = ["conditions", "drugs", "procedures"]

    indices_one, batch_conv = convert_input_fn((indices_one, batch_one_raw), mask_static=True)
    z0_true, ctx_full, *_ = nnet.encode_everything_test(indices_one, **batch_conv)

    B = z0_true.size(0)
    assert B == 1, "建议一次只跑 1 条样本(B=1)"
    device = z0_true.device

    patterns = seed528_enumerate_viewmasks_v2(mask_mode, device=device)
    gt_sets = get_gt_token_sets_from_batch(batch_conv, tokenizers)

    per_pattern = []
    for k in range(patterns.size(0)):
        view_mask = patterns[k].unsqueeze(0).expand(B, -1).contiguous()

        if strict_context_mask:
            batch2 = {}
            for kk, vv in batch_conv.items():
                if torch.is_tensor(vv):
                    batch2[kk] = vv.clone()
                elif isinstance(vv, list):
                    batch2[kk] = list(vv)
                else:
                    batch2[kk] = vv
            _apply_strict_view_mask(batch2, view_mask, feature_keys=feature_keys)
            _, ctx, *_ = nnet.encode_everything_test(indices_one, **batch2)
        else:
            ctx = ctx_full

        mses_rb, zhats_rb3d = dpm_inpaint_repeats_get_mse_and_zhat(
            nnet=nnet,
            z0_true=z0_true,
            context=ctx,
            view_mask_b3=view_mask,
            sample_steps=sample_steps,
            repeats=repeats,
            seed=seed_base + 10007 * k
        )
        mse_by_repeat = mses_rb[:, 0].numpy().astype(np.float32)

        disc_by_repeat = {fk: [] for fk in feature_keys}
        overlap_top = {}

        for r in range(repeats):
            z_hat = zhats_rb3d[r].to(device=device, dtype=z0_true.dtype)
            pred_sets_b = decode_latent_to_token_sets(z_hat, nnet, tokenizers, threshold=decode_threshold)

            for j, fk in enumerate(feature_keys):
                if bool(view_mask[0, j].item()) is False:
                    continue
                pred_set = pred_sets_b[fk][0]
                gt_set = gt_sets[fk]
                disc_by_repeat[fk].append(weighted_metrics(pred_set, gt_set, idf))

        if repeat_agg == "mean":
            # mse_agg = float(np.mean(mse_by_repeat))
            mse_by_repeat = list(mse_by_repeat)
            mse_agg = float(sum(mse_by_repeat) / max(len(mse_by_repeat), 1))
            disc_agg = {}
            for j, fk in enumerate(feature_keys):
                if not bool(view_mask[0, j].item()):
                    continue
                ws = disc_by_repeat[fk]
                disc_agg[fk] = {
                    "wrec": float(np.mean([x["wrec"] for x in ws])) if ws else 0.0,
                    "wprec": float(np.mean([x["wprec"] for x in ws])) if ws else 0.0,
                    "wj": float(np.mean([x["wj"] for x in ws])) if ws else 0.0,
                }
            best_r = int(np.argmin(mse_by_repeat))
        elif repeat_agg == "min":
            best_r = int(np.argmin(mse_by_repeat))
            mse_agg = float(mse_by_repeat[best_r])
            disc_agg = {}
            for j, fk in enumerate(feature_keys):
                if not bool(view_mask[0, j].item()):
                    continue
                if len(disc_by_repeat[fk]) == 0:
                    disc_agg[fk] = {"wrec": 0.0, "wprec": 0.0, "wj": 0.0}
                else:
                    disc_agg[fk] = disc_by_repeat[fk][best_r]
        else:
            raise ValueError("repeat_agg must be 'mean' or 'min'")

        z_best = zhats_rb3d[best_r].to(device=device, dtype=z0_true.dtype)
        pred_best = decode_latent_to_token_sets(z_best, nnet, tokenizers, threshold=decode_threshold)

        for j, fk in enumerate(feature_keys):
            if not bool(view_mask[0, j].item()):
                continue
            overlap_top[fk] = overlap_top_idf_tokens(pred_best[fk][0], gt_sets[fk], idf, topk=print_overlap_topk)

        # Return pred/gt sets for downstream plain/weighted Jaccard
        pred_best_sets = {fk: list(pred_best[fk][0]) for fk in feature_keys}
        gt_sets_list   = {fk: list(gt_sets[fk]) for fk in feature_keys}

        per_pattern.append({
            "pattern": patterns[k].detach().cpu().tolist(),
            "mse_by_repeat": mse_by_repeat,
            "mse_agg": mse_agg,
            "best_repeat": best_r,
            "disc_agg": disc_agg,
            "overlap_top": overlap_top,
            "pred_best_sets": pred_best_sets,
            "gt_sets": gt_sets_list,
        })

    return per_pattern


# ============================================================
# 9) Rare conditions pick (your code + add rare_conditions alias)
# ============================================================
def get_conditions_tokens_from_sample(sample_dict, keys_map):
    v = _pick_list(sample_dict, keys_map["conditions"])
    return _flatten_tokens(v)

def build_conditions_df_idf(train_subset, keys_map):
    df_cond = Counter()
    N = len(train_subset)
    for j in range(N):
        s = _unwrap_sample(train_subset[j])
        toks = get_conditions_tokens_from_sample(s, keys_map)
        df_cond.update(set(toks))
    idf_cond = {t: math.log((N + 1.0) / (c + 1.0)) + 1.0 for t, c in df_cond.items()}
    return N, df_cond, idf_cond

def score_samples_by_rare_conditions(train_subset, df_cond, idf_cond, keys_map, rare_df_le=2):
    rows = []
    for j in range(len(train_subset)):
        s = _unwrap_sample(train_subset[j])
        toks = get_conditions_tokens_from_sample(s, keys_map)
        uniq = set(toks)

        rare = [t for t in uniq if df_cond.get(t, 0) <= rare_df_le]
        rare.sort(key=lambda x: idf_cond.get(x, 1.0), reverse=True)

        rare_cnt = len(rare)
        rare_idf_sum = float(sum(idf_cond.get(t, 1.0) for t in rare))
        avg_rare_idf = rare_idf_sum / max(rare_cnt, 1)

        rows.append({
            "subset_j": j,
            "rare_cnt": rare_cnt,
            "rare_idf_sum": rare_idf_sum,
            "avg_rare_idf": avg_rare_idf,
            "cond_uniq": len(uniq),
            "cond_len": len(toks),
            "rare_codes_sorted": rare,   # original
            "rare_conditions": rare,     # alias to avoid KeyError
        })
    return rows

def pick_5_rare_condition_samples(rows, topk=5, require_at_least_one_rare=True):
    cand = rows
    if require_at_least_one_rare:
        cand = [r for r in rows if r["rare_cnt"] > 0]
    cand = sorted(cand, key=lambda r: (r["rare_cnt"], r["rare_idf_sum"], r["cond_uniq"]), reverse=True)
    return cand[:topk]

def print_rare5(picked, rare_show_top=20):
    print("\n===== Picked 5 samples with rare CONDITIONS codes =====")
    for i, r in enumerate(picked):
        rare_top = r.get("rare_conditions", r.get("rare_codes_sorted", []))[:rare_show_top]
        print(f"[{i}] subset_j={r['subset_j']} rare_cnt={r['rare_cnt']} "
              f"rare_idf_sum={r['rare_idf_sum']:.3f} avg_rare_idf={r['avg_rare_idf']:.3f} "
              f"cond_len={r['cond_len']}")
        print(f"    rare_conditions(top{rare_show_top})={rare_top}")

def pick_rare5_conditions_from_train_only(
    diff_config, config_health, hp_dataset,
    keys_map=None, min_len=1, rare_df_le=2
):
    if keys_map is None:
        keys_map = {
            "conditions": ["conditions_comp", "conditions_comps", "conditions"],
            "procedures": ["procedures_comp", "procedures_comps", "procedures"],
            "drugs":      ["drugs_comp", "drugs_comps", "drugs"],
        }

    dataset = get_dataset(config_health["DATASET"], aug=True, samples=hp_dataset, config=diff_config)
    train_ds = dataset.get_split(split="train", labeled=True)

    train_c = seed528_complete3v_subset_v1(train_ds, min_len=min_len, keys_map=keys_map)
    print("\n===== TRAIN size =====")
    print("train total =", len(train_ds), "train complete3v =", len(train_c))

    N, df_cond, idf_cond = build_conditions_df_idf(train_c, keys_map=keys_map)
    print("\n===== CONDITIONS DF stats =====")
    print("N_complete3v =", N)

    rare_global = [(t, c) for t, c in df_cond.items() if c <= rare_df_le]
    rare_global.sort(key=lambda x: x[1])
    print(f"global rare conditions codes: DF <= {rare_df_le} head50 =", rare_global[:50])

    rows = score_samples_by_rare_conditions(train_c, df_cond, idf_cond, keys_map=keys_map, rare_df_le=rare_df_le)
    picked = pick_5_rare_condition_samples(rows, topk=5, require_at_least_one_rare=True)
    print_rare5(picked, rare_show_top=20)

    picked_js = [r["subset_j"] for r in picked]
    return train_c, picked, picked_js


# ============================================================
# 10) all-views IDF + plain/weighted jaccard
# ============================================================
def build_allviews_df_idf(train_subset, keys_map):
    df_all = Counter()
    N = len(train_subset)
    for j in range(N):
        s = _unwrap_sample(train_subset[j])
        union = set()
        for view in ["conditions", "drugs", "procedures"]:
            v = _pick_list(s, keys_map[view])
            toks = _flatten_tokens(v)
            union |= set(toks)
        df_all.update(union)
    idf_all = {t: math.log((N + 1.0) / (c + 1.0)) + 1.0 for t, c in df_all.items()}
    return N, df_all, idf_all

def plain_jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return 0.0 if u == 0 else (len(a & b) / (u + 1e-12))

def weighted_jaccard(a: set, b: set, idf: dict) -> float:
    inter = a & b
    uni = a | b
    wi = sum(idf.get(t, 1.0) for t in inter)
    wu = sum(idf.get(t, 1.0) for t in uni)
    return 0.0 if wu == 0 else (wi / (wu + 1e-12))


# ============================================================
# 11) final audit: rare5 + all_views leakage probe
# ============================================================
@torch.no_grad()
def run_rare5_conditions_allviews_leakage_audit(
    nnet,
    diff_config,
    config_health,
    tokenizers,
    convert_input_fn,
    train_c,
    picked_js,
    picked_info=None,
    sample_steps=20,
    repeats=1,
    repeat_agg="mean",
    strict_context_mask=True,
    decode_threshold=0.5,
):
    keys_map = {
        "conditions": ["conditions_comp", "conditions_comps", "conditions"],
        "procedures": ["procedures_comp", "procedures_comps", "procedures"],
        "drugs":      ["drugs_comp", "drugs_comps", "drugs"],
    }

    _, _, idf_all = build_allviews_df_idf(train_c, keys_map)

    audit = {"picked_js": list(picked_js), "per_sample": []}

    for i, subset_j in enumerate(picked_js):
        indices_one, batch_one = get_one_batch_from_subset(train_c, subset_j, collate_fn_dict)
        batch_one = deepcopy_batch(batch_one)

        per_pattern = eval_one_sample_impute(
            nnet=nnet,
            indices_one=indices_one,
            batch_one_raw=batch_one,
            convert_input_fn=convert_input_fn,
            tokenizers=tokenizers,
            idf=idf_all,
            sample_steps=sample_steps,
            repeats=repeats,
            repeat_agg=repeat_agg,
            mask_mode="all_views",
            strict_context_mask=strict_context_mask,
            decode_threshold=decode_threshold,
            seed_base=528 + 1000 * i,
        )

        pr = per_pattern[0]
        mse = float(pr["mse_agg"])

        plainJ, weightedJ = {}, {}
        pred_cnt, gt_cnt, hit_cnt = {}, {}, {}
        for fk in ["conditions", "drugs", "procedures"]:
            pred_set = set(pr["pred_best_sets"].get(fk, []))
            gt_set   = set(pr["gt_sets"].get(fk, []))
            plainJ[fk] = plain_jaccard(pred_set, gt_set)
            weightedJ[fk] = weighted_jaccard(pred_set, gt_set, idf_all)

            pred_cnt[fk] = len(pred_set)
            gt_cnt[fk] = len(gt_set)
            hit_cnt[fk] = len(pred_set & gt_set)
            print("code_count(pred) =", pred_cnt, " | code_count(gt) =", gt_cnt, " | hit(intersection) =", hit_cnt)

        rare_list = []
        if picked_info is not None and i < len(picked_info):
            rare_list = picked_info[i].get("rare_conditions", picked_info[i].get("rare_codes_sorted", []))

        rec = {
            "subset_j": subset_j,
            "rare_conditions_top20": rare_list[:20],
            "mse_masked_allviews": mse,
            "plain_jaccard": plainJ,
            "weighted_jaccard": weightedJ,
            "disc_weighted_metrics(wrec/wprec/wj)": pr.get("disc_agg", {}),
            "overlap_top_idf": pr.get("overlap_top", {}),
        }
        audit["per_sample"].append(rec)

        print(f"\n=== rare5#{i} subset_j={subset_j} ===")
        if len(rare_list) > 0:
            print("rare_conditions(top20) =", rare_list[:20])
        print("all_views maskedMSE =", mse)
        print("plain Jaccard =", plainJ)
        print("weighted Jaccard =", weightedJ)
        print("disc_agg(wrec/wprec/wj) =", pr.get("disc_agg", {}))
        print("overlap_top_idf =", pr.get("overlap_top", {}))

    return audit

## 判断是否同时包含三个视图情况
# def run_rare5_conditions_allviews_leakage_audit(
#     nnet,
#     diff_config,
#     config_health,
#     tokenizers,
#     convert_input_fn,
#     train_c,
#     picked_js,
#     picked_info=None,
#     sample_steps=20,
#     repeats=1,
#     repeat_agg="mean",
#     strict_context_mask=True,
#     decode_threshold=0.5,
# ):
#     keys_map = {
#         "conditions": ["conditions_comp", "conditions_comps", "conditions"],
#         "procedures": ["procedures_comp", "procedures_comps", "procedures"],
#         "drugs":      ["drugs_comp", "drugs_comps", "drugs"],
#     }
#
#     _, _, idf_all = build_allviews_df_idf(train_c, keys_map)
#
#     audit = {"picked_js": list(picked_js), "per_sample": []}
#
#     for i, subset_j in enumerate(picked_js):
#         indices_one, batch_one = get_one_batch_from_subset(train_c, subset_j, collate_fn_dict)
#         batch_one = deepcopy_batch(batch_one)
#
#         per_pattern = eval_one_sample_impute(
#             nnet=nnet,
#             indices_one=indices_one,
#             batch_one_raw=batch_one,
#             convert_input_fn=convert_input_fn,
#             tokenizers=tokenizers,
#             idf=idf_all,
#             sample_steps=sample_steps,
#             repeats=repeats,
#             repeat_agg=repeat_agg,
#             mask_mode="all_views",
#             strict_context_mask=strict_context_mask,
#             decode_threshold=decode_threshold,
#             seed_base=528 + 1000 * i,
#         )
#
#         pr = per_pattern[0]
#         mse = float(pr["mse_agg"])
#
#         plainJ, weightedJ = {}, {}
#         pred_sets_one, gt_sets_one = {}, {}   # ===== NEW =====
#         for fk in ["conditions", "drugs", "procedures"]:
#             pred_set = set(pr["pred_best_sets"].get(fk, []))
#             gt_set   = set(pr["gt_sets"].get(fk, []))
#
#             pred_sets_one[fk] = pred_set        # ===== NEW =====
#             gt_sets_one[fk]   = gt_set          # ===== NEW =====
#
#             plainJ[fk] = plain_jaccard(pred_set, gt_set)
#             weightedJ[fk] = weighted_jaccard(pred_set, gt_set, idf_all)
#
#         # ===== NEW：三视图“完全包含”判定（GT ⊆ Pred）=====
#         full_ok, full_rep, full_missing = three_view_full_containment_once(
#             pred_sets_one, gt_sets_one,
#             feature_keys=("conditions", "drugs", "procedures"),
#             max_show_missing=30
#         )
#
#         rare_list = []
#         if picked_info is not None and i < len(picked_info):
#             rare_list = picked_info[i].get("rare_conditions", picked_info[i].get("rare_codes_sorted", []))
#
#         rec = {
#             "subset_j": subset_j,
#             "rare_conditions_top20": rare_list[:20],
#             "mse_masked_allviews": mse,
#             "plain_jaccard": plainJ,
#             "weighted_jaccard": weightedJ,
#             "disc_weighted_metrics(wrec/wprec/wj)": pr.get("disc_agg", {}),
#             "overlap_top_idf": pr.get("overlap_top", {}),
#
#             # ===== NEW：你要的“是否完全包含三视图GT” =====
#             "full_contain_3views(GT_subset_of_Pred)": full_ok,
#             "full_contain_report": full_rep,
#             "full_contain_missing_examples": ({} if full_ok else full_missing),
#         }
#         audit["per_sample"].append(rec)
#
#         print(f"\n=== rare5#{i} subset_j={subset_j} ===")
#         if len(rare_list) > 0:
#             print("rare_conditions(top20) =", rare_list[:20])
#         print("all_views maskedMSE =", mse)
#         print("plain Jaccard =", plainJ)
#         print("weighted Jaccard =", weightedJ)
#         print("disc_agg(wrec/wprec/wj) =", pr.get("disc_agg", {}))
#         print("overlap_top_idf =", pr.get("overlap_top", {}))
#
#         # ===== NEW：打印“完全包含”结果 =====
#         print("FULL-CONTAIN(all 3 views, GT⊆Pred) =", full_ok)
#         print("  full_contain_report =", full_rep)
#         if not full_ok:
#             print("  missing_examples =", full_missing)
#
#     return audit


def three_view_full_containment_once(pred_sets, gt_sets,
                                     feature_keys=("conditions","drugs","procedures"),
                                     max_show_missing=30):
    """
    判断：生成 pred_sets 是否完全包含 GT(gt_sets) 的三视图 code
    条件：对每个视图都满足 GT ⊆ Pred
    """
    report = {}
    missing = {}
    ok = True
    for fk in feature_keys:
        P = set(pred_sets.get(fk, set()))
        G = set(gt_sets.get(fk, set()))
        miss = sorted(list(G - P))

        report[fk] = {
            "gt_size": len(G),
            "pred_size": len(P),
            "hit_cnt": len(G & P),
            "miss_cnt": len(miss),
            "contain": (len(miss) == 0),
        }
        missing[fk] = miss[:max_show_missing]
        if len(miss) > 0:
            ok = False
    return ok, report, missing
