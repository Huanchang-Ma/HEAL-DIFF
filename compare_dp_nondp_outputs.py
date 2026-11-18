# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import gc
import json
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
import einops

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset, eICUDataset
from utils import (
    split_by_patient,
    load_pickle,
    save_pickle,
    set_random_seed,
    get_tokenizers,
)
from data import convert_dataset
from diffuser.diff_config import config as diff_config
from config import config
from diffuser.diff_dataset import get_dataset, collate_fn_dict
from diffuser import diff_utils as utils
from diffuser.dpm import NoiseScheduleVP, DPM_Solver
from train_hp import get_task_fn



DATASET = "MIII"
TASK = "PHE"

# DP_TAG = "ckpt260407_DP"
DP_TAG = "ckpt260407"
NONDP_TAG = "ckpt260407_nonDP"

ROOT_LOG = "/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog"
OUT_DIR = os.path.join(ROOT_LOG, "compare_outputs", DATASET, TASK)

DP_BEST_TXT = os.path.join(ROOT_LOG, DP_TAG, DATASET, TASK, "best_model_path.txt")
NONDP_BEST_TXT = os.path.join(ROOT_LOG, NONDP_TAG, DATASET, TASK, "best_model_path.txt")

N_COMPARE_BATCHES = 5   # 先对比前 5 个 batch
MASK_STATIC = True



def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def json_safe(x):
    if torch.is_tensor(x):
        if x.dim() == 0:
            return x.item()
        return x.detach().cpu().tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def read_best_model_info(txt_path: str):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"best_model_path.txt not found: {txt_path}")
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    if len(lines) == 0:
        raise RuntimeError(f"best_model_path.txt is empty: {txt_path}")

    model_path = lines[0]
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "nnet.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model path does not exist: {model_path}")

    num_train_visit = int(lines[1]) if len(lines) > 1 else None
    return model_path, num_train_visit


def _pick_device(cfg_gpu):
    s = str(cfg_gpu).strip() if cfg_gpu is not None else "0"
    if s.lower() in ("cpu", "-1", ""):
        return torch.device("cpu")
    if "," in s:
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
        torch.cuda.set_device(dev)
    except Exception as e:
        print(f"[warn] set_device 失败：{e}；继续使用 {dev}")
    return dev


def build_dataset_and_tokenizers(diff_config, config):
    root_to = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/{}/{}/processed/'.format(
        config['TASK'], config['DATASET']
    )

    if not os.path.exists(root_to + 'datasets_pre_stand.pkl'):
        raise FileNotFoundError(
            f"没有找到 {root_to + 'datasets_pre_stand.pkl'}。"
            f"请先按原训练脚本准备好数据。"
        )

    samples = load_pickle(root_to + 'datasets_pre_stand.pkl')

    # 与主脚本一致
    sample_dataset = convert_dataset(
        samples,
        dataset_name=config['DATASET'],
        task_name=config['TASK'],
        valid=True
    )

    tokenizers = get_tokenizers(sample_dataset)

    if os.path.exists(root_to + 'train_dataset.pkl'):
        train_samples = load_pickle(root_to + 'train_dataset.pkl')
        val_samples = load_pickle(root_to + 'val_dataset.pkl')
        test_samples = load_pickle(root_to + 'test_dataset.pkl')

        train_dataset = convert_dataset(
            train_samples,
            dataset_name=config['DATASET'],
            task_name=config['TASK'],
            valid=False
        )
        val_dataset = convert_dataset(
            val_samples,
            dataset_name=config['DATASET'],
            task_name=config['TASK'],
            valid=False
        )
        test_dataset = convert_dataset(
            test_samples,
            dataset_name=config['DATASET'],
            task_name=config['TASK'],
            valid=False
        )
    else:
        train_dataset, val_dataset, test_dataset = split_by_patient(
            sample_dataset,
            [config['RATIO'], (1 - config['RATIO']) / 2, (1 - config['RATIO']) / 2],
            train_ratio=1.0,
            warm_cold=False,
            seed=config['SEED']
        )

    hp_dataset = [train_dataset, val_dataset]

    dataset = get_dataset(
        config['DATASET'],
        aug=False,
        samples=hp_dataset,
        config=diff_config,
    )

    return dataset, hp_dataset, tokenizers, train_dataset, val_dataset, test_dataset, sample_dataset



def convert_input(batch_data, device, tokenizers, mask_static=True):
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

    if mask_static:
        batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
    else:
        m = 3
        batch['mask'] = utils.generate_mask(batch['conditions'].shape[0], m=m).to(device=device)

    aligned_data = [
        batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1)
        for feature_key in feature_keys
    ]
    batch['mask_hist'] = torch.cat(aligned_data, dim=-1)

    return indices.to(device), batch



def get_decode_threshold(diff_config, feature_key):
    default_thr = float(diff_config.get("threshod", 0.5))
    threshold_map = diff_config.get("sample", {}).get("decode_thresholds", {})
    if isinstance(threshold_map, dict) and feature_key in threshold_map:
        return float(threshold_map[feature_key])
    return default_thr


def get_decode_topk(diff_config, feature_key):
    topk_map = diff_config.get("sample", {}).get("decode_topk", {})
    if isinstance(topk_map, dict) and feature_key in topk_map:
        val = topk_map[feature_key]
        return None if val is None else int(val)
    if feature_key == "conditions":
        return int(diff_config.get("sample", {}).get("conditions_topk", 20))
    return None


def decode_with_stats(model, tokenizers, diff_config, latent_batch):
    model.eval()
    hard_samples = {}
    stats = {}
    feature_order = ['conditions', 'procedures', 'drugs']

    for feat_idx, feature_key in enumerate(feature_order):
        logits = model.get_logits(latent_batch[:, feat_idx, :], feature_key)
        probs = torch.sigmoid(logits)
        max_indices = torch.argmax(probs, dim=1)

        topk = get_decode_topk(diff_config, feature_key)
        thr = get_decode_threshold(diff_config, feature_key)

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

    return hard_samples, stats



def sample_once(model, diff_config, tokenizers, device, z_init, context, mask, apply_shrink=False):
    _betas = utils.stable_diffusion_beta_schedule()
    _schedule = utils.Schedule(_betas)

    _z_init = torch.randn(z_init.shape[0], *diff_config['z_shape'], device=device)
    z_init2 = z_init
    mask_expand = mask.unsqueeze(dim=-1).repeat(1, 1, diff_config['z_shape'][1])
    _z_init = torch.where(mask_expand, _z_init, z_init2)

    noise_schedule = NoiseScheduleVP(
        schedule='discrete',
        betas=torch.tensor(_betas, device=device).float()
    )

    def cfg_nnet(x, timesteps, context_):
        return model(x, timesteps, context=context_)

    def model_fn(x, t_continuous):
        t = t_continuous * _schedule.N
        return cfg_nnet(x, t, context)

    dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
    samples = dpm_solver.sample(
        _z_init,
        steps=diff_config['sample']['sample_steps'],
        eps=1. / _schedule.N,
        T=1.
    )

    if apply_shrink:
        samples = z_init + 0.3 * (samples - z_init)

    hard_samples, stats = decode_with_stats(model, tokenizers, diff_config, samples)
    latent_shift = ((samples - z_init) ** 2).reshape(samples.shape[0], -1).mean().item()

    return samples, hard_samples, stats, latent_shift



def unwrap_state_dict_if_needed(obj):
    if isinstance(obj, dict):
        # 常见几种可能
        for key in ["nnet", "model", "state_dict", "module", "network"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj



def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    diff_cfg = copy.deepcopy(diff_config)
    # cfg = copy.deepcopy(config)
    cfg = dict(config)

    cfg["DATASET"] = DATASET
    cfg["TASK"] = TASK

    set_random_seed(cfg['SEED'])
    set_all_seeds(cfg['SEED'])

    device = _pick_device(cfg.get('GPU', 0))
    diff_cfg['device'] = int(str(cfg.get('GPU', '0')).split(',')[0]) if device.type == "cuda" else "cpu"

    # 跟主脚本保持一致：MIV-Note 特判
    if cfg['DATASET'] == "MIV-Note":
        diff_cfg['nnet']['num_contx_token'] = 9

    dp_model_path, dp_num_train_visit = read_best_model_info(DP_BEST_TXT)
    nondp_model_path, nondp_num_train_visit = read_best_model_info(NONDP_BEST_TXT)

    print("[DP MODEL PATH]", dp_model_path)
    print("[NON-DP MODEL PATH]", nondp_model_path)
    print("[DP num_train_visit]", dp_num_train_visit)
    print("[NON-DP num_train_visit]", nondp_num_train_visit)

    dataset, hp_dataset, tokenizers, train_dataset, val_dataset, test_dataset, sample_dataset = \
        build_dataset_and_tokenizers(diff_cfg, cfg)

    diff_test_dataset = dataset.get_split(split='test', labeled=True)

    test_loader = DataLoader(
        diff_test_dataset,
        batch_size=diff_cfg['sample']['mini_batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_dict
    )

    # 初始化两套模型结构
    train_state_non_dp = utils.initialize_train_state(
        diff_cfg, device, tokenizers, nondp_num_train_visit
    )
    train_state_dp = utils.initialize_train_state(
        diff_cfg, device, tokenizers, dp_num_train_visit
    )

    non_dp_model = train_state_non_dp.nnet.to(device)
    dp_model = train_state_dp.nnet.to(device)

    nondp_state = torch.load(nondp_model_path, map_location=device, weights_only=False)
    dp_state = torch.load(dp_model_path, map_location=device, weights_only=False)

    nondp_state = unwrap_state_dict_if_needed(nondp_state)
    dp_state = unwrap_state_dict_if_needed(dp_state)

    missing1, unexpected1 = non_dp_model.load_state_dict(nondp_state, strict=False)
    missing2, unexpected2 = dp_model.load_state_dict(dp_state, strict=False)

    print("[LOAD NON-DP] missing keys:", len(missing1), "unexpected keys:", len(unexpected1))
    print("[LOAD DP] missing keys:", len(missing2), "unexpected keys:", len(unexpected2))

    non_dp_model.eval()
    dp_model.eval()

    results = []

    with torch.no_grad():
        for batch_id, batch_data in enumerate(test_loader):
            indices, batch = convert_input(batch_data, device, tokenizers, mask_static=MASK_STATIC)

            # 同一输入下 encode
            z_non_dp, context_non_dp, _, labels_non_dp, hard_labels_non_dp = \
                non_dp_model.encode_everything_test(indices, **batch)

            z_dp, context_dp, _, labels_dp, hard_labels_dp = \
                dp_model.encode_everything_test(indices, **batch)

            z_mse = F.mse_loss(z_non_dp, z_dp).item()
            context_mse = F.mse_loss(context_non_dp, context_dp).item()

            # 固定同一随机种子，分别生成
            set_all_seeds(cfg['SEED'] + batch_id)
            samples_non_dp, hard_non_dp, stats_non_dp, latent_shift_non_dp = sample_once(
                non_dp_model, diff_cfg, tokenizers, device,
                z_init=z_non_dp, context=context_non_dp, mask=batch['mask'], apply_shrink=False
            )

            set_all_seeds(cfg['SEED'] + batch_id)
            samples_dp, hard_dp, stats_dp, latent_shift_dp = sample_once(
                dp_model, diff_cfg, tokenizers, device,
                z_init=z_dp, context=context_dp, mask=batch['mask'], apply_shrink=True
            )

            for i in range(indices.size(0)):
                results.append({
                    "batch_id": batch_id,
                    "sample_id": int(indices[i].item()),

                    "z_mse_batch": z_mse,
                    "context_mse_batch": context_mse,
                    "latent_shift_non_dp_batch": latent_shift_non_dp,
                    "latent_shift_dp_batch": latent_shift_dp,

                    "gt_conditions": json_safe(batch["conditions"][i]),
                    "gt_procedures": json_safe(batch["procedures"][i]),
                    "gt_drugs": json_safe(batch["drugs"][i]),

                    "pred_non_dp_conditions": json_safe(hard_non_dp["conditions"][i]),
                    "pred_non_dp_procedures": json_safe(hard_non_dp["procedures"][i]),
                    "pred_non_dp_drugs": json_safe(hard_non_dp["drugs"][i]),

                    "pred_dp_conditions": json_safe(hard_dp["conditions"][i]),
                    "pred_dp_procedures": json_safe(hard_dp["procedures"][i]),
                    "pred_dp_drugs": json_safe(hard_dp["drugs"][i]),

                    "cond_conf_non_dp": float(stats_non_dp["conditions_conf"][i].item()),
                    "proc_conf_non_dp": float(stats_non_dp["procedures_conf"][i].item()),
                    "drug_conf_non_dp": float(stats_non_dp["drugs_conf"][i].item()),

                    "cond_conf_dp": float(stats_dp["conditions_conf"][i].item()),
                    "proc_conf_dp": float(stats_dp["procedures_conf"][i].item()),
                    "drug_conf_dp": float(stats_dp["drugs_conf"][i].item()),

                    "cond_len_non_dp": float(stats_non_dp["conditions_len"][i].item()),
                    "proc_len_non_dp": float(stats_non_dp["procedures_len"][i].item()),
                    "drug_len_non_dp": float(stats_non_dp["drugs_len"][i].item()),

                    "cond_len_dp": float(stats_dp["conditions_len"][i].item()),
                    "proc_len_dp": float(stats_dp["procedures_len"][i].item()),
                    "drug_len_dp": float(stats_dp["drugs_len"][i].item()),
                })

            print(
                f"[COMPARE] batch={batch_id} "
                f"z_mse={z_mse:.6f} context_mse={context_mse:.6f} "
                f"latent_shift_non_dp={latent_shift_non_dp:.6f} "
                f"latent_shift_dp={latent_shift_dp:.6f}"
            )

            if batch_id >= N_COMPARE_BATCHES - 1:
                break

    out_json = os.path.join(OUT_DIR, "dp_vs_nondp_outputs.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[DONE] saved compare results to {out_json}")


if __name__ == "__main__":
    main()