# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
DiffMV <-> TarDiff bridge entrypoint.

Design rules:
1. do not modify the original `main_unify.py` path
2. keep `TarDiff-master/` untouched
3. reuse TarDiff's gradient-dot guidance idea on top of the existing
   downstream model and augmented EHR samples
"""

from __future__ import annotations

import copy
import json
import os
from typing import Dict, Tuple

import torch

from config import config as base_config
from data import convert_dataset
from diffuser.diff_config import config as diff_config
from eval_dffs import augment_dataset, augment_dataset1, load_diffusion
from loader import get_dataloader, get_special_input
from tardiff_bridge.filtering import filter_augmented_samples_by_graddot
from tardiff_bridge.sampling_guidance import TarDiffSamplingGuidance
from tardiff_bridge_config import config as tardiff_cfg
from train_dffs import run_diffusion
from train_hp import run_pcf, underlying_model
from utils import get_tokenizers, load_pickle, save_pickle, set_random_seed


def _safe_config_copy(cfg: Dict) -> Dict:
    out = {}
    for key, value in cfg.items():
        try:
            out[key] = copy.deepcopy(value)
        except Exception:
            out[key] = value
    return out


def _default_root_to(task_config: Dict) -> str:
    root_to = "/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/{}/{}/processed/".format(
        task_config["TASK"], task_config["DATASET"]
    )
    # root_to = "/home/exp1/mhc/MMHealth-main-hjm/data/{}/{}/processed/".format(
    #     task_config["TASK"], task_config["DATASET"]
    # )
    return root_to


def _infer_mode(task_name: str) -> str:
    if task_name in ["PHE", "REC"]:
        return "multilabel"
    if task_name in ["DIAG", "LOS"]:
        return "multiclass"
    if task_name in ["MOR", "REA"]:
        return "binary"
    return "multilabel"


def _build_dataloaders_from_samples(train_samples, val_samples, test_samples, task_config: Dict):
    sample_dataset = convert_dataset(
        train_samples + val_samples + test_samples,
        dataset_name=task_config["DATASET"],
        task_name=task_config["TASK"],
        valid=True,
    )
    train_dataset = convert_dataset(train_samples, dataset_name=task_config["DATASET"], task_name=task_config["TASK"], valid=False)
    val_dataset = convert_dataset(val_samples, dataset_name=task_config["DATASET"], task_name=task_config["TASK"], valid=False)
    test_dataset = convert_dataset(test_samples, dataset_name=task_config["DATASET"], task_name=task_config["TASK"], valid=False)

    collate_fn = get_special_input(task_config)
    train_loader = get_dataloader(train_dataset, batch_size=task_config["BATCH"], shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = get_dataloader(val_dataset, batch_size=task_config["BATCH"], shuffle=False, drop_last=False, collate_fn=collate_fn)
    test_loader = get_dataloader(test_dataset, batch_size=task_config["BATCH"], shuffle=False, drop_last=False, collate_fn=collate_fn)
    return sample_dataset, train_loader, val_loader, test_loader


def _as_sample_list(data_obj):
    if isinstance(data_obj, list):
        return data_obj
    if hasattr(data_obj, "samples"):
        return data_obj.samples
    try:
        return list(data_obj)
    except TypeError:
        return data_obj


def _build_reference_sample_dataset(root_to: str, task_config: Dict):
    real_train, real_val, real_test = _load_real_datasets(root_to)
    return convert_dataset(
        _as_sample_list(real_train) + _as_sample_list(real_val) + _as_sample_list(real_test),
        dataset_name=task_config["DATASET"],
        task_name=task_config["TASK"],
        valid=False,
        all=True,
    )


def _with_default_aug_fields(base_collate_fn):
    def wrapped(batch):
        data = base_collate_fn(batch)
        data.setdefault("new_conditions", 0)
        data.setdefault("new_procedures", 0)
        data.setdefault("new_drugs", 0)
        data.setdefault("miss_feature", 0)
        return data

    return wrapped


def _build_real_dataloaders_for_guide(train_samples, val_samples, test_samples, task_config: Dict):
    sample_dataset = convert_dataset(
        train_samples + val_samples + test_samples,
        dataset_name=task_config["DATASET"],
        task_name=task_config["TASK"],
        valid=True,
    )
    train_dataset = convert_dataset(train_samples, dataset_name=task_config["DATASET"], task_name=task_config["TASK"], valid=False)
    val_dataset = convert_dataset(val_samples, dataset_name=task_config["DATASET"], task_name=task_config["TASK"], valid=False)
    test_dataset = convert_dataset(test_samples, dataset_name=task_config["DATASET"], task_name=task_config["TASK"], valid=False)

    collate_fn = _with_default_aug_fields(get_special_input(task_config))
    train_loader = get_dataloader(train_dataset, batch_size=task_config["BATCH"], shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = get_dataloader(val_dataset, batch_size=task_config["BATCH"], shuffle=False, drop_last=False, collate_fn=collate_fn)
    test_loader = get_dataloader(test_dataset, batch_size=task_config["BATCH"], shuffle=False, drop_last=False, collate_fn=collate_fn)
    return sample_dataset, train_loader, val_loader, test_loader


def _load_group_info(root_to: str):
    path = os.path.join(root_to, "group_patient.pkl")
    if os.path.exists(path):
        return load_pickle(path)
    return None


def _ensure_required_files(root_to: str):
    required = [
        "train_dataset.pkl",
        "val_dataset.pkl",
        "test_dataset.pkl",
        "train_samples.pkl",
        "val_samples.pkl",
        "test_samples.pkl",
    ]
    missing = [name for name in required if not os.path.exists(os.path.join(root_to, name))]
    if missing:
        raise FileNotFoundError(
            "TarDiff one-click run needs these files in processed dir: "
            + ", ".join(missing)
        )


def _run_final_downstream(train_samples, val_samples, test_samples, task_config: Dict, exp_num: str, aux_data: Dict, sample_dataset=None):
    built_sample_dataset, train_loader, val_loader, test_loader = _build_dataloaders_from_samples(
        train_samples, val_samples, test_samples, task_config
    )
    if sample_dataset is None:
        sample_dataset = built_sample_dataset

    set_random_seed(task_config["SEED"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    mode = _infer_mode(task_config["TASK"])
    scores, model = run_pcf(
        sample_dataset=sample_dataset,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        config=task_config,
        exp_num=exp_num,
        existing_model=None,
        mode=mode,
        aux_data=aux_data,
    )
    return scores, model


def _score_delta(base_scores: Dict, new_scores: Dict) -> Dict:
    delta = {}
    for key, value in new_scores.items():
        if key in base_scores and isinstance(value, (int, float)) and isinstance(base_scores[key], (int, float)):
            delta[key] = float(value - base_scores[key])
    return delta


def _strip_sample_fields(samples, fields_to_remove):
    cleaned = []
    for sample in samples:
        copied = dict(sample)
        for field in fields_to_remove:
            copied.pop(field, None)
        cleaned.append(copied)
    return cleaned


def _guide_exp_dir(task_config: Dict, exp_num: str) -> str:
    return os.path.join(
        task_config["LOGDIR"] + task_config["TASK"] + '/',
        task_config["DATASET"] + '-' + task_config["MODEL"] + '-' + exp_num,
    )


def _load_guide_model_from_ckpt(sample_dataset, task_config: Dict, mode: str, ckpt_path: str):
    model = underlying_model(
        task_config,
        sample_dataset,
        feature_keys=task_config["FEATURE"],
        label_key=task_config["LABEL"],
        special_input=None,
        mode=mode,
    )
    device = 'cuda:' + task_config["GPU"] if task_config["USE_CUDA"] else 'cpu'
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def train_guidance_model(root_to: str, task_config: Dict, exp_num: str = "tardiff-guide"):
    real_train = load_pickle(os.path.join(root_to, "train_dataset.pkl"))
    real_val = load_pickle(os.path.join(root_to, "val_dataset.pkl"))
    real_test = load_pickle(os.path.join(root_to, "test_dataset.pkl"))

    sample_dataset, train_loader, val_loader, test_loader = _build_real_dataloaders_for_guide(
        real_train, real_val, real_test, task_config
    )
    mode = _infer_mode(task_config["TASK"])
    guide_dir = _guide_exp_dir(task_config, exp_num)
    guide_ckpt = os.path.join(guide_dir, "best.ckpt")

    if tardiff_cfg.reuse_existing_guide and os.path.exists(guide_ckpt):
        print(f"[TarDiff] Reusing existing guide model: {guide_ckpt}")
        guide_model = _load_guide_model_from_ckpt(sample_dataset, task_config, mode, guide_ckpt)
        return guide_model, train_loader

    print(f"[TarDiff] Training new guide model: {guide_dir}")
    _, guide_model = run_pcf(
        sample_dataset=sample_dataset,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        config=task_config,
        exp_num=exp_num,
        existing_model=None,
        mode=mode,
        aux_data={},
    )
    return guide_model, train_loader


def build_sampling_guidance(root_to: str, task_config: Dict, exp_num: str = "tardiff-guide"):
    guide_model, train_loader = train_guidance_model(root_to, task_config, exp_num=exp_num)
    return TarDiffSamplingGuidance(
        guide_model=guide_model,
        train_loader=train_loader,
        device='cuda:' + task_config["GPU"] if task_config["USE_CUDA"] else 'cpu',
        normalize=tardiff_cfg.normalize,
        positive_guidance=tardiff_cfg.positive_guidance,
        negative_guidance=tardiff_cfg.negative_guidance,
        guidance_scale=tardiff_cfg.guidance_scale,
        detach_x0=tardiff_cfg.guidance_detach_x0,
    )

def _load_real_datasets(root_to: str):
    return (
        load_pickle(os.path.join(root_to, "train_dataset.pkl")),
        load_pickle(os.path.join(root_to, "val_dataset.pkl")),
        load_pickle(os.path.join(root_to, "test_dataset.pkl")),
    )


def _build_tokenizers_from_real(root_to: str, task_config: Dict):
    real_train, real_val, real_test = _load_real_datasets(root_to)
    sample_dataset = convert_dataset(
        real_train + real_val + real_test,
        dataset_name=task_config["DATASET"],
        task_name=task_config["TASK"],
        valid=True,
    )
    return get_tokenizers(sample_dataset)


def _diffusion_info_path(task_config: Dict) -> str:
    return os.path.join(
        diff_config['ckpt_root'],
        task_config['DATASET'],
        task_config['TASK'],
        'best_model_path.txt',
    )


def _sync_diffusion_runtime_config(task_config: Dict):
    diff_config['device'] = int(task_config['GPU']) if task_config.get('USE_CUDA', False) else 'cpu'
    diff_config['dataset']['name'] = task_config['DATASET']


def _load_or_train_diffusion_model(root_to: str, task_config: Dict, exp_num: str, tokenizers):
    _sync_diffusion_runtime_config(task_config)
    info_path = _diffusion_info_path(task_config)
    if tardiff_cfg.reuse_existing_diffusion and os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]
        if len(lines) >= 2:
            diff_config['nnet_path'] = lines[0]
            diff_config['num_train_visit'] = int(lines[1])
            print(f"[TarDiff] Reusing existing diffusion model: {diff_config['nnet_path']}")
            return load_diffusion(diff_config, tokenizers)

    real_train, real_val, _ = _load_real_datasets(root_to)
    print('[TarDiff] Training diffusion model for guided generation')
    best_model_path, generated_model, num_train_visit = run_diffusion(
        diff_config,
        task_config,
        [real_train, real_val],
        tokenizers,
        exp_num=exp_num + '-diffusion',
    )
    diff_config['nnet_path'] = best_model_path
    diff_config['num_train_visit'] = num_train_visit
    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    with open(info_path, 'w', encoding='utf-8') as file:
        file.write(best_model_path + '\n')
        file.write(str(num_train_visit) + '\n')
    return generated_model


def _runtime_summary(root_to: str) -> Dict:
    return {
        'mode': 'sampling_guidance' if tardiff_cfg.use_sampling_guidance else 'post_filter',
        'guidance_scale': tardiff_cfg.guidance_scale,
        'guidance_detach_x0': tardiff_cfg.guidance_detach_x0,
        'normalize': tardiff_cfg.normalize,
        'positive_guidance': tardiff_cfg.positive_guidance,
        'negative_guidance': tardiff_cfg.negative_guidance,
        'guided_save_suffix': tardiff_cfg.guided_save_suffix,
        'reuse_existing_guide': tardiff_cfg.reuse_existing_guide,
        'reuse_existing_diffusion': tardiff_cfg.reuse_existing_diffusion,
        'reuse_existing_guided_samples': tardiff_cfg.reuse_existing_guided_samples,
        'run_original_mix': tardiff_cfg.run_original_mix,
        'root_to': root_to,
    }


def _print_runtime_summary(root_to: str):
    summary = _runtime_summary(root_to)
    print('[TarDiff] Runtime config:')
    for key, value in summary.items():
        print(f'[TarDiff]   {key}: {value}')


def _guided_sample_paths(root_to: str):
    suffix = tardiff_cfg.guided_save_suffix
    return {
        'train': os.path.join(root_to, f'train_samples{suffix}.pkl'),
        'val': os.path.join(root_to, f'val_samples{suffix}.pkl'),
        'test': os.path.join(root_to, f'test_samples{suffix}.pkl'),
    }


def _baseline_sample_paths(root_to: str):
    return {
        'train': os.path.join(root_to, 'train_samples.pkl'),
        'val': os.path.join(root_to, 'val_samples.pkl'),
        'test': os.path.join(root_to, 'test_samples.pkl'),
    }


def _maybe_generate_baseline_samples(root_to: str, task_config: Dict, tokenizers, generated_model):
    paths = _baseline_sample_paths(root_to)
    if all(os.path.exists(path) for path in paths.values()):
        print('[TarDiff] Reusing existing baseline diffusion samples')
        return (
            load_pickle(paths['train']),
            load_pickle(paths['val']),
            load_pickle(paths['test']),
        )

    real_train, real_val, real_test = _load_real_datasets(root_to)
    print('[TarDiff] Generating baseline diffusion samples')
    train_samples, val_samples = augment_dataset(
        generated_model,
        diff_config,
        task_config,
        [real_train, real_val],
        tokenizers,
        root_to=root_to,
        sampling_guidance=None,
        save_suffix='',
    )
    test_samples = augment_dataset1(
        generated_model,
        diff_config,
        task_config,
        [real_train, real_test],
        tokenizers,
        root_to=root_to,
        sampling_guidance=None,
        save_suffix='',
    )
    return train_samples, val_samples, test_samples


def _maybe_generate_guided_samples(root_to: str, task_config: Dict, tokenizers, generated_model, sampling_guidance):
    paths = _guided_sample_paths(root_to)
    if tardiff_cfg.reuse_existing_guided_samples and all(os.path.exists(path) for path in paths.values()):
        print('[TarDiff] Reusing existing guided diffusion samples')
        return (
            load_pickle(paths['train']),
            load_pickle(paths['val']),
            load_pickle(paths['test']),
        )

    real_train, real_val, real_test = _load_real_datasets(root_to)
    print('[TarDiff] Generating TarDiff-guided diffusion samples')
    train_samples, val_samples = augment_dataset(
        generated_model,
        diff_config,
        task_config,
        [real_train, real_val],
        tokenizers,
        root_to=root_to,
        sampling_guidance=sampling_guidance,
        save_suffix=tardiff_cfg.guided_save_suffix,
    )
    test_samples = augment_dataset1(
        generated_model,
        diff_config,
        task_config,
        [real_train, real_test],
        tokenizers,
        root_to=root_to,
        sampling_guidance=sampling_guidance,
        save_suffix=tardiff_cfg.guided_save_suffix,
    )
    return train_samples, val_samples, test_samples


def run_tardiff_guided_generation(root_to: str | None = None, exp_num: str = 'tardiff'):
    cfg = _safe_config_copy(base_config)
    cfg['USE_NEW_MERGE'] = tardiff_cfg.use_new_merge
    set_random_seed(tardiff_cfg.random_seed)
    root_to = root_to or os.environ.get('TARDIFF_ROOT', '') or _default_root_to(cfg)
    if not root_to.endswith(os.sep):
        root_to = root_to + os.sep

    required = ['train_dataset.pkl', 'val_dataset.pkl', 'test_dataset.pkl']
    missing = [name for name in required if not os.path.exists(os.path.join(root_to, name))]
    if missing:
        raise FileNotFoundError('TarDiff guided run needs these files in processed dir: ' + ', '.join(missing))

    _print_runtime_summary(root_to)

    aux_data = {}
    p_grouped = _load_group_info(root_to)
    if p_grouped is not None:
        aux_data['p_grouped'] = p_grouped
    reference_sample_dataset = _build_reference_sample_dataset(root_to, cfg)

    print('[TarDiff] Step 1/6: train or reuse guidance model')
    sampling_guidance = build_sampling_guidance(root_to=root_to, task_config=cfg, exp_num=exp_num + '-guide')

    print('[TarDiff] Step 2/6: build tokenizers from real datasets')
    tokenizers = _build_tokenizers_from_real(root_to, cfg)

    print('[TarDiff] Step 3/6: train or reuse diffusion model')
    generated_model = _load_or_train_diffusion_model(root_to, cfg, exp_num, tokenizers)

    print('[TarDiff] Step 4/6: generate or reuse baseline diffusion samples')
    baseline_train, baseline_val, baseline_test = _maybe_generate_baseline_samples(root_to, cfg, tokenizers, generated_model)

    print('[TarDiff] Step 5/6: generate or reuse TarDiff-guided diffusion samples')
    guided_train, guided_val, guided_test = _maybe_generate_guided_samples(root_to, cfg, tokenizers, generated_model, sampling_guidance)
    print('[TarDiff] Guided sample paths:', _guided_sample_paths(root_to))

    print('[TarDiff] Step 6/6: run downstream comparison')
    baseline_scores = None
    if tardiff_cfg.run_original_mix:
        baseline_scores, _ = _run_final_downstream(
            train_samples=baseline_train,
            val_samples=baseline_val,
            test_samples=baseline_test,
            task_config=cfg,
            exp_num=exp_num + '-origmix',
            aux_data=aux_data,
            sample_dataset=reference_sample_dataset,
        )
    else:
        print('[TarDiff] Skipping original mix downstream run because run_original_mix=False')
    tardiff_scores, _ = _run_final_downstream(
        train_samples=guided_train,
        val_samples=guided_val,
        test_samples=guided_test,
        task_config=cfg,
        exp_num=exp_num + '-tardiffguided',
        aux_data=aux_data,
        sample_dataset=reference_sample_dataset,
    )

    summary = {
        'runtime': _runtime_summary(root_to),
        'baseline_paths': _baseline_sample_paths(root_to),
        'guided_paths': _guided_sample_paths(root_to),
        'baseline_mix_scores': baseline_scores,
        'tardiff_mix_scores': tardiff_scores,
        'delta_tardiff_minus_baseline': _score_delta(baseline_scores, tardiff_scores) if baseline_scores is not None else None,
    }
    compare_path = os.path.join(root_to, f'tardiff_compare_{exp_num}.json')
    with open(compare_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('[TarDiff] compare summary saved to:', compare_path)
    print('[TarDiff] baseline_mix_scores:', baseline_scores)
    print('[TarDiff] tardiff_mix_scores:', tardiff_scores)
    print('[TarDiff] delta_tardiff_minus_baseline:', summary['delta_tardiff_minus_baseline'])
    return summary


def run_tardiff_filter(root_to: str | None = None, exp_num: str = "tardiff"):
    cfg = _safe_config_copy(base_config)
    cfg["USE_NEW_MERGE"] = tardiff_cfg.use_new_merge
    set_random_seed(tardiff_cfg.random_seed)
    root_to = root_to or os.environ.get("TARDIFF_ROOT", "") or _default_root_to(cfg)
    if not root_to.endswith(os.sep):
        root_to = root_to + os.sep
    _ensure_required_files(root_to)

    aux_data = {}
    p_grouped = _load_group_info(root_to)
    if p_grouped is not None:
        aux_data["p_grouped"] = p_grouped
    reference_sample_dataset = _build_reference_sample_dataset(root_to, cfg)

    print("[TarDiff] Step 1/4: train guidance model on real datasets")
    guide_model, real_train_loader = train_guidance_model(root_to=root_to, task_config=cfg, exp_num=exp_num + "-guide")

    print("[TarDiff] Step 2/4: score and filter synthetic train samples")
    train_samples = load_pickle(os.path.join(root_to, "train_samples.pkl"))
    val_samples = load_pickle(os.path.join(root_to, "val_samples.pkl"))
    test_samples = load_pickle(os.path.join(root_to, "test_samples.pkl"))
    filtered_train_scored, scores, kept_idx = filter_augmented_samples_by_graddot(
        guide_model=guide_model,
        real_train_loader=real_train_loader,
        augmented_samples=train_samples,
        task_config=cfg,
        keep_ratio=tardiff_cfg.keep_ratio,
        min_keep=tardiff_cfg.min_keep,
        batch_size=tardiff_cfg.batch_size,
        normalize=tardiff_cfg.normalize,
        positive_guidance=tardiff_cfg.positive_guidance,
        negative_guidance=tardiff_cfg.negative_guidance,
        field_name=tardiff_cfg.score_field,
    )
    filtered_train = _strip_sample_fields(filtered_train_scored, [tardiff_cfg.score_field])

    output_path = os.path.join(root_to, f"train_samples_{tardiff_cfg.save_suffix}.pkl")
    save_pickle(filtered_train, output_path)

    filter_summary = {
        "input_path": os.path.join(root_to, "train_samples.pkl"),
        "output_path": output_path,
        "num_input": len(train_samples),
        "num_kept": len(filtered_train),
        "keep_ratio_real": float(len(filtered_train) / max(len(train_samples), 1)),
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
        "score_std": float(scores.std()) if len(scores) else 0.0,
        "top_indices": kept_idx[:10].tolist() if len(kept_idx) else [],
    }
    print("[TarDiff] filter summary:", filter_summary)

    print("[TarDiff] Step 3/4: run original mix downstream experiment")
    baseline_scores, _ = _run_final_downstream(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        task_config=cfg,
        exp_num=exp_num + "-origmix",
        aux_data=aux_data,
        sample_dataset=reference_sample_dataset,
    )

    print("[TarDiff] Step 4/4: run TarDiff mix downstream experiment")
    tardiff_scores, _ = _run_final_downstream(
        train_samples=filtered_train,
        val_samples=val_samples,
        test_samples=test_samples,
        task_config=cfg,
        exp_num=exp_num + "-tardiffmix",
        aux_data=aux_data,
        sample_dataset=reference_sample_dataset,
    )

    compare_summary = {
        "filter": filter_summary,
        "baseline_mix_scores": baseline_scores,
        "tardiff_mix_scores": tardiff_scores,
        "delta_tardiff_minus_baseline": _score_delta(baseline_scores, tardiff_scores),
    }
    compare_path = os.path.join(root_to, f"tardiff_compare_{exp_num}.json")
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(compare_summary, f, ensure_ascii=False, indent=2)

    print("[TarDiff] compare summary saved to:", compare_path)
    print("[TarDiff] baseline_mix_scores:", baseline_scores)
    print("[TarDiff] tardiff_mix_scores:", tardiff_scores)
    print("[TarDiff] delta_tardiff_minus_baseline:", compare_summary["delta_tardiff_minus_baseline"])
    return compare_summary


if __name__ == "__main__":
    if tardiff_cfg.use_sampling_guidance:
        run_tardiff_guided_generation()
    else:
        run_tardiff_filter()
