#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化 — train / val / test 三组 mask ratio sweep。
独立新文件，不修改现有代码。
用法：跑完 main_unify.py 或 main_unify_DiffmvDP0420_ablation.py 后，
      确保 diff_config.py 的 ckpt_root 指向正确的 ckpt 目录，然后：
      python main_unify_allviews_Visualization_new.py
"""

import os, sys, shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from pyhealth.datasets import SampleEHRDataset as SampleEHRDatasets
from utils import load_pickle, get_tokenizers
from data import convert_dataset
from diffuser.diff_config import config as diff_config
from config import config
from eval_dffs import load_diffusion, load_diffusion_dp, eval_diffusion_latent_mae_rmse_allviews


class SampleEHRDataset(SampleEHRDatasets):
    def _validate(self):
        return True


def _plot_one(ax, results, metric, label, color):
    ratios = sorted(results.keys())
    vals = [results[r][metric] for r in ratios]
    ax.plot(ratios, vals, marker="o", color=color, linewidth=2, label=label)


def save_all(results_cond, results_proc, results_drug, save_dir, split_name):
    """保存 evaluation_metrics.log 和 PNG 图"""
    os.makedirs(save_dir, exist_ok=True)

    # log
    with open(os.path.join(save_dir, f"evaluation_metrics_{split_name}.log"), "w", encoding="utf-8") as f:
        for label, results in [("conditions", results_cond), ("procedures", results_proc), ("drugs", results_drug)]:
            f.write(f"===== {label}-scan mask ratio sweep =====\n")
            for ratio in sorted(results.keys()):
                r = results[ratio]
                f.write(f"[{label}-scan] ratio={ratio:.2f}: "
                        f"latent_MAE={r['latent_mae']:.6f}, latent_RMSE={r['latent_rmse']:.6f}, "
                        f"disc_MAE={r['discrete_mae']}, disc_RMSE={r['discrete_rmse']}, "
                        f"Jaccard={r['jaccard']:.6f}, latent_count={r['latent_count']}, "
                        f"disc_count={r['discrete_count']}\n")
            f.write("\n")

    # 三张图
    for metric, title, ylabel in [("latent_mae", "Latent MAE vs Mask Ratio", "Latent MAE"),
                                   ("latent_rmse", "Latent RMSE vs Mask Ratio", "Latent RMSE"),
                                   ("jaccard", "Jaccard Index vs Mask Ratio", "Jaccard")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_one(ax, results_cond, metric, "Conditions", "#1f77b4")
        _plot_one(ax, results_proc, metric, "Procedures", "#ff7f0e")
        _plot_one(ax, results_drug, metric, "Drugs", "#2ca02c")
        ax.set_xlabel("Mask Ratio"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric}_vs_ratio_{split_name}.png"), dpi=300); plt.close()


def main():
    root_to = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/{}/{}/processed/'.format(
        config["TASK"], config["DATASET"])

    # ---- 加载数据 ----
    samples = load_pickle(root_to + "datasets_pre_stand.pkl")
    train_samples = load_pickle(root_to + "train_dataset.pkl")
    val_samples   = load_pickle(root_to + "val_dataset.pkl")
    test_samples  = load_pickle(root_to + "test_dataset.pkl")

    sample_dataset = convert_dataset(samples, dataset_name=config["DATASET"], task_name=config["TASK"],
                                     valid=False, all=True)
    train_dataset = convert_dataset(train_samples, dataset_name=config["DATASET"], task_name=config["TASK"], valid=False)
    val_dataset   = convert_dataset(val_samples,   dataset_name=config["DATASET"], task_name=config["TASK"], valid=False)
    test_dataset  = convert_dataset(test_samples,  dataset_name=config["DATASET"], task_name=config["TASK"], valid=False)

    # ---- 加载模型 ----
    diff_config["device"] = int(config["GPU"])
    if config["DATASET"] == "MIV-Note":
        diff_config["nnet"]["num_contx_token"] = 9

    tokenizers = get_tokenizers(sample_dataset)
    best_info_path = os.path.join(diff_config["ckpt_root"], config["DATASET"], config["TASK"], "best_model_path.txt")

    with open(best_info_path, "r") as f:
        lines = f.readlines()
    diff_config["nnet_path"] = lines[0].strip()
    diff_config["num_train_visit"] = int(lines[1].strip())

    # 自动判断 original vs ablation（根据 ckpt_root 目录名）
    is_ablation = "ablation" in diff_config.get("ckpt_root", "")
    loader_fn = load_diffusion_dp if is_ablation else load_diffusion
    print(f"[Visualization] using {'load_diffusion_dp' if is_ablation else 'load_diffusion'}")
    model = loader_fn(diff_config, tokenizers)

    # ---- train / val / test 三组 ----
    splits = [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]
    ratios = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    for split_name, split_dataset in splits:
        print(f"\n===== evaluating {split_name} =====")

        # eval_diffusion_latent_mae_rmse_allviews 硬编码读 test_dataset.pkl，
        # 故创建临时目录，把目标 split 的 pkl 改名为 test_dataset.pkl
        temp_root = f"/tmp/vis_{split_name}/"
        os.makedirs(temp_root, exist_ok=True)
        for f in ["train_dataset.pkl", "val_dataset.pkl", "test_dataset.pkl", "datasets_pre_stand.pkl"]:
            src = os.path.join(root_to, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(temp_root, f))
        # 覆盖 test_dataset.pkl 为当前 split
        src_pkl = os.path.join(root_to, f"{split_name}_dataset.pkl")
        if os.path.exists(src_pkl):
            shutil.copy2(src_pkl, os.path.join(temp_root, "test_dataset.pkl"))

        save_dir = f"/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/DP_jiaming/MAE/{split_name}/"

        (results_cond, results_proc, results_drug,
         _, _, _) = eval_diffusion_latent_mae_rmse_allviews(
            nnet=model, diff_config=diff_config, config_health=config,
            hp_dataset=[train_dataset, split_dataset],
            tokenizers=tokenizers, root_to=temp_root, sample_steps=50,
            cond_ratios=ratios, proc_ratios=ratios, drug_ratios=ratios,
            fully_mask_other_views=False,
        )

        save_all(results_cond, results_proc, results_drug, save_dir, split_name)

    print("\nAll done! train/val/test 三组评估完成。")


if __name__ == "__main__":
    main()
