import os
import gc
import json
import copy
import argparse
import subprocess
from pathlib import Path

import pandas as pd
import torch

from config import config
from diffuser.diff_config import config as diff_config
from eval_dffs import augment_dataset, augment_dataset1, load_diffusion_dp, load_diffusion
from train_hp import run_pcf, get_task_fn
from loader import get_dataloader, get_special_input
from data import convert_dataset
from utils import load_pickle, save_pickle, get_tokenizers, set_random_seed


DEFAULT_GEN_SUBDIRS = [
    "ckpt2",
    "ckpt4",
    "ckpt6",
    "ckpt7",
    "ckpt9",
    "ckpt260331-2",
    "ckpt260331-4",
    "ckpt260331-6",
    "ckpt260331-7",
    "ckpt260331-9",
]


def append_csv_row(csv_path, row_dict):
    row_df = pd.DataFrame([row_dict])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        new_df = pd.concat([old_df, row_df], ignore_index=True)
    else:
        new_df = row_df
    new_df.to_csv(csv_path, index=False)


def cleanup_processed_pickles(processed_root):
    """每轮开始前先删掉旧的 samples，确保完全覆盖。"""
    processed_root = Path(processed_root)
    for fname in [
        "samples.pkl",
        "train_samples.pkl",
        "val_samples.pkl",
        "test_samples.pkl",
    ]:
        fpath = processed_root / fname
        if fpath.exists():
            fpath.unlink()
            print(f"[Cleanup] removed: {fpath}")


# def read_best_model_path_txt(difflog_root, gen_subdir, dataset, task):
#     """
#     按用户指定规则读取 best ckpt 路径：
#     - ckpt2/4/6/7/9      -> best_model_path.txt
#     - ckpt260331-*       -> best_model_path_eps8p0.txt
#     """
#     task_dir = Path(difflog_root) / gen_subdir / dataset / task
#
#     if not task_dir.exists():
#         raise FileNotFoundError(f"Task dir not found: {task_dir}")
#
#     if gen_subdir.startswith("ckpt260331-"):
#         txt_name = "best_model_path_eps8p0.txt"
#     else:
#         txt_name = "best_model_path.txt"
#
#     txt_path = task_dir / txt_name
#     if not txt_path.exists():
#         raise FileNotFoundError(f"best model path txt not found: {txt_path}")
#
#     with open(txt_path, "r", encoding="utf-8") as f:
#         best_ckpt_path = f.read().strip()
#
#     if not best_ckpt_path:
#         raise ValueError(f"Empty best model path in: {txt_path}")
#
#     # 若 txt 中写的是相对路径，则相对 task_dir 解析
#     best_ckpt_path_obj = Path(best_ckpt_path)
#     if not best_ckpt_path_obj.is_absolute():
#         best_ckpt_path_obj = (task_dir / best_ckpt_path_obj).resolve()
#
#     if not best_ckpt_path_obj.exists():
#         raise FileNotFoundError(
#             f"best ckpt path from txt does not exist:\n"
#             f"txt={txt_path}\n"
#             f"path={best_ckpt_path_obj}"
#         )
#
#     return str(best_ckpt_path_obj), str(txt_path)
from pathlib import Path

def read_best_model_path_txt(difflog_root, gen_subdir, dataset, task):
    task_dir = Path(difflog_root) / gen_subdir / dataset / task

    if not task_dir.exists():
        raise FileNotFoundError(f"Task dir not found: {task_dir}")

    if gen_subdir.startswith("ckpt260331-"):
        txt_name = "best_model_path_eps8p0.txt"
    else:
        txt_name = "best_model_path.txt"

    txt_path = task_dir / txt_name
    if not txt_path.exists():
        raise FileNotFoundError(f"best model path txt not found: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) == 0:
        raise ValueError(f"Empty txt file: {txt_path}")

    # 第1行：nnet路径
    nnet_path = lines[0]
    nnet_path_obj = Path(nnet_path)
    if not nnet_path_obj.is_absolute():
        nnet_path_obj = (task_dir / nnet_path_obj).resolve()

    if not nnet_path_obj.exists():
        raise FileNotFoundError(
            f"nnet path from txt does not exist:\n"
            f"txt={txt_path}\n"
            f"path={nnet_path_obj}"
        )

    # 第2行：num_train_visit
    num_train_visit = None
    if len(lines) >= 2:
        try:
            num_train_visit = int(lines[1])
        except ValueError:
            print(f"[Warn] second line is not int in {txt_path}: {lines[1]}")

    return str(nnet_path_obj), str(txt_path), num_train_visit


def build_generator_items(difflog_root, gen_subdirs, dataset, task):
    items = []

    for subdir in gen_subdirs:
        nnet_path, path_txt, num_train_visit = read_best_model_path_txt(
            difflog_root=difflog_root,
            gen_subdir=subdir,
            dataset=dataset,
            task=task,
        )
        items.append(
            {
                "gen_subdir": subdir,
                "gen_tag": subdir.replace("/", "_").replace(" ", "_"),
                "gen_root": str(Path(difflog_root) / subdir / dataset / task),
                "path_txt": path_txt,
                "nnet_path": nnet_path,
                "num_train_visit": num_train_visit,
                "use_dp_loader": subdir.startswith("ckpt260331-"),
            }
        )
    return items


def run_all_generators_overwrite_processed(
    diff_config,
    config,
    sample_dataset,
    train_dataset,
    val_dataset,
    test_dataset,
    tokenizers,
    p_grouped,
    mode,
    root_to,
    difflog_root,
    gen_subdirs,
    exp_num="20",
    mmhealth_root="/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/",
):
    summary_csv = os.path.join(
        diff_config["workdir"],
        config["DATASET"],
        config["TASK"],
        "downstream_generator_summary.csv",
    )

    processed_root = root_to  # 固定 processed 路径，循环覆盖
    generator_items = build_generator_items(
        difflog_root=difflog_root,
        gen_subdirs=gen_subdirs,
        dataset=config["DATASET"],
        task=config["TASK"],
    )

    print("\n========== Generator list ==========")
    for item in generator_items:
        print(f"[Generator] {item['gen_tag']}")
        print(f"  path_txt : {item['path_txt']}")
        print(f"  nnet_path: {item['nnet_path']}")
    print("===================================\n")

    for item in generator_items:
        gen_subdir = item["gen_subdir"]
        gen_tag = item["gen_tag"]
        gen_root = item["gen_root"]
        path_txt = item["path_txt"]
        nnet_path = item["nnet_path"]
        num_train_visit = item.get("num_train_visit", None)
        use_dp_loader = item.get("use_dp_loader", False)

        generated_model = None
        train_samples = None
        val_samples = None
        test_samples = None
        train_dataset_gen = None
        val_dataset_gen = None
        test_dataset_gen = None
        train_dataloader = None
        val_dataloader = None
        test_dataloader = None

        print(f"\n===== generator={gen_tag} : overwrite processed/*.pkl =====")

        try:
            # 1) 加载当前生成模型
            diff_cfg = copy.deepcopy(diff_config)
            diff_cfg["nnet_path"] = nnet_path
            diff_cfg["num_train_visit"] = (
                num_train_visit if num_train_visit is not None else len(train_dataset)
            )

            print(
                f"[Load diffusion] gen_tag={gen_tag}, "
                f"gen_subdir={gen_subdir}, "
                f"path_txt={path_txt}, "
                f"nnet_path={diff_cfg['nnet_path']}, "
                f"num_train_visit={diff_cfg['num_train_visit']}, "
                f"use_dp_loader={use_dp_loader}"
            )

            if use_dp_loader:
                generated_model = load_diffusion_dp(diff_cfg, tokenizers)
            else:
                generated_model = load_diffusion(diff_cfg, tokenizers)

            # 2) 清理旧 samples，并重新生成，覆盖 fixed processed 路径
            cleanup_processed_pickles(processed_root)

            train_samples, val_samples = augment_dataset(
                generated_model,
                diff_cfg,
                config,
                [train_dataset, val_dataset],
                tokenizers,
                root_to=processed_root,
            )
            test_samples = augment_dataset1(
                generated_model,
                diff_cfg,
                config,
                [train_dataset, test_dataset],
                tokenizers,
                root_to=processed_root,
            )

            save_pickle(train_samples, os.path.join(processed_root, "train_samples.pkl"))
            save_pickle(val_samples, os.path.join(processed_root, "val_samples.pkl"))
            save_pickle(test_samples, os.path.join(processed_root, "test_samples.pkl"))

            print(f"[Saved] train/val/test samples overwritten in: {processed_root}")

            # 3) 跑 downstream
            train_dataset_gen = convert_dataset(
                train_samples,
                dataset_name=config["DATASET"],
                task_name=config["TASK"],
                valid=False,
            )
            val_dataset_gen = convert_dataset(
                val_samples,
                dataset_name=config["DATASET"],
                task_name=config["TASK"],
                valid=False,
            )
            test_dataset_gen = convert_dataset(
                test_samples,
                dataset_name=config["DATASET"],
                task_name=config["TASK"],
                valid=False,
            )

            collate_fn = get_special_input(config)
            train_dataloader = get_dataloader(
                train_dataset_gen,
                batch_size=config["BATCH"],
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn,
            )
            val_dataloader = get_dataloader(
                val_dataset_gen,
                batch_size=config["BATCH"],
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )
            test_dataloader = get_dataloader(
                test_dataset_gen,
                batch_size=config["BATCH"],
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )

            exp_num_gen = f"{exp_num}_{gen_tag}"

            downstream_metrics = run_pcf(
                sample_dataset,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                config,
                exp_num=exp_num_gen,
                existing_model=None,
                mode=mode,
                aux_data={"p_grouped": p_grouped},
            )

            downstream_ckpt = (
                Path(mmhealth_root)
                / "log"
                / "ckpt"
                / config["TASK"]
                / f"{config['DATASET']}-{config['MODEL']}-{exp_num_gen}"
                / "best.ckpt"
            )

            if not downstream_ckpt.exists():
                raise FileNotFoundError(f"Downstream ckpt not found: {downstream_ckpt}")

            output_dir = Path(mmhealth_root) / "src/outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_json = output_dir / (
                f"downstream_task_leakage_"
                f"{config['TASK'].lower()}_"
                f"{config['DATASET'].lower()}_"
                f"{config['MODEL']}_"
                f"{gen_tag}_synthetic_only.json"
            )

            audit_workdir = os.path.join(mmhealth_root, "src")
            env = os.environ.copy()
            old_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = audit_workdir if not old_pythonpath else audit_workdir + os.pathsep + old_pythonpath

            cmd = [
                "python", "-m", "audit.downstream_task_leakage",
                "--mmhealth-root", mmhealth_root,
                "--task", config["TASK"],
                "--dataset", config["DATASET"],
                "--model", config["MODEL"],
                "--exp-num", exp_num_gen,
                "--ckpt", str(downstream_ckpt),
                "--feature-source", "synthetic_only",
                "--agg", "mean",
                "--fpr-targets", "0.001,0.005,0.01,0.05",
                "--bootstrap-iters", "1000",
                "--bootstrap-confidence", "0.95",
                "--bootstrap-seed", "528",
                "--subgroup-analysis",
                "--min-subgroup-patients", "30",
                "--output", str(output_json),
            ]
            subprocess.run(cmd, check=True, cwd=audit_workdir, env=env)

            with open(output_json, "r", encoding="utf-8") as f:
                audit_result = json.load(f)

            mia_auc = audit_result.get("downstream_patient_mia", {}).get("auc", float("nan"))

            append_csv_row(
                summary_csv,
                {
                    "status": "success",
                    "gen_tag": gen_tag,
                    "gen_root": gen_root,
                    "path_txt": path_txt,
                    "nnet_path": nnet_path,
                    "processed_root": processed_root,
                    "exp_num_gen": exp_num_gen,

                    "jaccard_samples": downstream_metrics.get("jaccard_samples", float("nan")) if isinstance(
                        downstream_metrics, dict) else float("nan"),
                    "f1_samples": downstream_metrics.get("f1_samples", float("nan")) if isinstance(downstream_metrics,
                                                                                                   dict) else float(
                        "nan"),
                    "pr_auc_samples": downstream_metrics.get("pr_auc_samples", float("nan")) if isinstance(
                        downstream_metrics, dict) else float("nan"),
                    "roc_auc_samples": downstream_metrics.get("roc_auc_samples", float("nan")) if isinstance(
                        downstream_metrics, dict) else float("nan"),
                    "precision_samples": downstream_metrics.get("precision_samples", float("nan")) if isinstance(
                        downstream_metrics, dict) else float("nan"),
                    "recall_samples": downstream_metrics.get("recall_samples", float("nan")) if isinstance(
                        downstream_metrics, dict) else float("nan"),
                    "loss": downstream_metrics.get("loss", float("nan")) if isinstance(downstream_metrics,
                                                                                       dict) else float("nan"),

                    "downstream_mia_auc": mia_auc,
                    "audit_json": str(output_json),
                    "downstream_metrics_json": str(downstream_metrics_json),
                },
            )

            print(f"[Done] {gen_tag} finished.")

        except Exception as e:
            print(f"[Error] generator={gen_tag} failed: {e}")

            append_csv_row(
                summary_csv,
                {
                    "status": "failed",
                    "gen_tag": gen_tag,
                    "gen_root": gen_root,
                    "path_txt": path_txt if 'path_txt' in locals() else "",
                    "nnet_path": nnet_path if 'nnet_path' in locals() else "",
                    "processed_root": processed_root,
                    "exp_num_gen": f"{exp_num}_{gen_tag}",
                    "downstream_auc": float("nan"),
                    "downstream_mia_auc": float("nan"),
                    "audit_json": "",
                    "error": str(e),
                },
            )

        finally:
            del generated_model
            del train_samples, val_samples, test_samples
            del train_dataset_gen, val_dataset_gen, test_dataset_gen
            del train_dataloader, val_dataloader, test_dataloader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PHE")
    parser.add_argument("--dataset", type=str, default="MIII")
    parser.add_argument("--model", type=str, default="ours")
    parser.add_argument("--exp-num", type=str, default="20")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--mmhealth-root",
        type=str,
        default="/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/",
    )
    parser.add_argument(
        "--difflog-root",
        type=str,
        default="",
        help="生成模型日志根目录，默认自动拼成 {mmhealth-root}/log/difflog",
    )
    parser.add_argument(
        "--gen-subdirs",
        nargs="+",
        default=DEFAULT_GEN_SUBDIRS,
        help="要遍历的生成模型子目录列表",
    )
    args = parser.parse_args()

    set_random_seed(config["SEED"])

    config["TASK"] = args.task
    config["DATASET"] = args.dataset
    config["MODEL"] = args.model
    config["GPU"] = args.gpu

    if str(args.gpu).lower() == "cpu":
        diff_config["device"] = "cpu"
    else:
        diff_config["device"] = str(args.gpu).split(",")[0].strip()

    mmhealth_root = args.mmhealth_root
    difflog_root = (
        args.difflog_root
        if args.difflog_root
        else os.path.join(mmhealth_root, "log", "difflog")
    )

    root_to = os.path.join(
        mmhealth_root,
        "data",
        config["TASK"],
        config["DATASET"],
        "processed",
    )

    train_dataset = load_pickle(os.path.join(root_to, "train_dataset.pkl"))
    val_dataset = load_pickle(os.path.join(root_to, "val_dataset.pkl"))
    test_dataset = load_pickle(os.path.join(root_to, "test_dataset.pkl"))
    base_samples = load_pickle(os.path.join(root_to, "datasets_pre_stand.pkl"))
    p_grouped = load_pickle(os.path.join(root_to, "group_patient.pkl"))

    sample_dataset = convert_dataset(
        base_samples,
        dataset_name=config["DATASET"],
        task_name=config["TASK"],
        valid=False,
        all=True,
    )

    tokenizers = get_tokenizers(sample_dataset)
    task_fn, mode = get_task_fn(config)
    _ = task_fn

    run_all_generators_overwrite_processed(
        diff_config=diff_config,
        config=config,
        sample_dataset=sample_dataset,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        tokenizers=tokenizers,
        p_grouped=p_grouped,
        mode=mode,
        root_to=root_to,
        difflog_root=difflog_root,
        gen_subdirs=args.gen_subdirs,
        exp_num=args.exp_num,
        mmhealth_root=mmhealth_root,
    )


if __name__ == "__main__":
    main()