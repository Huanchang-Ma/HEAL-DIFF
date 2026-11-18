import os
import gc
import json
import copy
import argparse
import subprocess
import pandas as pd
import torch

from config import config
from diffuser.diff_config import config as diff_config
from eval_dffs import augment_dataset, augment_dataset1, load_diffusion_dp
from train_hp import run_pcf, get_task_fn
from loader import get_dataloader, get_special_input
from data import convert_dataset
from utils import load_pickle, save_pickle, get_tokenizers, set_random_seed

def extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "nnet", "network"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def strip_prefix_if_needed(state_dict):
    # 兼容 DataParallel / 包装前缀
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ["module.", "model.", "nnet."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        new_sd[nk] = v
    return new_sd


def infer_num_train_visit_from_ckpt(nnet_path):
    ckpt = torch.load(nnet_path, map_location="cpu")
    state_dict = extract_state_dict(ckpt)
    state_dict = strip_prefix_if_needed(state_dict)

    if "total_visit_emb" not in state_dict:
        raise KeyError(
            f"'total_visit_emb' not found in checkpoint: {nnet_path}"
        )

    return int(state_dict["total_visit_emb"].shape[0])

def append_csv_row(csv_path, row_dict):
    row_df = pd.DataFrame([row_dict])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        new_df = pd.concat([old_df, row_df], ignore_index=True)
    else:
        new_df = row_df
    new_df.to_csv(csv_path, index=False)


def run_all_eps_overwrite_processed(
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
    exp_num="20",
    mmhealth_root="/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/",
):
    generation_csv = os.path.join(
        diff_config["workdir"],
        config["DATASET"],
        config["TASK"],
        "generation_dp_summary.csv",
    )
    gen_df = pd.read_csv(generation_csv).sort_values("actual_epsilon")

    summary_csv = os.path.join(
        diff_config["workdir"],
        config["DATASET"],
        config["TASK"],
        "downstream_dp_summary.csv",
    )

    processed_root = root_to  # 固定路径，循环覆盖

    for _, row in gen_df.iterrows():
        target_epsilon = float(row["target_epsilon"])
        actual_epsilon = float(row["actual_epsilon"])
        eps_tag = f"eps{str(target_epsilon).replace('.', 'p')}"
        print(f"\n===== epsilon={target_epsilon} : overwrite processed/*.pkl =====")
        # 1) 读取当前 epsilon 的 diffusion 模型
        diff_cfg = copy.deepcopy(diff_config)

        diff_cfg["nnet_path"] = row["nnet_path"]
        csv_num_train_visit = None
        if "num_train_visit" in gen_df.columns and pd.notna(row["num_train_visit"]):
            csv_num_train_visit = int(row["num_train_visit"])
        ckpt_num_train_visit = infer_num_train_visit_from_ckpt(diff_cfg["nnet_path"])
        current_train_len = len(train_dataset)
        diff_cfg["num_train_visit"] = ckpt_num_train_visit
        print(
            f"[Load diffusion] eps={eps_tag}, "
            f"nnet_path={diff_cfg['nnet_path']}, "
            f"csv_num_train_visit={csv_num_train_visit}, "
            f"current_train_len={current_train_len}, "
            f"ckpt_num_train_visit={ckpt_num_train_visit}, "
            f"final_num_train_visit={diff_cfg['num_train_visit']}"
        )

        generated_model = load_diffusion_dp(diff_cfg, tokenizers)

        # 2) 重新生成 samples.pkl，覆盖固定 processed 路径
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

        # 3) 跑 downstream
        train_dataset_eps = convert_dataset(
            train_samples,
            dataset_name=config["DATASET"],
            task_name=config["TASK"],
            valid=False,
        )
        val_dataset_eps = convert_dataset(
            val_samples,
            dataset_name=config["DATASET"],
            task_name=config["TASK"],
            valid=False,
        )
        test_dataset_eps = convert_dataset(
            test_samples,
            dataset_name=config["DATASET"],
            task_name=config["TASK"],
            valid=False,
        )

        collate_fn = get_special_input(config)
        train_dataloader = get_dataloader(
            train_dataset_eps,
            batch_size=config["BATCH"],
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        val_dataloader = get_dataloader(
            val_dataset_eps,
            batch_size=config["BATCH"],
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
        test_dataloader = get_dataloader(
            test_dataset_eps,
            batch_size=config["BATCH"],
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        exp_num_eps = f"{exp_num}_{eps_tag}"

        downstream_metrics = run_pcf(
            sample_dataset,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            config,
            exp_num=exp_num_eps,
            existing_model=None,
            mode=mode,
            aux_data={"p_grouped": p_grouped},
        )

        from pathlib import Path
        downstream_ckpt = Path(mmhealth_root) / "log" / "ckpt" / config[
            "TASK"] / f"{config['DATASET']}-{config['MODEL']}-{exp_num_eps}" / "best.ckpt"

        if not downstream_ckpt.exists():
            raise FileNotFoundError(f"Downstream ckpt not found: {downstream_ckpt}")

        output_json = os.path.join(
            "outputs",
            f"downstream_task_leakage_{config['TASK'].lower()}_{config['DATASET'].lower()}_{config['MODEL']}_{eps_tag}.json",
        )
        # output_json = os.path.join(
        #     "outputs",
        #     f"downstream_task_leakage_{config['TASK'].lower()}_{config['DATASET'].lower()}_{config['MODEL']}_{eps_tag}_mixed.json",
        # )

        cmd = [
            "python", "-m", "audit.downstream_task_leakage",
            "--mmhealth-root", mmhealth_root,
            "--task", config["TASK"],
            "--dataset", config["DATASET"],
            "--model", config["MODEL"],
            "--exp-num", exp_num_eps,
            "--ckpt", str(downstream_ckpt),
            "--feature-source", "synthetic_only",
            "--agg", "mean",
            "--fpr-targets", "0.001,0.005,0.01,0.05",
            "--bootstrap-iters", "1000",
            "--bootstrap-confidence", "0.95",
            "--bootstrap-seed", "528",
            "--subgroup-analysis",
            "--min-subgroup-patients", "30",
            "--output", output_json,
        ]
        # cmd = [
        #     "python", "-m", "audit.downstream_task_leakage",
        #     "--mmhealth-root", mmhealth_root,
        #     "--task", config["TASK"],
        #     "--dataset", config["DATASET"],
        #     "--model", config["MODEL"],
        #     "--exp-num", exp_num_eps,
        #     "--ckpt", str(downstream_ckpt),
        #     # "--feature-source", "mixed",
        #     "--agg", "mean",
        #     "--fpr-targets", "0.001,0.005,0.01,0.05",
        #     "--bootstrap-iters", "1000",
        #     "--bootstrap-confidence", "0.95",
        #     "--bootstrap-seed", "528",
        #     "--subgroup-analysis",
        #     "--min-subgroup-patients", "30",
        #     "--output", output_json,
        # ]
        subprocess.run(cmd, check=True)

        with open(output_json, "r") as f:
            audit_result = json.load(f)

        mia_auc = audit_result.get("downstream_patient_mia", {}).get("auc", float("nan"))

        append_csv_row(
            summary_csv,
            {
                "target_epsilon": target_epsilon,
                "actual_epsilon": actual_epsilon,
                "eps_tag": eps_tag,
                "processed_root": processed_root,
                "downstream_auc": downstream_metrics.get("auc", float("nan")) if isinstance(downstream_metrics, dict) else float("nan"),
                "downstream_mia_auc": mia_auc,
                "audit_json": output_json,
            },
        )

        del generated_model
        del train_samples, val_samples, test_samples
        del train_dataset_eps, val_dataset_eps, test_dataset_eps
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
    parser.add_argument("--mmhealth-root", type=str, default="/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/")
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

    root_to = os.path.join(
        args.mmhealth_root,
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

    run_all_eps_overwrite_processed(
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
        exp_num=args.exp_num,
        mmhealth_root=args.mmhealth_root,
    )


if __name__ == "__main__":
    main()


# ## 2603
# import os
# import gc
# import json
# import copy
# import argparse
# import subprocess
# import pandas as pd
# import torch
#
# from config import config
# from diffuser.diff_config import config as diff_config
# from eval_dffs import augment_dataset, augment_dataset1, load_diffusion_dp
# from train_hp import run_pcf, get_task_fn
# from loader import get_dataloader, get_special_input
# from data import convert_dataset
# from utils import load_pickle, save_pickle, get_tokenizers, set_random_seed
#
#
# def append_csv_row(csv_path, row_dict):
#     row_df = pd.DataFrame([row_dict])
#     os.makedirs(os.path.dirname(csv_path), exist_ok=True)
#     if os.path.exists(csv_path):
#         old_df = pd.read_csv(csv_path)
#         new_df = pd.concat([old_df, row_df], ignore_index=True)
#     else:
#         new_df = row_df
#     new_df.to_csv(csv_path, index=False)
#
#
# def run_all_eps_overwrite_processed(
#     diff_config,
#     config,
#     sample_dataset,
#     train_dataset,
#     val_dataset,
#     test_dataset,
#     tokenizers,
#     p_grouped,
#     mode,
#     root_to,
#     exp_num="20",
#     mmhealth_root="/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/",
# ):
#     generation_csv = os.path.join(
#         diff_config["workdir"],
#         config["DATASET"],
#         config["TASK"],
#         "generation_dp_summary.csv",
#     )
#     gen_df = pd.read_csv(generation_csv).sort_values("actual_epsilon")
#
#     summary_csv = os.path.join(
#         diff_config["workdir"],
#         config["DATASET"],
#         config["TASK"],
#         "downstream_dp_summary.csv",
#     )
#
#     processed_root = root_to  # 固定路径，循环覆盖
#
#     for _, row in gen_df.iterrows():
#         target_epsilon = float(row["target_epsilon"])
#         actual_epsilon = float(row["actual_epsilon"])
#         eps_tag = f"eps{str(target_epsilon).replace('.', 'p')}"
#         print(f"\n===== epsilon={target_epsilon} : overwrite processed/*.pkl =====")
#         # 1) 读取当前 epsilon 的 diffusion 模型
#         diff_cfg = copy.deepcopy(diff_config)
#         diff_cfg["nnet_path"] = row["nnet_path"]
#         # 优先从 generation_dp_summary.csv 里读；没有就回退到当前 train_dataset 的长度
#         if "num_train_visit" in gen_df.columns and pd.notna(row["num_train_visit"]):
#             diff_cfg["num_train_visit"] = int(row["num_train_visit"])
#         else:
#             diff_cfg["num_train_visit"] = len(train_dataset)
#         print(
#             f"[Load diffusion] eps={eps_tag}, nnet_path={diff_cfg['nnet_path']}, num_train_visit={diff_cfg['num_train_visit']}")
#
#         generated_model = load_diffusion_dp(diff_cfg, tokenizers)
#
#         # 2) 重新生成 samples.pkl，覆盖固定 processed 路径
#         train_samples, val_samples = augment_dataset(
#             generated_model,
#             diff_cfg,
#             config,
#             [train_dataset, val_dataset],
#             tokenizers,
#             root_to=processed_root,
#         )
#         test_samples = augment_dataset1(
#             generated_model,
#             diff_cfg,
#             config,
#             [train_dataset, test_dataset],
#             tokenizers,
#             root_to=processed_root,
#         )
#
#         save_pickle(train_samples, os.path.join(processed_root, "train_samples.pkl"))
#         save_pickle(val_samples, os.path.join(processed_root, "val_samples.pkl"))
#         save_pickle(test_samples, os.path.join(processed_root, "test_samples.pkl"))
#
#         # 3) 跑 downstream
#         train_dataset_eps = convert_dataset(
#             train_samples,
#             dataset_name=config["DATASET"],
#             task_name=config["TASK"],
#             valid=False,
#         )
#         val_dataset_eps = convert_dataset(
#             val_samples,
#             dataset_name=config["DATASET"],
#             task_name=config["TASK"],
#             valid=False,
#         )
#         test_dataset_eps = convert_dataset(
#             test_samples,
#             dataset_name=config["DATASET"],
#             task_name=config["TASK"],
#             valid=False,
#         )
#
#         collate_fn = get_special_input(config)
#         train_dataloader = get_dataloader(
#             train_dataset_eps,
#             batch_size=config["BATCH"],
#             shuffle=True,
#             drop_last=True,
#             collate_fn=collate_fn,
#         )
#         val_dataloader = get_dataloader(
#             val_dataset_eps,
#             batch_size=config["BATCH"],
#             shuffle=False,
#             drop_last=False,
#             collate_fn=collate_fn,
#         )
#         test_dataloader = get_dataloader(
#             test_dataset_eps,
#             batch_size=config["BATCH"],
#             shuffle=False,
#             drop_last=False,
#             collate_fn=collate_fn,
#         )
#
#         exp_num_eps = f"{exp_num}_{eps_tag}"
#
#         downstream_metrics = run_pcf(
#             sample_dataset,
#             train_dataloader,
#             val_dataloader,
#             test_dataloader,
#             config,
#             exp_num=exp_num_eps,
#             existing_model=None,
#             mode=mode,
#             aux_data={"p_grouped": p_grouped},
#         )
#
#         from pathlib import Path
#         downstream_ckpt = Path(mmhealth_root) / "log" / "ckpt" / config[
#             "TASK"] / f"{config['DATASET']}-{config['MODEL']}-{exp_num_eps}" / "best.ckpt"
#
#         if not downstream_ckpt.exists():
#             raise FileNotFoundError(f"Downstream ckpt not found: {downstream_ckpt}")
#
#         # output_json = os.path.join(
#         #     "outputs",
#         #     f"downstream_task_leakage_{config['TASK'].lower()}_{config['DATASET'].lower()}_{config['MODEL']}_{eps_tag}.json",
#         # )
#         output_json = os.path.join(
#             "outputs",
#             f"downstream_task_leakage_{config['TASK'].lower()}_{config['DATASET'].lower()}_{config['MODEL']}_{eps_tag}_mixed.json",
#         )
#
#         cmd = [
#             "python", "-m", "audit.downstream_task_leakage",
#             "--mmhealth-root", mmhealth_root,
#             "--task", config["TASK"],
#             "--dataset", config["DATASET"],
#             "--model", config["MODEL"],
#             "--exp-num", exp_num_eps,
#             "--ckpt", str(downstream_ckpt),
#             "--feature-source", "synthetic_only",
#             "--agg", "mean",
#             "--fpr-targets", "0.001,0.005,0.01,0.05",
#             "--bootstrap-iters", "1000",
#             "--bootstrap-confidence", "0.95",
#             "--bootstrap-seed", "528",
#             "--subgroup-analysis",
#             "--min-subgroup-patients", "30",
#             "--output", output_json,
#         ]
#         # cmd = [
#         #     "python", "-m", "audit.downstream_task_leakage",
#         #     "--mmhealth-root", mmhealth_root,
#         #     "--task", config["TASK"],
#         #     "--dataset", config["DATASET"],
#         #     "--model", config["MODEL"],
#         #     "--exp-num", exp_num_eps,
#         #     "--ckpt", str(downstream_ckpt),
#         #     # "--feature-source", "mixed",
#         #     "--agg", "mean",
#         #     "--fpr-targets", "0.001,0.005,0.01,0.05",
#         #     "--bootstrap-iters", "1000",
#         #     "--bootstrap-confidence", "0.95",
#         #     "--bootstrap-seed", "528",
#         #     "--subgroup-analysis",
#         #     "--min-subgroup-patients", "30",
#         #     "--output", output_json,
#         # ]
#         subprocess.run(cmd, check=True)
#
#         with open(output_json, "r") as f:
#             audit_result = json.load(f)
#
#         mia_auc = audit_result.get("downstream_patient_mia", {}).get("auc", float("nan"))
#
#         append_csv_row(
#             summary_csv,
#             {
#                 "target_epsilon": target_epsilon,
#                 "actual_epsilon": actual_epsilon,
#                 "eps_tag": eps_tag,
#                 "processed_root": processed_root,
#                 "downstream_auc": downstream_metrics.get("auc", float("nan")) if isinstance(downstream_metrics, dict) else float("nan"),
#                 "downstream_mia_auc": mia_auc,
#                 "audit_json": output_json,
#             },
#         )
#
#         del generated_model
#         del train_samples, val_samples, test_samples
#         del train_dataset_eps, val_dataset_eps, test_dataset_eps
#         del train_dataloader, val_dataloader, test_dataloader
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--task", type=str, default="PHE")
#     parser.add_argument("--dataset", type=str, default="MIII")
#     parser.add_argument("--model", type=str, default="ours")
#     parser.add_argument("--exp-num", type=str, default="20")
#     parser.add_argument("--gpu", type=str, default="0")
#     parser.add_argument("--mmhealth-root", type=str, default="/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/")
#     args = parser.parse_args()
#
#     set_random_seed(config["SEED"])
#
#     config["TASK"] = args.task
#     config["DATASET"] = args.dataset
#     config["MODEL"] = args.model
#     config["GPU"] = args.gpu
#     if str(args.gpu).lower() == "cpu":
#         diff_config["device"] = "cpu"
#     else:
#         diff_config["device"] = str(args.gpu).split(",")[0].strip()
#
#     root_to = os.path.join(
#         args.mmhealth_root,
#         "data",
#         config["TASK"],
#         config["DATASET"],
#         "processed",
#     )
#
#     train_dataset = load_pickle(os.path.join(root_to, "train_dataset.pkl"))
#     val_dataset = load_pickle(os.path.join(root_to, "val_dataset.pkl"))
#     test_dataset = load_pickle(os.path.join(root_to, "test_dataset.pkl"))
#     base_samples = load_pickle(os.path.join(root_to, "datasets_pre_stand.pkl"))
#     p_grouped = load_pickle(os.path.join(root_to, "group_patient.pkl"))
#
#     sample_dataset = convert_dataset(
#         base_samples,
#         dataset_name=config["DATASET"],
#         task_name=config["TASK"],
#         valid=False,
#         all=True,
#     )
#
#     tokenizers = get_tokenizers(sample_dataset)
#     task_fn, mode = get_task_fn(config)
#     _ = task_fn
#
#     run_all_eps_overwrite_processed(
#         diff_config=diff_config,
#         config=config,
#         sample_dataset=sample_dataset,
#         train_dataset=train_dataset,
#         val_dataset=val_dataset,
#         test_dataset=test_dataset,
#         tokenizers=tokenizers,
#         p_grouped=p_grouped,
#         mode=mode,
#         root_to=root_to,
#         exp_num=args.exp_num,
#         mmhealth_root=args.mmhealth_root,
#     )
#
#
# if __name__ == "__main__":
#     main()