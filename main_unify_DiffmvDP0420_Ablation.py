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
# from train_dffs import run_diffusion
# from train_dffs_DiffmvDP04 import run_diffusion_DP
# from train_dffs_DiffmvDP0420_Ablation import run_diffusion_DP
from train_dffs_DiffmvDP0420_Ablation2 import run_diffusion_DP
# from eval_dffs import augment_dataset, load_diffusion, evaluate_diff
from eval_dffs import augment_dataset,augment_dataset1, load_diffusion, load_diffusion_dp, evaluate_diff, eval_diffusion_latent_mae_rmse, eval_diffusion_latent_mae_rmse20, eval_diffusion_latent_mae_rmse20all, eval_diffusion_latent_mae_rmse_allviews, plot_latent_sq_error_density, plot_sq_error_hist_for_scan
from train_hp import run_pcf, run_pretrain, load_pretrain_model, get_task_fn, add_imp_fields
from baseline_utils import missing_statistics
from data import generate_patient_group
set_random_seed(config['SEED'])
# from shot_parse import SHOTDataset

import copy
import numpy as np
import csv
import os, time, json
import pandas as pd

import random
seed = 528
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

############################# ALL Views ###############################
FEATURE_ORDER = ('conditions', 'procedures', 'drugs')  # 与三视图顺序一致

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



def run_multi_eps_generation_only(train_dataset, val_dataset, tokenizers, exp_num='20'):
    import copy
    import gc
    import torch
    import pandas as pd
    import os

    epsilon_list = [0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 4.0, 8.0, 16.0]
    rows = []

    for eps in epsilon_list:
        print(f"\n========== Start epsilon = {eps} ==========")

        diff_cfg = copy.deepcopy(diff_config)
        diff_cfg['train']['dp_sgd'] = True
        diff_cfg['train']['dp_target_epsilon'] = float(eps)
        diff_cfg['train']['resume'] = False

        eps_tag = f"eps{str(eps).replace('.', 'p')}"

        out = run_diffusion_DP(
            diff_cfg,
            config,
            [train_dataset, val_dataset],
            tokenizers,
            exp_num=f"{exp_num}_{eps_tag}"
        )

        # 兼容你当前 return 是 tuple 的情况
        if isinstance(out, tuple):
            best_model_path, generated_model, num_train_visit = out
            rows.append({
                "target_epsilon": float(eps),
                "best_model_path": best_model_path,
                "num_train_visit": num_train_visit,
            })
            del generated_model
        else:
            rows.append(out)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"========== Finish epsilon = {eps} ==========\n")

    save_csv = os.path.join(
        config['WORKDIR'] if 'WORKDIR' in config else './',
        f"multi_eps_index_{config['DATASET']}_{config['TASK']}.csv"
    )
    pd.DataFrame(rows).to_csv(save_csv, index=False)
    print("Saved multi-epsilon index to:", save_csv)


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
    # if augment:
    #     diff_config['device'] = int(config['GPU'])
    #     if config['DATASET'] == "MIV-Note":
    #         diff_config['nnet']['num_contx_token'] = 9
    #
    #     tokenizers = get_tokenizers(sample_dataset)
    #
    #     epsilon_list = [0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 4.0, 8.0, 16.0]
    #
    #     for eps in epsilon_list:
    #         diff_cfg = copy.deepcopy(diff_config)
    #         diff_cfg['train']['dp_sgd'] = True
    #         diff_cfg['train']['dp_target_epsilon'] = float(eps)
    #         diff_cfg['train']['resume'] = False
    #
    #         eps_tag = f"eps{str(eps).replace('.', 'p')}"
    #         print(f"Start training epsilon = {eps}")
    #
    #         best_model_path, generated_model, num_train_visit = run_diffusion_DP(
    #             diff_cfg,
    #             config,
    #             [train_dataset, val_dataset],
    #             tokenizers,
    #             exp_num=f"{exp_num}_{eps_tag}"
    #         )
    #         del generated_model
    #         gc.collect()
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()
    #     return
    if augment:
        diff_config['device'] = int(config['GPU'])
        if config['DATASET'] == "MIV-Note": # 要改train_diff, eval_diff
            # diff_config['nnet']['num_contx_token'] = 10
            diff_config['nnet']['num_contx_token'] = 9
        tokenizers = get_tokenizers(sample_dataset)
        if retrain or not os.path.exists(diff_config['ckpt_root']+ config['DATASET']):
            print('start retrain diffusion model!')
            print('Sample data of 1 patient, input for diffusion dataset ', train_dataset[0])
            # best_model_path, generated_model, num_train_visit = run_diffusion(diff_config, config, [train_dataset, test_dataset], tokenizers, exp_num=exp_num) # 在val上测试防止泄漏
            # best_model_path, generated_model, num_train_visit = run_diffusion(diff_config, config, [train_dataset, val_dataset], tokenizers, exp_num=exp_num) # 在val上测试防止泄漏

            best_model_path, generated_model, num_train_visit = run_diffusion_DP(diff_config, config, [train_dataset, val_dataset], tokenizers, exp_num=exp_num)  # 在val上测试防止泄漏
            with open(diff_config['ckpt_root'] + config['DATASET'] + '/' + config['TASK'] + "/best_model_path.txt", "w") as file:
                print(best_model_path, file=file)
                print(num_train_visit, file=file)

            # ##########
            # # epsilon_list = [0.01, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 4.0, 8.0, 16.0, 500, 1000, 3000, 5000, 10000]
            # epsilon_list = [1000, 3000, 5000, 10000]
            # # epsilon_list = [8.0]
            # for eps in epsilon_list:
            #     diff_cfg = copy.deepcopy(diff_config)
            #     diff_cfg['train']['dp_sgd'] = True
            #     diff_cfg['train']['dp_target_epsilon'] = float(eps)
            #     diff_cfg['train']['resume'] = False
            #     eps_tag = f"eps{str(eps).replace('.', 'p')}"
            #     print(f"Start training epsilon = {eps}")
            #     best_model_path, generated_model, num_train_visit = run_diffusion_DP(
            #         diff_cfg,
            #         config,
            #         [train_dataset, val_dataset],
            #         tokenizers,
            #         exp_num=f"{exp_num}_{eps_tag}"
            #     )
            #     info_path = os.path.join(
            #         diff_cfg['ckpt_root'],
            #         config['DATASET'],
            #         config['TASK'],
            #         f"best_model_path_{eps_tag}.txt"
            #     )
            #     os.makedirs(os.path.dirname(info_path), exist_ok=True)
            #     with open(info_path, "w") as file:
            #         print(best_model_path, file=file)
            #         print(num_train_visit, file=file)
            #     del generated_model
            #     gc.collect()
            #     if torch.cuda.is_available():
            #         torch.cuda.empty_cache()
            #     # assert False
            # print('all epsilon diffusion training done!')
            # return
            # ##################

            diff_config["nnet_path"] = best_model_path
            diff_config["num_train_visit"] = num_train_visit

            train_samples, val_samples = augment_dataset(generated_model,diff_config, config,  [train_dataset, val_dataset], tokenizers, root_to=root_to)
            test_samples = augment_dataset1(generated_model, diff_config, config, [train_dataset, test_dataset], tokenizers, root_to=root_to)
            # train_samples, test = augment_dataset(generated_model, diff_config, config, [train_dataset, test_dataset], tokenizers, root_to=root_to)
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
                # generated_model = load_diffusion(diff_config, tokenizers)
                generated_model = load_diffusion_dp(diff_config, tokenizers)

                print('Load diffusion model done!')
                train_samples, val_samples = augment_dataset(generated_model,diff_config, config,  [train_dataset, val_dataset], tokenizers, root_to=root_to)
                test_samples = augment_dataset1(generated_model, diff_config, config, [train_dataset, test_dataset], tokenizers, root_to=root_to)
                # train_samples, test_samples = augment_dataset(generated_model,diff_config, config,  [train_dataset, test_dataset], tokenizers, root_to=root_to)

            # else:
            #     print("Have been augmented!")
            #     # 1) 先读取之前保存的 best_model_path 和 num_train_visit
            #     best_info_path = diff_config['ckpt_root'] + config['DATASET'] + '/' + config[
            #         'TASK'] + "/best_model_path.txt"
            #     with open(best_info_path, "r") as f:
            #         lines = f.readlines()
            #     best_model_path = lines[0].strip()
            #     num_train_visit = int(lines[1].strip())
            #     diff_config["nnet_path"] = best_model_path
            #     diff_config["num_train_visit"] = num_train_visit
            #     # 2) 加载已经训练好的 diffusion 模型（不会重新训练）
            #     print("Load diffusion model for latent evaluation...")
            #     generated_model = load_diffusion(diff_config, tokenizers)
            #     print("Load diffusion model done!")
            #     # 4) 评估完再释放 val/test dataset，加载已经生成好的增强样本
            #     del val_dataset, test_dataset
            #     train_samples = load_pickle(root_to + 'train_samples.pkl')
            #     test_samples = load_pickle(root_to + 'test_samples.pkl')

            else:
                print("Have been augmented!")
                del train_dataset, val_dataset, test_dataset
                train_samples = load_pickle(root_to + 'train_samples.pkl')
                val_samples = load_pickle(root_to + 'val_samples.pkl')
                test_samples = load_pickle(root_to + 'test_samples.pkl')

        train_dataset = convert_dataset(train_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
        val_dataset = convert_dataset(val_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
        test_dataset = convert_dataset(test_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
        print("Augment Done!")


    collate_fn = get_special_input(config)
    # missing_statistics(train_dataset, config)
    train_dataloader = get_dataloader(train_dataset, batch_size=config['BATCH'], shuffle=True, drop_last=True, collate_fn=collate_fn)
    # val_dataloader = get_dataloader(val_dataset, batch_size=config['BATCH']*5, shuffle=False, drop_last=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=config['BATCH'], shuffle=False, drop_last=False, collate_fn=collate_fn)
    test_dataloader = get_dataloader(test_dataset, batch_size=config['BATCH'], shuffle=False, drop_last=False, collate_fn=collate_fn)
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
        run_pretrain(sample_dataset, config, exp_num=exp_num, special_input={'train': train_dataset, 'test': test_dataset})
        end_time = time.time()

        model = load_pretrain_model(sample_dataset, config, special_input=None, exp_num=exp_num, mode=mode)
        print("Pretrain done! Cost {} s".format(end_time - start))
        existing_model = model

    start = time.time()
    set_random_seed(config['SEED'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run_pcf(sample_dataset, train_dataloader, val_dataloader, test_dataloader, config, exp_num=exp_num, existing_model=existing_model, mode=mode, aux_data={'p_grouped': p_grouped})
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
