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
from train_dffs import run_diffusion
# from eval_dffs import augment_dataset, load_diffusion, evaluate_diff
from eval_dffs import augment_dataset, load_diffusion, evaluate_diff, eval_diffusion_latent_mae_rmse, eval_diffusion_latent_mae_rmse20, eval_diffusion_latent_mae_rmse20all, eval_diffusion_latent_mae_rmse_allviews, plot_latent_sq_error_density, plot_sq_error_hist_for_scan
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





def run_single_config(augment=False, retrain=False, exp_num='3', pretrain=True):
    """
    :param augment: whether augmented dataset
    :param retrain: whether retrain the diffusion model
    :param exp_num:
    :return:
    """
    # load datasets
    # STEP 1: load data
    # root_to = '/home/czhaobo/MMHealth/data/{}/{}/processed/'.format(config['TASK'], config['DATASET'])
    root_to = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/{}/{}/processed/'.format(config['TASK'], config['DATASET'])

    task_fn, mode = get_task_fn(config)
    if not os.path.exists(root_to + 'datasets_pre_stand.pkl'):
        print("Prepare dataset!")
        if config['DATASET'] == 'MIII':
            base_dataset = MIMIC3Dataset(
                # root="/home/czhaobo/HyperHealth/data/physionet.org/files/mimiciii/1.4",
                root='/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/mimic-iii-clinical-database-1.4/',
                tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
                code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": config['ATCLEVEL']}})}, # 这里graphcare的ATC-level是3；和我们在data阶段有差别
                dev=False,
                refresh_cache=False,
            )
        elif config['DATASET'] == 'eICU':
            base_dataset = eICUDataset(
                # root="/home/czhaobo/HyperHealth/data/physionet.org/files/eicu-crd/2.0",
                root='/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/eicu-collaborative-research-database-2.0/',
                tables=["diagnosis", "medication", "physicalExam", "treatment", "admissionDx"],
                dev=False,
                refresh_cache=False,
            )
        elif config['DATASET'] == 'MIV':
            base_dataset = MIMIC4Dataset(
                # root="/home/czhaobo/HyperHealth/data/physionet.org/files/mimiciv/2.0/hosp",
                root='/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/mimic-iv-3.1/hosp/',
                tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": config['ATCLEVEL']}})}, # 4
                dev=False,
                refresh_cache=False,
            )
        elif config['DATASET'] == 'MIV-Note': # 这里可以通过subject_id进行联系。因为有一部分数据被MIMIC官方雪藏了。所有用2.2以上的版本
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
                # root="/home/czhaobo/HyperHealth/data/physionet.org/files/mimiciv/2.0/hosp",
                root='/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/mimic-iv-3.1/hosp/',
                # root='/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/',
                tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": config['ATCLEVEL']}})}, # 4
                dev=False, ## PHE任务 原始状态
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
            best_model_path, generated_model, num_train_visit = run_diffusion(diff_config, config, [train_dataset, val_dataset], tokenizers, exp_num=exp_num) # 在val上测试防止泄漏
            # best_model_path, generated_model, num_train_visit = run_diffusion(diff_config, config, [train_dataset, val_dataset], tokenizers, exp_num=exp_num) # 在val上测试防止泄漏
            with open(diff_config['ckpt_root'] + config['DATASET'] + '/' + config['TASK'] + "/best_model_path.txt", "w") as file:
                print(best_model_path, file=file)
                print(num_train_visit, file=file)

            diff_config["nnet_path"] = best_model_path
            diff_config["num_train_visit"] = num_train_visit

            train_samples, test_samples = augment_dataset(generated_model,diff_config, config,  [train_dataset, val_dataset], tokenizers, root_to=root_to)
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
                # train_samples, test_samples = augment_dataset(generated_model,diff_config, config,  [train_dataset, val_dataset], tokenizers, root_to=root_to)
                train_samples, test_samples = augment_dataset(generated_model,diff_config, config,  [train_dataset, test_dataset], tokenizers, root_to=root_to)

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

                # 3) 在真正的 test_dataset 上做 latent 空间 MAE/RMSE + Jaccard 评估
                #    这里 hp_dataset 传 [train_dataset, test_dataset]，
                #    这样 get_dataset(...).get_split('test') 就对应真实 test。
                ##
                # hp_dataset_for_eval = [train_dataset, test_dataset]
                # eval_diffusion_latent_mae_rmse1(
                #     nnet=generated_model,
                #     diff_config=diff_config,
                #     config_health=config,
                #     hp_dataset=hp_dataset_for_eval,
                #     tokenizers=tokenizers,
                #     root_to=root_to,
                #     sample_steps=diff_config['sample']['sample_steps'],
                #     use_fixed_mask=False,  # 用原来的 mask 方式
                # )
                ##
                # hp_dataset_for_eval = [train_dataset, test_dataset]
                # eval_diffusion_latent_mae_rmse(
                #     nnet=generated_model,
                #     diff_config=diff_config,
                #     config_health=config,
                #     hp_dataset=hp_dataset_for_eval,
                #     tokenizers=tokenizers,
                #     root_to=root_to,
                #     sample_steps=diff_config['sample']['sample_steps'],
                #     use_fixed_mask=False,  # 用原来的 mask 方式
                # )
                ##
                # hp_dataset = [train_dataset, test_dataset]
                # cond_ratios = (0.2, 0.5, 0.7)
                # proc_ratios = (0.2, 0.5, 0.7)
                # drug_ratios = (0.2, 0.5, 0.7)
                # eval_diffusion_latent_mae_rmse20(
                #     nnet=generated_model,
                #     diff_config=diff_config,
                #     config_health=config,
                #     hp_dataset=hp_dataset,
                #     tokenizers=tokenizers,
                #     root_to=root_to,
                #     sample_steps=50,
                #     cond_ratios=cond_ratios,
                #     proc_ratios=proc_ratios,
                #     drug_ratios=drug_ratios,
                # )
                # ##
                # hp_dataset = [train_dataset, test_dataset]
                # cond_ratios = (0.2, 0.5, 0.7, 1.0)
                # proc_ratios = (0.2, 0.5, 0.7, 1.0)
                # drug_ratios = (0.2, 0.5, 0.7, 1.0)
                # eval_diffusion_latent_mae_rmse20all(
                #     nnet=generated_model,
                #     diff_config=diff_config,
                #     config_health=config,
                #     hp_dataset=hp_dataset,
                #     tokenizers=tokenizers,
                #     root_to=root_to,
                #     sample_steps=50,
                #     cond_ratios=cond_ratios,
                #     proc_ratios=proc_ratios,
                #     drug_ratios=drug_ratios,
                #     fully_mask_other_views=True,
                # )
                # assert False

                ##
                # hp_dataset = [train_dataset, test_dataset]
                # cond_ratios = (0.1, 0.2, 0.3, 0.4, 0.6, 0.7)
                # proc_ratios = (0.1, 0.2, 0.3, 0.4, 0.6, 0.7)
                # drug_ratios = (0.1, 0.2, 0.3, 0.4, 0.6, 0.7)
                # eval_diffusion_latent_mae_rmse_allviews(
                #     nnet=generated_model,
                #     diff_config=diff_config,
                #     config_health=config,
                #     hp_dataset=hp_dataset,
                #     tokenizers=tokenizers,
                #     root_to=root_to,
                #     sample_steps=50,
                #     cond_ratios=cond_ratios,
                #     proc_ratios=proc_ratios,
                #     drug_ratios=drug_ratios,
                #     fully_mask_other_views=False,
                # )
                # ####################################################################
                # hp_dataset = [train_dataset, test_dataset]
                # cond_ratios = (0.1, 0.3, 0.5, 0.7, 1.0)
                # proc_ratios = (0.1, 0.3, 0.5, 0.7, 1.0)
                # drug_ratios = (0.1, 0.3, 0.5, 0.7, 1.0)
                # # cond_ratios = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
                # # proc_ratios = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
                # # drug_ratios = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
                # (results_cond_final,
                #  results_proc_final,
                #  results_drug_final,
                #  cond_latent_abs_err,
                #  proc_latent_abs_err,
                #  drug_latent_abs_err) = eval_diffusion_latent_mae_rmse_allviews(
                #     nnet=generated_model,
                #     diff_config=diff_config,
                #     config_health=config,
                #     hp_dataset=hp_dataset,
                #     tokenizers=tokenizers,
                #     root_to=root_to,
                #     sample_steps=50,
                #     cond_ratios=cond_ratios,
                #     proc_ratios=proc_ratios,
                #     drug_ratios=drug_ratios,
                #     fully_mask_other_views=False,  # 或 True，看你想要哪种设置
                # )
                # plot_sq_error_hist_for_scan(
                #     abs_err_by_ratio=cond_latent_abs_err,
                #     title='Conditions latent absolute error (Missing Ratio)',
                #     use_prob=False,
                #     save_path="/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/src/MAE/conditions_scan_multi_ratio.png",
                # )
                # plot_sq_error_hist_for_scan(
                #     abs_err_by_ratio=proc_latent_abs_err,
                #     title='Procedures latent absolute error (Missing Ratio)',
                #     use_prob=False,
                #     save_path="/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/src/MAE/procedures_scan_multi_ratio.png",
                # )
                # # 比如画 drugs-scan 的 multi-ratio 曲线
                # plot_sq_error_hist_for_scan(
                #     abs_err_by_ratio=drug_latent_abs_err,
                #     title='Drugs latent absolute error (Missing Ratio)',
                #     use_prob=False,
                #     save_path="/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/src/MAE/drugs_scan_multi_ratio.png",
                # )
                #
                # assert False
                # ####################################################################

                # 4) 评估完再释放 val/test dataset，加载已经生成好的增强样本
                del val_dataset, test_dataset
                train_samples = load_pickle(root_to + 'train_samples.pkl')
                test_samples = load_pickle(root_to + 'test_samples.pkl')



            # else:
            #     print("Have been augmented!")
            #     del val_dataset, test_dataset
            #     train_samples = load_pickle(root_to + 'train_samples.pkl')
            #     test_samples = load_pickle(root_to + 'test_samples.pkl')

                # ## 查看train_samples train_dataset数据格式
                # print("type(train_samples):", type(train_samples))
                # try:
                #     print("len(train_samples):", len(train_samples))
                # except TypeError:
                #     print("train_samples 没有 len()，可能不是 list")
                #
                # print("=== train_samples[0] 原始打印 ===")
                # print(train_samples[0])
                #
                # # 如果是 dict，看看有哪些 key、每个 key 的类型/形状
                # if isinstance(train_samples[0], dict):
                #     print("=== train_samples[0] 字段信息 ===")
                #     for k, v in train_samples[0].items():
                #         if isinstance(v, torch.Tensor):
                #             print(f"  {k}: Tensor, shape={v.shape}, dtype={v.dtype}")
                #         else:
                #             # 尝试打印长度
                #             try:
                #                 l = len(v)
                #             except TypeError:
                #                 l = "N/A"
                #             print(f"  {k}: type={type(v)}, len={l}")
                # assert False

                # ## 查看train_samples train_dataset数据格式
                # print("type(train_dataset):", type(train_dataset))
                # try:
                #     print("len(train_dataset):", len(train_dataset))
                # except TypeError:
                #     print("train_dataset 没有 len()，可能不是 list")
                #
                # print("=== train_dataset[0] 原始打印 ===")
                # print(train_dataset[0])
                #
                # # 如果是 dict，看看有哪些 key、每个 key 的类型/形状
                # if isinstance(train_dataset[0], dict):
                #     print("=== train_dataset[0] 字段信息 ===")
                #     for k, v in train_dataset[0].items():
                #         if isinstance(v, torch.Tensor):
                #             print(f"  {k}: Tensor, shape={v.shape}, dtype={v.dtype}")
                #         else:
                #             # 尝试打印长度
                #             try:
                #                 l = len(v)
                #             except TypeError:
                #                 l = "N/A"
                #             print(f"  {k}: type={type(v)}, len={l}")
                # assert False

        # train_dataset = convert_dataset(train_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
        # test_dataset = convert_dataset(test_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)

    #     # === 先生成 imp_* 字段（缺失用new_，非缺失用原始）===
    #     train_samples = add_imp_fields(
    #         train_samples,
    #         feature_order=('conditions', 'procedures', 'drugs'),
    #         new_prefix='new_',
    #         out_prefix='imp_',
    #         incomplete_keys=('incomplete', 'incomplete_raw'),
    #         visit_wise=True,
    #         inplace=False,
    #         verbose=True
    #     )
    #
    #     test_samples = add_imp_fields(
    #         test_samples,
    #         feature_order=('conditions', 'procedures', 'drugs'),
    #         new_prefix='new_',
    #         out_prefix='imp_',
    #         incomplete_keys=('incomplete', 'incomplete_raw'),
    #         visit_wise=True,
    #         inplace=False,
    #         verbose=False
    #     )
    #
    #     # 2) 让下游模型用 imp_* 作为输入特征
    #     config['FEATURE'] = ['imp_conditions', 'imp_procedures', 'imp_drugs']
    #
    #     # 3) 关键：convert_dataset 时 all=True，确保 input_info 覆盖 imp_* keys
    #     train_dataset = convert_dataset(
    #         train_samples, dataset_name=config['DATASET'], task_name=config['TASK'],
    #         valid=False, all=True
    #     )
    #     test_dataset = convert_dataset(
    #         test_samples, dataset_name=config['DATASET'], task_name=config['TASK'],
    #         valid=False, all=True
    #     )
    #
    #     print("Augment+Merge Done!")
    #
    #     # 4) 关键：把 model 初始化用的 dataset 换成“包含 imp_* 的训练集 dataset”
    #     #    （否则 tokenizer 仍然从旧 sample_dataset 里找 imp_conditions，会 KeyError）
    #     sample_dataset = train_dataset
    #
    #     # 5) dataloader
    #     collate_fn = get_special_input(config)
    #     train_dataloader = get_dataloader(train_dataset, batch_size=config['BATCH'], shuffle=True,
    #                                       drop_last=True, collate_fn=collate_fn)
    #     test_dataloader = get_dataloader(test_dataset, batch_size=config['BATCH'], shuffle=False,
    #                                      drop_last=False, collate_fn=collate_fn)
    #
    #     # 6) 验证
    #     # print("FEATURE keys used by model:", config['FEATURE'])
    #     # b = next(iter(train_dataloader))
    #     # print("Batch keys:", b.keys())
    #     # print("Has imp_conditions?", 'imp_conditions' in b)
    #
    #     # print("AAAAAA", train_dataset[0])
    #     print("Augment Done!")
    #
    # # collate_fn = get_special_input(config)
    # # # missing_statistics(train_dataset, config)
    # # train_dataloader = get_dataloader(train_dataset, batch_size=config['BATCH'], shuffle=True, drop_last=True, collate_fn=collate_fn)
    # # # val_dataloader = get_dataloader(val_dataset, batch_size=config['BATCH']*5, shuffle=False, drop_last=True)
    # # test_dataloader = get_dataloader(test_dataset, batch_size=config['BATCH'], shuffle=False, drop_last=False, collate_fn=collate_fn) # 少数类可能抽不到，草

    # # ===================== ALL_Views 结果保存初始化（放在 ratio 循环外） =====================
    # save_dir = os.path.join(config['LOGDIR'], config['TASK'], "Ratio_Results")
    # os.makedirs(save_dir, exist_ok=True)
    #
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    # base_path = os.path.join(
    #     save_dir,
    #     f"{config['DATASET']}-{config['TASK']}-{config['MODEL']}-{exp_num}-dataset_task_ratio_scores-{timestamp}"
    # )
    # csv_path = base_path + ".csv"
    #
    # results_min = []  # 只存四列
    # prefix = f"{config['DATASET']}-{config['MODEL']}-{exp_num}-{timestamp}"
    # # csv_path = os.path.join(save_dir, prefix + ".csv")
    # json_path = os.path.join(save_dir, prefix + ".json")
    #
    # # 是否保存每个 ratio 的样本
    # SAVE_PKLS = False
    # pkl_dir = os.path.join(save_dir, prefix + "_pkls")
    # if SAVE_PKLS:
    #     os.makedirs(pkl_dir, exist_ok=True)
    #
    # results = []  # 用来存每个 ratio 的指标
    #
    # raw_train_samples = copy.deepcopy(train_samples)
    # raw_test_samples = copy.deepcopy(test_samples)
    #
    # # raw_train_samples/raw_test_samples 必须先有（之前已经在 sweep 前 deepcopy 了）
    # tokenizer_samples_all = copy.deepcopy(raw_train_samples) + copy.deepcopy(raw_test_samples)
    #
    # # 用全量样本建一个 dataset（让 label_tokenizer 词表覆盖所有 label）
    # tokenizer_dataset_all = convert_dataset(
    #     tokenizer_samples_all,
    #     dataset_name=config['DATASET'],
    #     task_name=config['TASK'],
    #     valid=False,
    #     all=True
    # )
    #
    # # 从“原始样本”里筛出三视图都完整的子集，供 ratio>0 人工掩码用
    # complete_train = filter_complete_samples(raw_train_samples)
    # complete_test = filter_complete_samples(raw_test_samples)
    #
    # print(f"[complete subset] train={len(complete_train)}/{len(raw_train_samples)}  "
    #       f"test={len(complete_test)}/{len(raw_test_samples)}")
    #
    # ratios = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    #
    # for r in ratios:
    #     print(f"\n================= ratio={r} =================")
    #
    #     if r == 0.0:
    #         # ratio=0：用旧流程的基准数据（一定用 deepcopy，避免被后续循环污染）
    #         train_samples = copy.deepcopy(raw_train_samples)
    #         test_samples = copy.deepcopy(raw_test_samples)
    #
    #         # 生成 imp_*（真实缺失位置用 new_*，非缺失用原始）
    #         train_samples = add_imp_fields(
    #             train_samples,
    #             feature_order=('conditions', 'procedures', 'drugs'),
    #             new_prefix='new_', out_prefix='imp_',
    #             incomplete_keys=('incomplete', 'incomplete_raw'),
    #             visit_wise=True, inplace=False, verbose=True
    #         )
    #         test_samples = add_imp_fields(
    #             test_samples,
    #             feature_order=('conditions', 'procedures', 'drugs'),
    #             new_prefix='new_', out_prefix='imp_',
    #             incomplete_keys=('incomplete', 'incomplete_raw'),
    #             visit_wise=True, inplace=False, verbose=False
    #         )
    #
    #     else:
    #         # ratio>0：三视图完整样本 -> 人工按视图掩码 -> 补全 -> 融合 imp_*
    #         masked_train = apply_view_mask_ratio(complete_train, ratio=r, seed=123, mode='bernoulli')
    #         masked_test = apply_view_mask_ratio(complete_test, ratio=r, seed=456, mode='bernoulli')
    #
    #         train_samples, test_samples = augment_dataset(
    #             nnet=generated_model,
    #             config=diff_config,
    #             config_health=config,
    #             hp_dataset=[masked_train, masked_test],
    #             tokenizers=tokenizers,
    #             root_to=None
    #         )
    #
    #         # 缺失视图用 new_，非缺失视图用真实值 -> imp_*
    #         train_samples = build_imp_fields(train_samples)
    #         test_samples = build_imp_fields(test_samples)
    #
    #     # 下游统一使用 imp_*
    #     config['FEATURE'] = ['imp_conditions', 'imp_procedures', 'imp_drugs']
    #
    #     # convert_dataset 必须 all=True，确保 input_info 里有 imp_*，否则 tokenizer KeyError
    #     train_dataset = convert_dataset(
    #         train_samples, dataset_name=config['DATASET'], task_name=config['TASK'],
    #         valid=False, all=True
    #     )
    #     test_dataset = convert_dataset(
    #         test_samples, dataset_name=config['DATASET'], task_name=config['TASK'],
    #         valid=False, all=True
    #     )
    #
    #     # tokenizer 用的 dataset 必须包含 imp_*，所以 sample_dataset 用 train_dataset
    #     sample_dataset = tokenizer_dataset_all
    #
    #     config['USE_NEW_MERGE'] = True
    #     config['FEATURE'] = ['conditions', 'procedures', 'drugs']  # 保持原样（Diffrm forward 写死用这些）
    #
    #     collate_fn = get_special_input(config)
    #     train_dataloader = get_dataloader(
    #         train_dataset, batch_size=config['BATCH'],
    #         shuffle=True, drop_last=True, collate_fn=collate_fn
    #     )
    #     test_dataloader = get_dataloader(
    #         test_dataset, batch_size=config['BATCH'],
    #         shuffle=False, drop_last=False, collate_fn=collate_fn
    #     )
    #
    #     # sanity check（建议保留一次）
    #     b = next(iter(train_dataloader))
    #     print("FEATURE:", config['FEATURE'])
    #     print("Has imp_conditions?", 'imp_conditions' in b)
    #
    #     # --------- 训练 + 评估（这里拿到 scores）---------
    #     existing_model = None
    #     start = time.time()
    #     scores, _ = run_pcf(
    #         sample_dataset, train_dataloader, test_dataloader, config,
    #         exp_num=f"{exp_num}-ratio{r}",
    #         existing_model=existing_model,
    #         mode=mode,
    #         aux_data={'p_grouped': p_grouped}
    #     )
    #     end = time.time()
    #     print("Training done! Cost {} s".format(end - start))
    #
    #     # 保存每个 ratio 的样本（很大）
    #     if SAVE_PKLS:
    #         with open(os.path.join(pkl_dir, f"train_samples_ratio{r}.pkl"), "wb") as f:
    #             pickle.dump(train_samples, f)
    #         with open(os.path.join(pkl_dir, f"test_samples_ratio{r}.pkl"), "wb") as f:
    #             pickle.dump(test_samples, f)
    #
    #     # --------- 收集并保存 ---------
    #     # 清洗 scores（torch/numpy 标量转 float，便于序列化）
    #     scores_clean = {}
    #     if isinstance(scores, dict):
    #         for k, v in scores.items():
    #             try:
    #                 scores_clean[k] = float(v.item()) if hasattr(v, "item") else float(v)
    #             except Exception:
    #                 scores_clean[k] = v
    #     else:
    #         scores_clean = {"scores": scores}
    #
    #     row = {
    #         "dataset": config["DATASET"],
    #         "task": config["TASK"],
    #         "ratio": float(r),
    #         "scores": json.dumps(scores_clean, ensure_ascii=False)  # 一格存完整scores
    #     }
    #     results_min.append(row)
    #
    #     # 每轮覆盖写一次 Excel（防止跑一半崩掉丢结果）
    #     df = pd.DataFrame(results_min)
    #     df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    #     print("[saved csv]", csv_path)

        # # ===================== ALL_Views 结果保存初始化（放在 ratio 循环外） =====================
        # save_dir = os.path.join(config['LOGDIR'], config['TASK'], "Ratio_Results")
        # os.makedirs(save_dir, exist_ok=True)
        #
        # timestamp = time.strftime("%Y%m%d-%H%M%S")
        # base_path = os.path.join(
        #     save_dir,
        #     f"{config['DATASET']}-{config['TASK']}-{config['MODEL']}-{exp_num}-dataset_task_ratio_scores-{timestamp}"
        # )
        # csv_path = base_path + ".csv"
        #
        # results_min = []  # 只存四列
        # prefix = f"{config['DATASET']}-{config['MODEL']}-{exp_num}-{timestamp}"
        # # csv_path = os.path.join(save_dir, prefix + ".csv")
        # json_path = os.path.join(save_dir, prefix + ".json")
        #
        # # 是否保存每个 ratio 的样本
        # SAVE_PKLS = False
        # pkl_dir = os.path.join(save_dir, prefix + "_pkls")
        # if SAVE_PKLS:
        #     os.makedirs(pkl_dir, exist_ok=True)
        #
        # results = []  # 用来存每个 ratio 的指标
        #
        # raw_train_samples = copy.deepcopy(train_samples)
        # raw_test_samples = copy.deepcopy(test_samples)
        #
        # # raw_train_samples/raw_test_samples 必须先有（之前已经在 sweep 前 deepcopy 了）
        # tokenizer_samples_all = copy.deepcopy(raw_train_samples) + copy.deepcopy(raw_test_samples)
        #
        # # 用全量样本建一个 dataset（让 label_tokenizer 词表覆盖所有 label）
        # tokenizer_dataset_all = convert_dataset(
        #     tokenizer_samples_all,
        #     dataset_name=config['DATASET'],
        #     task_name=config['TASK'],
        #     valid=False,
        #     all=True
        # )
        #
        # # 从“原始样本”里筛出三视图都完整的子集，供 ratio>0 人工掩码用
        # complete_train = filter_complete_samples(raw_train_samples)
        # complete_test = filter_complete_samples(raw_test_samples)
        #
        # print(f"[complete subset] train={len(complete_train)}/{len(raw_train_samples)}  "
        #       f"test={len(complete_test)}/{len(raw_test_samples)}")
        #
        # ratios = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        #
        # for r in ratios:
        #     print(f"\n================= ratio={r} =================")
        #
        #     if r == 0.0:
        #         # ratio=0：用旧流程的基准数据（一定用 deepcopy，避免被后续循环污染）
        #         train_samples = copy.deepcopy(raw_train_samples)
        #         test_samples = copy.deepcopy(raw_test_samples)
        #
        #         # 生成 imp_*（真实缺失位置用 new_*，非缺失用原始）
        #         train_samples = add_imp_fields(
        #             train_samples,
        #             feature_order=('conditions', 'procedures', 'drugs'),
        #             new_prefix='new_', out_prefix='imp_',
        #             incomplete_keys=('incomplete', 'incomplete_raw'),
        #             visit_wise=True, inplace=False, verbose=True
        #         )
        #         test_samples = add_imp_fields(
        #             test_samples,
        #             feature_order=('conditions', 'procedures', 'drugs'),
        #             new_prefix='new_', out_prefix='imp_',
        #             incomplete_keys=('incomplete', 'incomplete_raw'),
        #             visit_wise=True, inplace=False, verbose=False
        #         )
        #
        #     else:
        #         # ratio>0：三视图完整样本 -> 人工按视图掩码 -> 补全 -> 融合 imp_*
        #         masked_train = apply_view_mask_ratio(complete_train, ratio=r, seed=123, mode='bernoulli')
        #         masked_test = apply_view_mask_ratio(complete_test, ratio=r, seed=456, mode='bernoulli')
        #
        #         train_samples, test_samples = augment_dataset(
        #             nnet=generated_model,
        #             config=diff_config,
        #             config_health=config,
        #             hp_dataset=[masked_train, masked_test],
        #             tokenizers=tokenizers,
        #             root_to=None
        #         )
        #
        #         # 缺失视图用 new_，非缺失视图用真实值 -> imp_*
        #         train_samples = build_imp_fields(train_samples)
        #         test_samples = build_imp_fields(test_samples)
        #
        #     # 下游统一使用 imp_*
        #     config['FEATURE'] = ['imp_conditions', 'imp_procedures', 'imp_drugs']
        #
        #     # convert_dataset 必须 all=True，确保 input_info 里有 imp_*，否则 tokenizer KeyError
        #     train_dataset = convert_dataset(
        #         train_samples, dataset_name=config['DATASET'], task_name=config['TASK'],
        #         valid=False, all=True
        #     )
        #     test_dataset = convert_dataset(
        #         test_samples, dataset_name=config['DATASET'], task_name=config['TASK'],
        #         valid=False, all=True
        #     )
        #
        #     # tokenizer 用的 dataset 必须包含 imp_*，所以 sample_dataset 用 train_dataset
        #     sample_dataset = tokenizer_dataset_all
        #
        #     config['USE_NEW_MERGE'] = True
        #     config['FEATURE'] = ['conditions', 'procedures', 'drugs']  # 保持原样（Diffrm forward 写死用这些）
        #
        #     collate_fn = get_special_input(config)
        #     train_dataloader = get_dataloader(
        #         train_dataset, batch_size=config['BATCH'],
        #         shuffle=True, drop_last=True, collate_fn=collate_fn
        #     )
        #     test_dataloader = get_dataloader(
        #         test_dataset, batch_size=config['BATCH'],
        #         shuffle=False, drop_last=False, collate_fn=collate_fn
        #     )
        #
        #     # sanity check（建议保留一次）
        #     b = next(iter(train_dataloader))
        #     print("FEATURE:", config['FEATURE'])
        #     print("Has imp_conditions?", 'imp_conditions' in b)
        #
        #     # --------- 训练 + 评估（这里拿到 scores）---------
        #     existing_model = None
        #     start = time.time()
        #     scores, _ = run_pcf(
        #         sample_dataset, train_dataloader, test_dataloader, config,
        #         exp_num=f"{exp_num}-ratio{r}",
        #         existing_model=existing_model,
        #         mode=mode,
        #         aux_data={'p_grouped': p_grouped}
        #     )
        #     end = time.time()
        #     print("Training done! Cost {} s".format(end - start))
        #
        #     # 保存每个 ratio 的样本（很大）
        #     if SAVE_PKLS:
        #         with open(os.path.join(pkl_dir, f"train_samples_ratio{r}.pkl"), "wb") as f:
        #             pickle.dump(train_samples, f)
        #         with open(os.path.join(pkl_dir, f"test_samples_ratio{r}.pkl"), "wb") as f:
        #             pickle.dump(test_samples, f)
        #
        #     # --------- 收集并保存 ---------
        #     # 清洗 scores（torch/numpy 标量转 float，便于序列化）
        #     scores_clean = {}
        #     if isinstance(scores, dict):
        #         for k, v in scores.items():
        #             try:
        #                 scores_clean[k] = float(v.item()) if hasattr(v, "item") else float(v)
        #             except Exception:
        #                 scores_clean[k] = v
        #     else:
        #         scores_clean = {"scores": scores}
        #
        #     row = {
        #         "dataset": config["DATASET"],
        #         "task": config["TASK"],
        #         "ratio": float(r),
        #         "scores": json.dumps(scores_clean, ensure_ascii=False)  # 一格存完整scores
        #     }
        #     results_min.append(row)
        #
        #     # 每轮覆盖写一次 Excel（防止跑一半崩掉丢结果）
        #     df = pd.DataFrame(results_min)
        #     df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        #     print("[saved csv]", csv_path)
        # ===================== ALL_Views 结果保存初始化（放在 ratio 循环外） =====================
        save_dir = os.path.join(config['LOGDIR'], config['TASK'], "Ratio_Results")
        os.makedirs(save_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_path = os.path.join(
            save_dir,
            f"{config['DATASET']}-{config['TASK']}-{config['MODEL']}-{exp_num}-dataset_task_ratio_scores-{timestamp}"
        )
        csv_path = base_path + ".csv"

        results_min = []  # 只存四列
        prefix = f"{config['DATASET']}-{config['MODEL']}-{exp_num}-{timestamp}"
        # csv_path = os.path.join(save_dir, prefix + ".csv")
        json_path = os.path.join(save_dir, prefix + ".json")

        # 是否保存每个 ratio 的样本
        SAVE_PKLS = False
        pkl_dir = os.path.join(save_dir, prefix + "_pkls")
        if SAVE_PKLS:
            os.makedirs(pkl_dir, exist_ok=True)

        results = []  # 用来存每个 ratio 的指标

        raw_train_samples = copy.deepcopy(train_samples)
        raw_test_samples = copy.deepcopy(test_samples)

        # raw_train_samples/raw_test_samples 必须先有（之前已经在 sweep 前 deepcopy 了）
        tokenizer_samples_all = copy.deepcopy(raw_train_samples) + copy.deepcopy(raw_test_samples)
        tokenizer_dataset_all = convert_dataset(
            tokenizer_samples_all,
            dataset_name=config['DATASET'],
            task_name=config['TASK'],
            valid=False,
            all=True
        )

        ratios = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        views = ("conditions", "procedures", "drugs")
        # ratios = ( 0.7, 0.8, 0.9, 1.0)

        # 结果只存要的四项 + masked_view
        results_min = []

        # CSV 先写表头一次
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "task", "masked_view", "ratio", "scores"])
            w.writeheader()

        for view_to_mask in views:
            for r in ratios:
                # view_to_mask = "drugs"
                print(f"\n========== view={view_to_mask}  ratio={r} ==========")
                # assert False

                # 1) 在原始样本上，只对一个视图施加额外缺失（只在三视图完整处）
                # masked_train = mask_one_view_on_complete_samplewise(raw_train_samples, view_to_mask, r, seed=528)
                # masked_test = mask_one_view_on_complete_samplewise(raw_test_samples, view_to_mask, r, seed=529)
                masked_train = raw_train_samples  # 训练集不做 ratio 掩码
                masked_test = (raw_test_samples if r == 0.0
                               else mask_one_view_on_complete_samplewise(raw_test_samples, view_to_mask, r, seed=528))

                # 2) 调生成模型补全：会产生 new_conditions/new_procedures/new_drugs
                # 同时也会对“原本真实缺失”的部分做补全
                train_samples_aug, test_samples_aug = augment_dataset(
                    nnet=generated_model,
                    config=diff_config,
                    config_health=config,
                    hp_dataset=[masked_train, masked_test],
                    tokenizers=tokenizers,
                    root_to=None
                )

                # 3) 下游：不做 all=True（避免 ratio=1 时 validate 崩）
                train_dataset = convert_dataset(
                    train_samples_aug, dataset_name=config['DATASET'], task_name=config['TASK'],
                    valid=False, all=False
                )
                test_dataset = convert_dataset(
                    test_samples_aug, dataset_name=config['DATASET'], task_name=config['TASK'],
                    valid=False, all=False
                )

                # 4) 下游特征保持原始 key，用 collate 动态把缺失替换成 new_（不改 pkl）
                config['USE_NEW_MERGE'] = True
                config['FEATURE'] = ['conditions', 'procedures', 'drugs']

                collate_fn = get_special_input(config)
                train_dataloader = get_dataloader(train_dataset, batch_size=config['BATCH'],
                                                  shuffle=True, drop_last=True, collate_fn=collate_fn)
                test_dataloader = get_dataloader(test_dataset, batch_size=config['BATCH'],
                                                 shuffle=False, drop_last=False, collate_fn=collate_fn)

                # 5) sample_dataset 用全量 tokenizer_dataset_all
                sample_dataset = tokenizer_dataset_all

                # 6) 训练/评估下游
                start = time.time()
                scores, _ = run_pcf(
                    sample_dataset, train_dataloader, test_dataloader, config,
                    exp_num=f"{exp_num}-{view_to_mask}-ratio{r}",
                    existing_model=None,
                    mode=mode,
                    aux_data={'p_grouped': p_grouped}
                )
                print("Training done! Cost {} s".format(time.time() - start))

                # 7) 保存（dataset/task/ratio/scores），scores 存成一格 JSON 字符串
                scores_clean = {}
                if isinstance(scores, dict):
                    for k, v in scores.items():
                        try:
                            scores_clean[k] = float(v.item()) if hasattr(v, "item") else float(v)
                        except Exception:
                            scores_clean[k] = v
                else:
                    scores_clean = {"scores": scores}

                row = {
                    "dataset": config["DATASET"],
                    "task": config["TASK"],
                    "masked_view": view_to_mask,
                    "ratio": float(r),
                    "scores": json.dumps(scores_clean, ensure_ascii=False),
                }
                results_min.append(row)

                # 追加写 CSV（每次只写一行，不重复 header、不重复全量）
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=["dataset", "task", "masked_view", "ratio", "scores"])
                    w.writerow(row)

                print("[saved csv]", csv_path)


    # load_dataloader = time.time()
    # print('Dataloader done!, Cost {} s'.format(load_dataloader - endt))
    # # cache clear
    # # torch.cuda.empty_cache() # 占用内存清理
    # gc.collect()
    #
    # print("===============Train Healthcare Prediction Model!===============")
    # existing_model = None
    # if pretrain:
    #     print("Pretrain on the baseline models")
    #     start = time.time()
    #     run_pretrain(sample_dataset, config, exp_num=exp_num, special_input={'train': train_dataset, 'test': test_dataset})
    #     end_time = time.time()
    #
    #     model = load_pretrain_model(sample_dataset, config, special_input=None, exp_num=exp_num, mode=mode)
    #     print("Pretrain done! Cost {} s".format(end_time - start))
    #     existing_model = model
    #
    # start = time.time()
    # run_pcf(sample_dataset, train_dataloader, test_dataloader, config, exp_num=exp_num, existing_model=existing_model, mode=mode, aux_data={'p_grouped': p_grouped})
    # end = time.time()
    # print("Training done! Cost {} s".format(end - start))





## PHE\LOS任务
if __name__ == '__main__':
    retrain = False
    augment = True # ours要开
    pretrain = False # 只有在一些二次训练的baseline上面开启。SMART, CORGAN
    # pretrain = True
    if config['MODEL'] in  ['ours', 'MedDiff']:
        retrain = False
        # retrain = True
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


# ## Pretrain Diffmv
# if __name__ == '__main__':
#     retrain = False
#     augment = True # ours要开
#     # pretrain = False # 只有在一些二次训练的baseline上面开启。SMART, CORGAN
#     pretrain = True
#     if config['MODEL'] in  ['ours', 'MedDiff']:
#         # retrain = False
#         retrain = True
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
