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
from eval_dffs import augment_dataset, load_diffusion, evaluate_diff, eval_diffusion_latent_mae_rmse, eval_diffusion_latent_mae_rmse20, eval_diffusion_latent_mae_rmse20all, eval_diffusion_latent_mae_rmse_allviews, plot_latent_sq_error_density, plot_sq_error_hist_for_scan, mia_attack_p_audit, mia_attackP_threshold_lsimple_min, mia_attackP_threshold_lsimple_mean
from eval_dffs import leak_audit_run, build_convert_input_fn_health_v1, seed528_run_complete3v_randommask_dpm_loss_v1, seed528_run_complete3v_dpm_then_attackP_v1, seed528_print_attackP_table_v1
from eval_dffs import rare_privacy_audit, print_cases_with_translation, max_jaccard_all_new_vs_train
from eval_dffs import audit_gen_leakage_against_train, train_covered_by_generated, leakage_audit_gen_vs_train, leakage_audit_gen_vs_train_viewwise
from generation_MIA import run_rare5_conditions_allviews_leakage_audit, pick_rare5_conditions_from_train_only, run_extreme5_train_only_impute_eval, build_convert_input_fn_health_v1, build_convert_input_fn_health_v11, seed528_run_complete3v_dpm_then_attackP_v2, seed528_one_sample_loss_vs_steps, audit_weighted_nn_v1, _slice_batch_one, pick_random_test_sample_and_curve
from train_hp import run_pcf, run_pretrain, load_pretrain_model, get_task_fn
from baseline_utils import missing_statistics
from data import generate_patient_group
from diffuser.diff_dataset import get_dataset

set_random_seed(config['SEED'])
# from shot_parse import SHOTDataset


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
                print("Run quick MIA audit (Attack P) ...")

                # ########################### 基于损失阈值的成员推断MIA(一步采样) ##########################
                # mia_res = mia_attack_p_audit(
                #     generated_model,
                #     diff_config,
                #     config,  # 这里的 config 就是 config_health
                #     [train_dataset, test_dataset],
                #     tokenizers,
                #     max_batches=500,
                #     repeats=4,
                #     mia_alphas=(0.2, 0.1, 0.05, 0.01, 0.001),
                #     mask_mode="train",  # 或 "data"
                #     use_train_encoder=False  # 想更贴训练True
                # )
                # print("MIA results:", mia_res)
                # mia_res = mia_attackP_threshold_lsimple_mean(
                #     generated_model, diff_config, config, [train_dataset, test_dataset], tokenizers,
                #     mia_fprs=(0.2, 0.1, 0.05, 0.01, 0.001),
                #     mask_mode="train",
                #     max_batches=500,
                #     repeats=4
                # )
                # print("MIA results:", mia_res)
                #
                # mia_res = mia_attackP_threshold_lsimple_min(
                #     generated_model, diff_config, config, [train_dataset, test_dataset], tokenizers,
                #     mia_fprs=(0.2, 0.1, 0.05, 0.01),
                #     mask_mode="train",
                #     max_batches=500,
                #     repeats=16
                # )
                # print("MIA results:", mia_res)
                # assert False

                # #################### NN-Jaccard 训练集真实库（用于对比） ########################
                # hp_dataset = [train_dataset, test_dataset]
                # dataset = get_dataset(config['DATASET'], aug=True, samples=hp_dataset, config=diff_config)
                # train_ds = dataset.get_split(split='train', labeled=True)
                # release_ds_like = load_pickle(root_to + 'test_samples.pkl')
                # # 当前的 test_samples（Subset 内含 new_*）
                # # release_ds_like = test_samples
                # res = leak_audit_run(train_ds, release_ds_like, title_prefix="MIII Release vs Train", topk=20)
                # assert False

                ########################### 基于损失阈值的成员推断MIA(多步采样) ###########################
                print("Run DPM-step MIA !!!!!!")
                # keys_map = {
                #     "conditions": ["conditions_comp", "conditions_comps", "new_conditions", "conditions"],
                #     "procedures": ["procedures_comp", "procedures_comps", "new_procedures", "procedures"],
                #     "drugs": ["drugs_comp", "drugs_comps", "new_drugs", "drugs"],
                # }
                # mia_res = seed528_run_complete3v_dpm_then_attackP_v1(
                #     nnet=generated_model,
                #     diff_config=diff_config,
                #     config_health=config,
                #     hp_dataset=[train_dataset, test_dataset],
                #     tokenizers=tokenizers,
                #     convert_input_fn=build_convert_input_fn_health_v1(tokenizers, device, config),
                #     min_len=1,
                #     sample_steps=20,
                #     repeats=4,
                #     agg="mean",
                #     strict_context=True,
                #     max_batches_train=None,  # None=全跑；想加速就填整数
                #     max_batches_test=None,
                #     seed_base=528,
                #     alphas=(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001),
                # )
                # mia_res = seed528_run_complete3v_dpm_then_attackP_v1(
                #     nnet=generated_model,
                #     diff_config=diff_config,
                #     config_health=config,
                #     hp_dataset=[train_dataset, test_dataset],
                #     tokenizers=tokenizers,
                #     convert_input_fn=build_convert_input_fn_health_v1(tokenizers, device, config),
                #     min_len=1,
                #     sample_steps=20,
                #     repeats=4,
                #     agg="mean",
                #     strict_context=True,
                #     max_batches_train=None,  # None=全跑；想加速就填整数
                #     max_batches_test=None,
                #     seed_base=528,
                #     alphas=(0.5, 0.2, 0.1, 0.05, 0.01),
                # )
                # mia = seed528_print_attackP_table_v1(mia_res)
                # print("MIA Attack-P (from masked-loss):", mia)
                # assert False
                # ################################ 加权Jaccard ##################################
                # # 1) DPM-loss + Attack-P（支持all_single/all_nonempty）
                # hp_dataset = [train_dataset, test_dataset]
                # convert_input_fn = build_convert_input_fn_health_v1(tokenizers, device, config)
                # mia_res, out = seed528_run_complete3v_dpm_then_attackP_v2(
                #     nnet=generated_model,
                #     diff_config=diff_config,
                #     config_health=config,
                #     hp_dataset=hp_dataset,
                #     tokenizers=tokenizers,
                #     convert_input_fn=convert_input_fn,
                #     sample_steps=20,
                #     repeats=4,
                #     agg="mean",
                #     strict_context=True,
                #     mask_mode="all_views",  # 关键：三视图全掩码
                #     # mask_mode="all_single",
                #     # mask_agg="min",  # K=1时无意义
                #     mask_agg="mean",  # K=1时无意义
                #     # alphas=(0.5, 0.2, 0.1, 0.05, 0.01),
                #     alphas=(0.7, 0.6, 0.5, 0.4, 0.3),
                #     return_losses=True,
                #     return_per_mask=False
                # )
                # seed528_print_attackP_table_v1(mia_res)
                # chosen_global, chosen_in_batch, curve = pick_random_test_sample_and_curve(
                #     nnet=generated_model,
                #     out_dict=out,
                #     convert_input_fn=convert_input_fn,
                #     steps_list=(5, 10, 20, 50, 100),
                #     repeats=16,
                #     agg="min",
                #     strict_context=True,
                #     # mask_mode="all_single",  # 单样本诊断时更敏感；也可 all_views
                #     mask_mode="all_views",
                #     mask_agg="mean",
                #     seed=528
                # )
                # print("chosen_global:", chosen_global, "chosen_in_batch:", chosen_in_batch)
                # print("loss-vs-steps:", curve)
                # # 2) 单条测试样本，多步steps曲线（挑test_loader里一条）
                # test_loader = out["test_loader"]
                # for indices, batch in test_loader:
                #     batch_one = _slice_batch_one(batch, 0)
                #     idx_one = indices[:1] if isinstance(indices, (list, torch.Tensor)) else indices
                #     break
                # curve = seed528_one_sample_loss_vs_steps(
                #     nnet=generated_model,
                #     indices_one=idx_one,
                #     batch_one=batch_one,
                #     convert_input_fn=convert_input_fn,
                #     steps_list=(5, 10, 20, 50),
                #     repeats=16,
                #     agg="min",
                #     strict_context=True,
                #     # mask_mode="all_single",
                #     mask_mode="all_views",
                #     mask_agg="mean",
                # )
                # print("loss vs steps:", curve)
                # # 3) Weighted audit：Gen(new_*) vs Train(real)
                # train_samples = load_pickle("/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/PHE/MIII/processed/train_samples.pkl")
                # test_samples = load_pickle("/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/data/PHE/MIII/processed/test_samples.pkl")  # query用test里new_*
                # audit_weighted_nn_v1(
                #     train_samples=train_samples,
                #     query_samples=test_samples,
                #     feats=("conditions","procedures","drugs"),
                #     query_use="new",
                #     train_use="raw",
                #     score_key="wrec",
                #     topk_tokens=50,      # 可选：只用最稀有top50 tokens做匹配（更敏感）
                #     topn=30,
                #     th_wrec=0.9,
                #     print_details=True
                # )
                # assert False

                # ############### 罕见Code判断 ##############
                # def dataset_to_samples(ds, max_n=None):
                #     """把 train_dataset/val_dataset/test_dataset 转为 list[dict] 供后续使用。"""
                #     n = len(ds)
                #     if max_n is not None:
                #         n = min(n, max_n)
                #     return [ds[i] for i in range(n)]
                # train_samples_real = dataset_to_samples(train_dataset)  # 真实训练数据
                # train_samples = load_pickle(root_to + 'train_samples.pkl')
                # test_samples = load_pickle(root_to + 'test_samples.pkl')
                # # summary only
                # ## 测试样本只有缺失时才用new_*
                # summary = rare_privacy_audit(
                #     train_samples=train_samples_real,
                #     test_samples_aug=test_samples,
                #     K=10,
                #     sim_th=0.7,
                #     rare_from_feat="conditions",
                #     audit_feats=("conditions", "procedures", "drugs"),
                #     use_imp_for_test=True,
                #     use_imp_for_train=False,
                # )
                # print("[Rare privacy audit]", summary)
                # # # summary + cases
                # summary, cases = rare_privacy_audit(
                #     train_samples=train_samples_real,
                #     test_samples_aug=test_samples,
                #     K=10,
                #     sim_th=0.7,
                #     rare_from_feat="conditions",
                #     audit_feats=("conditions", "procedures", "drugs"),
                #     use_imp_for_test=True,
                #     use_imp_for_train=False,
                #     return_cases=True,
                #     max_cases=500,
                # )
                # print("[Rare privacy audit]", summary)
                # print("high-risk cases:", len(cases))
                # ## 打印 patient_id / visit_id + 翻译 overlap
                # print_cases_with_translation(
                #     train_samples=train_samples_real,
                #     test_samples_aug=test_samples,
                #     cases=cases,
                #     audit_feats=("conditions", "procedures", "drugs"),
                #     use_imp_for_test=True,
                #     max_print=500,
                # )
                # ## 测试样本全部用new_*
                # summary, results, high = max_jaccard_all_new_vs_train(
                #     train_samples_real=train_samples_real,
                #     test_samples_aug=test_samples,
                #     feats=("conditions", "procedures", "drugs"),
                #     sim_th=0.7,
                #     top_print=20,
                #     print_details=True,
                #     max_overlap_print=500,
                # )
                # print(summary)
                # assert False

                ########################## 训练样本 5条极端样本情况 ###########################
                # ====== 0) device / convert_input_fn ======
                hp_dataset = [train_dataset, test_dataset]
                device = torch.device("cuda:" + str(diff_config["device"]) if torch.cuda.is_available() else "cpu")
                convert_input_fn = build_convert_input_fn_health_v11(tokenizers, device, config)
                # ====== 1) 关键：mask_modes 必须是 tuple/list，而不是字符串 ======
                mask_modes = ("all_views",)  # 只跑全mask（更像“无条件生成/复刻倾向”审计）
                # mask_modes = ("all_single", "all_views")   #（但all_single不适合直接判泄露）
                # ====== 2) 跑 train-only extreme5 + all_views 复刻审计 ======
                # results = run_extreme5_train_only_impute_eval(
                #     nnet=generated_model,  # 加载的 diffusion nnet
                #     diff_config=diff_config,
                #     config_health=config,
                #     hp_dataset=hp_dataset,
                #     tokenizers=tokenizers,
                #     convert_input_fn=convert_input_fn,
                #     mask_modes=mask_modes,
                #     sample_steps=20,
                #     repeats=16,
                #     repeat_agg="mean",  # mean: 平均补全表现；min: 更“最容易复刻”的敏感探针
                #
                #     strict_context_mask=True,  # 避免 mask 仍从输入泄露 GT
                #     decode_threshold=config.get("threshod", 0.5),
                #
                #     run_loss_vs_steps=True,
                #     steps_list=(5, 10, 20, 50, 100),
                #
                #     # ---- 可选：统计输出控制 ----
                #     min_len=1,
                #     topk_common=30,
                #     rare_df_le=2,
                #     overlap_topk=15,
                # )
                # print("\n[MAIN] run_extreme5_train_only_impute_eval finished.")
                # 1) 先从训练集里挑“conditions 罕见code”的 5 条
                train_c, picked_info, picked_js = pick_rare5_conditions_from_train_only(
                    diff_config=diff_config,
                    config_health=config,
                    hp_dataset=hp_dataset,
                    rare_df_le=2,  # 你可以设 1 或 2（越小越罕见）
                    min_len=1
                )
                print("picked_js =", picked_js)
                # 2) 对5条样本跑 all_views 掩码生成，并对比原始训练样本：MSE + Jaccard + weighted Jaccard
                audit = run_rare5_conditions_allviews_leakage_audit(
                    nnet=generated_model,
                    diff_config=diff_config,
                    config_health=config,
                    tokenizers=tokenizers,
                    convert_input_fn=convert_input_fn,
                    train_c=train_c,
                    picked_js=picked_js,
                    picked_info=picked_info,
                    sample_steps=20,
                    repeats=16,
                    # repeats=1,
                    repeat_agg="mean",  # mean: 平均补全能力；min: 更像“最容易复刻”的敏感探针
                    strict_context_mask=True,  # 非常关键：避免mask仍泄露GT
                    decode_threshold=config.get("threshod", 0.5),
                )
                print("\n[MAIN] rare5 all_views leakage audit finished.")

                assert False

                # ########################## 生成数据中测试集对训练集覆盖程度(不看Jaccard) ##########################
                # train_samples = load_pickle(root_to + 'train_samples.pkl')
                # test_samples = load_pickle(root_to + 'test_samples.pkl')
                # # sum_test, rows_test = audit_gen_leakage_against_train(
                # #     train_samples_real=train_samples,
                # #     query_samples=test_samples,
                # #     feats=("conditions", "procedures", "drugs"),
                # #     query_use_new=True,  # 用 new_*
                # #     train_use_new=False,  # 训练库用真实
                # #     score_key="wrec",  # 重点找“覆盖训练样本”的泄露
                # #     print_details=True,
                # # )
                # # sum_train, rows_train = audit_gen_leakage_against_train(
                # #     train_samples_real=train_samples,
                # #     query_samples=train_samples,
                # #     feats=("conditions", "procedures", "drugs"),
                # #     query_use_new=True,
                # #     train_use_new=False,
                # #     score_key="wrec",
                # #     print_details=True,
                # # )
                # # sum_cover, rows_cover = train_covered_by_generated(
                # #     train_samples_real=train_samples,
                # #     gen_samples_pool=(train_samples + test_samples),
                # #     feats=("conditions", "procedures", "drugs"),
                # #     gen_use_new=True,
                # #     th_cover=0.9,
                # #     print_details=True,
                # # )
                # # ## 看有多少tokens
                # # summary_test, rows_test = leakage_audit_gen_vs_train(
                # #     train_samples=train_samples,
                # #     query_samples=test_samples,
                # #     feats=("conditions", "procedures", "drugs"),
                # #     query_use="new_*",
                # #     train_use="raw",
                # #     score_key="wrec",  # 建议先看 wrec（包含式覆盖）
                # #     th_wrec=0.9,
                # #     print_details=True,
                # #     return_all=True,
                # # )
                # # summary_test, rows_test = leakage_audit_gen_vs_train(
                # #     train_samples=train_samples,
                # #     query_samples=train_samples,
                # #     feats=("conditions", "procedures", "drugs"),
                # #     query_use="new_*",
                # #     train_use="raw",
                # #     score_key="wrec",  # 建议先看 wrec（包含式覆盖）
                # #     th_wrec=0.9,
                # #     print_details=True,
                # #     return_all=True,
                # # )
                # ## test_samples 生成 vs train_samples 原始()
                # # summary, rows = leakage_audit_gen_vs_train_viewwise1(
                # #     train_samples=train_samples,
                # #     query_samples=test_samples,  # 这里放 test
                # #     query_use="new_*",
                # #     train_use="raw",
                # #     match_score_key="wrec",  # containment 导向（训练样本是否被包含）
                # #     th_wrec=0.9,  # 你想要“近似包含”的阈值
                # #     strict_contain=True,  # 统计严格 T ⊆ G 的比例
                # #     print_details=True,
                # #     return_all=True,
                # #     topk=20,
                # # )
                # summary, rows = leakage_audit_gen_vs_train_viewwise(
                #     train_samples=train_samples,
                #     query_samples=test_samples,
                #     query_use="new_*",
                #     train_use="raw",
                #     match_score_key="wrec",
                #     th_wrec=0.9,
                #     strict_contain=True,
                #     rare_weight_quantile=0.90,
                #     # 正常子集：只保留 union unique 最小的 50%
                #     normal_keep_frac=0.5,
                #     # normal_keep_frac=0.3,
                #     # normal_keep_frac=1.0,
                #     normal_keep_by="union",
                #     print_details=True,
                #     return_all=True
                # )
                # assert False

                del val_dataset, test_dataset
                train_samples = load_pickle(root_to + 'train_samples.pkl')
                test_samples = load_pickle(root_to + 'test_samples.pkl')

                # 4) 评估完再释放 val/test dataset，加载已经生成好的增强样本






            # else:
            #     print("Have been augmented!")
            #     del val_dataset, test_dataset
            #     train_samples = load_pickle(root_to + 'train_samples.pkl')
            #     test_samples = load_pickle(root_to + 'test_samples.pkl')
            #
            #     ## 查看train_samples train_dataset数据格式
            #     print("type(train_samples):", type(test_samples))
            #     try:
            #         print("len(train_samples):", len(test_samples))
            #     except TypeError:
            #         print("train_samples 没有 len()，可能不是 list")
            #
            #     print("=== test_samples 原始打印 ===")
            #     print(test_samples[51])
            #
            #     # 如果是 dict，看看有哪些 key、每个 key 的类型/形状
            #     if isinstance(test_samples[51], dict):
            #         print("=== train_samples[0] 字段信息 ===")
            #         for k, v in test_samples[51].items():
            #             if isinstance(v, torch.Tensor):
            #                 print(f"  {k}: Tensor, shape={v.shape}, dtype={v.dtype}")
            #             else:
            #                 # 尝试打印长度
            #                 try:
            #                     l = len(v)
            #                 except TypeError:
            #                     l = "N/A"
            #                 print(f"  {k}: type={type(v)}, len={l}")
            #     # assert False
            #
            #     ## 查看train_samples train_dataset数据格式
            #     print("type(train_dataset):", type(train_dataset))
            #     try:
            #         print("len(train_dataset):", len(train_dataset))
            #     except TypeError:
            #         print("train_dataset 没有 len()，可能不是 list")
            #
            #     print("=== train_dataset[0] 原始打印 ===")
            #     print(train_dataset[0])
            #
            #     # 如果是 dict，看看有哪些 key、每个 key 的类型/形状
            #     if isinstance(train_dataset[0], dict):
            #         print("=== train_dataset[0] 字段信息 ===")
            #         for k, v in train_dataset[0].items():
            #             if isinstance(v, torch.Tensor):
            #                 print(f"  {k}: Tensor, shape={v.shape}, dtype={v.dtype}")
            #             else:
            #                 # 尝试打印长度
            #                 try:
            #                     l = len(v)
            #                 except TypeError:
            #                     l = "N/A"
            #                 print(f"  {k}: type={type(v)}, len={l}")
            #     assert False

        train_dataset = convert_dataset(train_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
        test_dataset = convert_dataset(test_samples, dataset_name=config['DATASET'], task_name=config['TASK'], valid=False)
        # print("AAAAAA", train_dataset[0])
        print("Augment Done!")

    collate_fn = get_special_input(config)
    # missing_statistics(train_dataset, config)
    train_dataloader = get_dataloader(train_dataset, batch_size=config['BATCH'], shuffle=True, drop_last=True, collate_fn=collate_fn)
    # val_dataloader = get_dataloader(val_dataset, batch_size=config['BATCH']*5, shuffle=False, drop_last=True)
    test_dataloader = get_dataloader(test_dataset, batch_size=config['BATCH'], shuffle=False, drop_last=False, collate_fn=collate_fn) # 少数类可能抽不到，草
    load_dataloader = time.time()
    print('Dataloader done!, Cost {} s'.format(load_dataloader - endt))

    # cache clear
    # torch.cuda.empty_cache() # 占用内存清理
    gc.collect()


    print("===============Train Healthcare Prediction Model!===============")
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
    run_pcf(sample_dataset, train_dataloader, test_dataloader, config, exp_num=exp_num, existing_model=existing_model, mode=mode, aux_data={'p_grouped': p_grouped})
    end = time.time()
    print("Training done! Cost {} s".format(end - start))





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
