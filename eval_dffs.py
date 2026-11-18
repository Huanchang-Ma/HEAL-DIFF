# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : eval_dffs.py
# Time       ：12/11/2024 1:57 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description： 生成aug sequence
"""
import os
import torch
import einops
import tempfile
import numpy as np
import torch.nn.functional as F
from utils import set_random_seed
from utils import save_pickle
from diffuser import diff_utils as utils
from absl import logging
from diffuser.diff_dataset import get_dataset, collate_fn_dict
from torch.utils.data import DataLoader, Subset
from diffuser.dpm import NoiseScheduleVP, DPM_Solver
from diffuser.diff_dataset import merge_miss
from utils import pad_list
import pickle
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from collections.abc import Mapping
import json, ast

import torch.nn as nn


def _replace_multihead_attention_for_dp(module):
    from opacus.layers import DPMultiheadAttention

    class BoolMaskDPMultiheadAttention(DPMultiheadAttention):
        def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            is_causal=False,
        ):
            if key_padding_mask is not None and key_padding_mask.dtype is not torch.bool:
                key_padding_mask = key_padding_mask != 0
            if attn_mask is not None and attn_mask.dtype is not torch.bool:
                attn_mask = attn_mask != 0
            return super().forward(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            dp_attn = BoolMaskDPMultiheadAttention(
                embed_dim=child.embed_dim,
                num_heads=child.num_heads,
                dropout=child.dropout,
                bias=child.in_proj_bias is not None,
                add_bias_kv=child.bias_k is not None,
                add_zero_attn=child.add_zero_attn,
                kdim=child.kdim,
                vdim=child.vdim,
                batch_first=child.batch_first,
            )
            dp_attn.load_state_dict(child.state_dict())
            setattr(module, name, dp_attn)
            replaced += 1
        else:
            replaced += _replace_multihead_attention_for_dp(child)
    return replaced

def _replace_batchnorm_for_dp(module):
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d):
            new_layer = nn.LayerNorm(child.num_features, elementwise_affine=False)
            setattr(module, name, new_layer)
            replaced += 1
        else:
            replaced += _replace_batchnorm_for_dp(child)
    return replaced

## main_unify_DP.py
def load_diffusion_dp(config, tokenizers):
    device = torch.device(
        'cuda:' + str(config['device']) if torch.cuda.is_available() else 'cpu'
    )

    if 'num_train_visit' not in config:
        raise KeyError("load_diffusion_dp needs config['num_train_visit']")

    nnet = utils.get_nnet(config['num_train_visit'], tokenizers, **config['nnet'])

    # 和 DP 训练时保持一致：先替换结构，再加载权重
    _replace_multihead_attention_for_dp(nnet)
    _replace_batchnorm_for_dp(nnet)

    logging.info(f'load DP nnet from {config["nnet_path"]}')
    state_dict = torch.load(config['nnet_path'], map_location='cpu')

    missing_keys, unexpected_keys = nnet.load_state_dict(state_dict, strict=False)

    # 这些键在 DP 训练时被移除了，不作为致命错误
    allowed_missing_prefixes = (
        "total_visit_emb",
        "visit_centroids",
        "context_encoder.cluster.visit_centroids",
    )
    allowed_missing_exact = {
        "context_encoder.cluster.bn.running_mean",
        "context_encoder.cluster.bn.running_var",
        "context_encoder.cluster.bn.num_batches_tracked",
    }

    bad_missing = [
        k for k in missing_keys
        if (k not in allowed_missing_exact) and (not k.startswith(allowed_missing_prefixes))
    ]

    if bad_missing:
        raise RuntimeError(f"Bad missing keys after DP-compatible load: {bad_missing}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys after DP-compatible load: {unexpected_keys}")

    nnet = nnet.to(device)
    nnet.eval()
    return nnet

# ## main_unify_DiffmvDP
# def load_diffusion_dp(config, tokenizers):
#     device = torch.device(
#         'cuda:' + str(config['device']) if torch.cuda.is_available() else 'cpu'
#     )
#
#     if 'num_train_visit' not in config:
#         raise KeyError("load_diffusion_dp needs config['num_train_visit']")
#
#     nnet = utils.get_nnet(config['num_train_visit'], tokenizers, **config['nnet'])
#
#     # 和 DP 训练时保持一致：先替换结构，再加载权重
#     _replace_multihead_attention_for_dp(nnet)
#     _replace_batchnorm_for_dp(nnet)
#
#     logging.info(f'load DP nnet from {config["nnet_path"]}')
#     state_dict = torch.load(config['nnet_path'], map_location='cpu')
#
#     # 这些键是训练期缓存/缓冲区，允许因为 num_train_visit 不同而 shape 不匹配
#     allowed_shape_mismatch_prefixes = (
#         "total_visit_emb",
#         "visit_2cluster",   # 如果你后来也把它存进 state_dict，这里一起兼容
#     )
#
#     # 这些键如果 shape 不一致，也允许丢弃后重新用当前模型初始化值
#     allowed_shape_mismatch_exact = {
#         # 如有需要可继续加
#     }
#
#     # strict=False 不会忽略 shape mismatch，所以要在 load 前手动删掉
#     dropped_shape_mismatch = []
#     current_state = nnet.state_dict()
#
#     for k in list(state_dict.keys()):
#         if k not in current_state:
#             continue
#
#         ckpt_v = state_dict[k]
#         cur_v = current_state[k]
#
#         if hasattr(ckpt_v, "shape") and hasattr(cur_v, "shape") and ckpt_v.shape != cur_v.shape:
#             if k in allowed_shape_mismatch_exact or k.startswith(allowed_shape_mismatch_prefixes):
#                 logging.warning(
#                     f"[load_diffusion_dp] drop mismatched key: {k}, "
#                     f"ckpt={tuple(ckpt_v.shape)}, current={tuple(cur_v.shape)}"
#                 )
#                 state_dict.pop(k)
#                 dropped_shape_mismatch.append(k)
#             else:
#                 raise RuntimeError(
#                     f"Bad shape mismatch for key '{k}': "
#                     f"ckpt={tuple(ckpt_v.shape)}, current={tuple(cur_v.shape)}"
#                 )
#
#     missing_keys, unexpected_keys = nnet.load_state_dict(state_dict, strict=False)
#
#     # 这些键在 DP 训练/下游加载时允许缺失
#     allowed_missing_prefixes = (
#         "total_visit_emb",
#         "visit_centroids",
#         "visit_2cluster",
#         "context_encoder.cluster.visit_centroids",
#     )
#     allowed_missing_exact = {
#         "context_encoder.cluster.bn.running_mean",
#         "context_encoder.cluster.bn.running_var",
#         "context_encoder.cluster.bn.num_batches_tracked",
#     }
#
#     bad_missing = [
#         k for k in missing_keys
#         if (k not in allowed_missing_exact) and (not k.startswith(allowed_missing_prefixes))
#     ]
#
#     if bad_missing:
#         raise RuntimeError(f"Bad missing keys after DP-compatible load: {bad_missing}")
#     if unexpected_keys:
#         raise RuntimeError(f"Unexpected keys after DP-compatible load: {unexpected_keys}")
#
#     if dropped_shape_mismatch:
#         logging.info(f"[load_diffusion_dp] dropped mismatched keys: {dropped_shape_mismatch}")
#
#     nnet = nnet.to(device)
#     nnet.eval()
#     return nnet



def convert_input(batch_data, tokenizers, device, config_health=None):
    # padding到相同的长度便于转为tensor
    indices, batch = batch_data
    feature_keys = ['conditions', 'drugs', 'procedures']

    for feature_key in feature_keys:
        # tokenizers
        # print("KKKKKKKKK", batch[feature_key])
        batch[feature_key] = tokenizers[feature_key].batch_encode_2d(batch[feature_key])
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

        batch['has_note'] = False
        # if config_health['DATASET'] == 'MIV-Note':
        #     batch['note'] = torch.tensor(batch['note'], device=device)  # cur, B,D
        #     batch['note' + '_hist'] = pad_list(batch['note' + '_hist'], device=device)  # hist, B,T,D
        #     batch['note' + '_hist_mask'] = batch['note' + '_hist'].sum(dim=-1) != 0  # B,T
        #
        #     # 作为context,需要合并
        #     batch['note' + '_hist'] = torch.cat([batch['note' + '_hist'], batch['note'].unsqueeze(dim=1)], dim=1)  # B,T+1,D
        #     batch['note' + '_mask'] = batch['note'].sum(dim=-1) != 0  # B,D
        #     batch['note' + '_hist_mask'] = torch.cat([batch['note' + '_hist_mask'], batch['note' + '_mask'].unsqueeze(dim=1)], dim=1)  # B,T+1
        #
        #     batch['has_note'] = True


    batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)


    aligned_data = [batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1) for feature_key in
                    feature_keys]  # B,V,1
    batch['mask_hist'] = torch.cat(aligned_data, dim=-1)  # B, V, 3

    return indices, batch



def evaluate_diff(best_model_path, config, config_health, hp_dataset, tokenizers, exp_num='0'):
    print("eval_diffusion")
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config["nnet_path"] = best_model_path

    set_random_seed(config_health['SEED'])
    data_n = config_health['DATASET']
    task = config_health['TASK']
    device = torch.device('cuda:' + str(config['device']) if torch.cuda.is_available() else 'cpu')
    logging.info(f'using device: {device}')

    utils.set_logger(log_level='info', fname=os.path.join(config['workdir'], data_n + '-' + task + '-' + exp_num + '-output.log'))

    dataset = get_dataset(data_n, aug=False, samples=hp_dataset, config=config)
    train_dataset = dataset.get_split(split='train', labeled=True)
    batch_size = config['train']['batch_size'] #// accelerator.num_processes
    num_train_visit = len(train_dataset)
    print("XXXXXXXXXXX", num_train_visit)

    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config['sample']['mini_batch_size'], shuffle=True,
                                     drop_last=False,
                                     num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=collate_fn_dict)
    nnet = utils.get_nnet(num_train_visit, tokenizers, **config['nnet'])
    logging.info(f'load nnet from {config["nnet_path"]}')
    nnet.load_state_dict(torch.load(config['nnet_path'], map_location='cpu'))
    nnet = nnet.to(device)
    # nnet.test_e_step()

    nnet.eval()

    def cfg_nnet(x, timesteps, context):
        # classifier-free
        _cond = nnet(x, timesteps, context=context)
        if config['sample']['scale'] == 0:
            return _cond
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet(x, timesteps, context=_empty_context)
        return _cond + config['sample']['scale'] * (_cond - _uncond)


    def decode(_batch, get_logits=None, thred=config['threshod']): # 可以进一步rounding
        nnet.eval()
        hard_samples = {}
        for index, feature_key in enumerate(tokenizers.keys()):
            logits = get_logits(_batch[:,index,:], feature_key)
            array_prob = F.sigmoid(logits).detach().cpu().numpy()
            max_indices = np.argmax(array_prob, axis=1) # B

            array_prob[array_prob >= thred] = 1
            array_prob[array_prob < thred] = 0  # 优化同步
            indices = []
            for index, row in enumerate(array_prob):
                indice = (np.where(row == 1)[0]+2).tolist()
                if len(indice) <1:
                    indice = [max_indices[index]+2]
                indices.append(indice)
            # indices = [(np.where(row == 1)[0]+2).tolist() for row in array_prob] # 加上padding项
            tokens = tokenizers[feature_key].batch_decode_2d(indices) # indices+2
            hard_samples[feature_key] = tokens # 没有考虑缺失的比例信息。
        # print(hard_samples['conditions'])
        # print("AAAAAAA", hard_samples.keys(), len(hard_samples['conditions']))
        # print("BBBBB", hard_samples['conditions'][0])

        return _batch, hard_samples


    def get_test_generator():
        while True:
            for indices, data in test_dataset_loader:
                yield indices, data

    test_generator = get_test_generator()
    _betas = utils.stable_diffusion_beta_schedule()
    N = len(_betas)

    logging.info(config['sample'])
    logging.info(f'sample: n_samples={len(test_dataset)}')

    def dpm_solver_sample(_n_samples, _sample_steps, **kwargs):
        
        _z_init = torch.randn(_n_samples, *config['z_shape'], device=device)  # 从正态分布中采样噪声。 # B,3,128
        z_init2 = kwargs['z_init']
        mask = kwargs['mask'].unsqueeze(dim=-1).repeat(1, 1, config['z_shape'][1])  # B,3,128
        _z_init = torch.where(mask, _z_init, z_init2)
        noise_schedule = NoiseScheduleVP(schedule='discrete',
                                         betas=torch.tensor(_betas, device=device).float())  # 选一个加噪声的方式
        def model_fn(x, t_continuous):
            t = t_continuous * N
            return cfg_nnet(x, t, kwargs['context'])

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / N, T=1.)  # embedding， 感觉这里需要rounding
        return decode(_z, nnet.get_logits)

    def sample_fn(_n_samples):
        indices, batch = next(test_generator)
        assert indices.size(0) == _n_samples
        indices, sample_input = convert_input((indices, batch),tokenizers, device, config_health)
        _z, _context_emb, _, labels, hard_labels = nnet.encode_everything_test(indices, **batch)
        return labels, hard_labels, dpm_solver_sample(_n_samples, config['sample']['sample_steps'], context=_context_emb, z_init=_z, mask=sample_input['mask'])


    with tempfile.TemporaryDirectory() as temp_path:
        path = config['sample']['path'] + data_n + '/' + task or temp_path
        os.makedirs(path, exist_ok=True)
        logging.info(f'Samples are saved in {path}')
        utils.sample2dir(path, len(test_dataset), config['sample']['mini_batch_size'], sample_fn,tokenizers,
                         dataset.unpreprocess)
        _quality, _jaccard = utils.calculate_quality(path)
        logging.info(f'nnet_path={config["nnet_path"]}, quality={_quality}, jaccard={_jaccard}') # 日志不会少，但是tem_path会少


def load_diffusion(config, tokenizers):
    # data_n = config_health['DATASET']
    device = torch.device('cuda:' + str(config['device']) if torch.cuda.is_available() else 'cpu')
    # dataset = get_dataset(data_n, aug=True, samples=hp_dataset, config=config)
    # train_dataset = dataset.get_split(split='train', labeled=True)
    # batch_size = config['train']['batch_size'] #// accelerator.num_processes
    nnet = utils.get_nnet(config['num_train_visit'], tokenizers, **config['nnet'])
    logging.info(f'load nnet from {config["nnet_path"]}')  # 这个path哪里来的
    nnet.load_state_dict(torch.load(config['nnet_path'], map_location='cpu'))
    nnet = nnet.to(device)
    return nnet



def augment_dataset(nnet, config, config_health, hp_dataset, tokenizers, root_to=None):
    nnet.eval()
    device = torch.device('cuda:' + str(config['device']) if torch.cuda.is_available() else 'cpu')
    # 转换数据集
    data_n = config_health['DATASET']
    task = config_health['TASK']
    # 下面开始补全
    dataset = get_dataset(data_n, aug=True, samples=hp_dataset, config=config) # 无关。
    batch_size = config['train']['batch_size']  # // accelerator.num_processes
    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                      num_workers=8, pin_memory=True, persistent_workers=True,
                                      collate_fn=collate_fn_dict)
    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config['sample']['mini_batch_size'], shuffle=False,
                                     drop_last=False,
                                     num_workers=8, pin_memory=True, persistent_workers=True,
                                     collate_fn=collate_fn_dict)

    def cfg_nnet(x, timesteps, context):
        # classifier-free
        _cond = nnet(x, timesteps, context=context)
        if config['sample']['scale'] == 0:
            return _cond
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet(x, timesteps, context=_empty_context)
        return _cond + config['sample']['scale'] * (_cond - _uncond)

    def decode(_batch, get_logits=None, thred=config['threshod']):
        prior_data = config.get("prior_data", None)
        if prior_data is not None and config.get("sample", {}).get("use_prior", False):
            from prior_utils import apply_prior_decode_np
            hard_samples = apply_prior_decode_np(_batch, nnet, tokenizers, config)
            return _batch, hard_samples

        nnet.eval()
        hard_samples = {}
        for index, feature_key in enumerate(tokenizers.keys()):
            logits = get_logits(_batch[:,index,:],feature_key)
            array_prob = F.sigmoid(logits).detach().cpu().numpy()
            max_indices = np.argmax(array_prob, axis=1) # B

            array_prob[array_prob >= thred] = 1
            array_prob[array_prob < thred] = 0  # 优化同步
            indices = []
            for index, row in enumerate(array_prob):
                indice = (np.where(row == 1)[0]+2).tolist()
                if len(indice) <1:
                    indice = [max_indices[index]+2]
                indices.append(indice)
            tokens = tokenizers[feature_key].batch_decode_2d(indices) # indices+2
            hard_samples[feature_key] = tokens # 没有考虑缺失的比例信息。

        return _batch, hard_samples

    def get_train_generator():
        while True:
            for indices, data in train_dataset_loader:
                yield indices, data

    def get_test_generator():
        while True:
            for indices, data in test_dataset_loader:
                yield indices, data

    train_generator = get_train_generator()
    test_generator = get_test_generator()


    _betas = utils.stable_diffusion_beta_schedule()
    N = len(_betas)

    def dpm_solver_sample(_n_samples, _sample_steps, **kwargs):
        """去噪声"""

        _z_init = torch.randn(_n_samples, *config['z_shape'], device=device)  # 从正态分布中采样噪声。 # B,3,128
        z_init2 = kwargs['z_init']
        mask = kwargs['mask'].unsqueeze(dim=-1).repeat(1, 1, config['z_shape'][1])  # B,3,128
        _z_init = torch.where(mask, _z_init, z_init2)  # 从正态分布中采样噪声。 # B,3,128
        # 先加噪声，重新定义schedule
        noise_schedule = NoiseScheduleVP(schedule='discrete',
                                         betas=torch.tensor(_betas, device=device).float())  # 选一个加噪声的方式
        def model_fn(x, t_continuous):
            t = t_continuous * N
            return cfg_nnet(x, t, kwargs['context'])

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / N, T=1.)  # embedding， 感觉这里需要rounding

        return decode(_z, nnet.get_logits)

    def sample_fn_train(_n_samples):
        indices, batch = next(train_generator)
        # print("CCCBBB", indices.size(0), _n_samples)
        # assert indices.size(0) == _n_samples
        indices, sample_input = convert_input((indices, batch), tokenizers, device, config_health)
        _z, _context_emb, _, labels, hard_labels = nnet.encode_everything_test(indices, **batch)
        # return labels, hard_labels, dpm_solver_sample(_n_samples, config['sample']['sample_steps'], context=_context_emb, z_init=_z, mask=sample_input['mask'])
        return labels, hard_labels, dpm_solver_sample(indices.size(0), config['sample']['sample_steps'], context=_context_emb,  z_init=_z, mask=sample_input['mask'])

    def sample_fn(_n_samples):
        indices, batch = next(test_generator)
        assert indices.size(0) == _n_samples
        indices, sample_input = convert_input((indices, batch),tokenizers, device, config_health)
        _z, _context_emb, _, labels,hard_labels = nnet.encode_everything_test(indices, **batch)
        return labels,hard_labels, dpm_solver_sample(_n_samples, config['sample']['sample_steps'], context=_context_emb,  z_init=_z, mask=sample_input['mask'])


    logging.info(config['sample'])

    # for train
    logging.info(f'sample: n_samples={len(train_dataset)}')
    with tempfile.TemporaryDirectory() as temp_path:
        path = config['sample']['path'] + data_n + '/' + task or temp_path
        os.makedirs(path, exist_ok=True)
        logging.info(f'Samples are saved in {path}')
        aug_train_sample, aug_train_hardsample = utils.sample2dir_test(path, len(train_dataset), config['sample']['mini_batch_size'], sample_fn_train,
                         dataset.unpreprocess)

    # for test
    logging.info(f'sample: n_samples={len(test_dataset)}')
    with tempfile.TemporaryDirectory() as temp_path:
        path = config['sample']['path'] + data_n + '/' + task or temp_path
        os.makedirs(path, exist_ok=True)
        logging.info(f'Samples are saved in {path}')
        aug_test_sample, aug_test_hardsample = utils.sample2dir_test(path, len(test_dataset), config['sample']['mini_batch_size'], sample_fn,
                         dataset.unpreprocess)

    if config_health['MODEL'] == 'ours':
        soft = False
        aug_train_sample = merge_miss(hp_dataset[0], aug_train_hardsample, soft=soft)
        aug_test_sample = merge_miss(hp_dataset[1], aug_test_hardsample, soft=soft)
    elif config_health['MODEL'] == 'MedDiff':
        soft = True
        aug_train_sample = merge_miss(hp_dataset[0], aug_train_sample, soft=soft)
        aug_test_sample = merge_miss(hp_dataset[1], aug_test_sample, soft=soft)


    # 应当把D*3*d放到原有的dataset里面
    if root_to is not None:
        # 存储以便读取
        save_pickle(aug_train_sample, root_to+'train_samples.pkl')
        save_pickle(aug_test_sample, root_to+'val_samples.pkl')
        print("All aug data save to {}.".format(path))

    return aug_train_sample, aug_test_sample

############################# test_dataloader ###########################
def augment_dataset1(nnet, config, config_health, hp_dataset, tokenizers, root_to=None):
    nnet.eval()
    device = torch.device('cuda:' + str(config['device']) if torch.cuda.is_available() else 'cpu')
    # 转换数据集
    data_n = config_health['DATASET']
    task = config_health['TASK']
    # 下面开始补全
    dataset = get_dataset(data_n, aug=True, samples=hp_dataset, config=config) # 无关。
    batch_size = config['train']['batch_size']  # // accelerator.num_processes
    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                      num_workers=8, pin_memory=True, persistent_workers=True,
                                      collate_fn=collate_fn_dict)
    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config['sample']['mini_batch_size'], shuffle=False,
                                     drop_last=False,
                                     num_workers=8, pin_memory=True, persistent_workers=True,
                                     collate_fn=collate_fn_dict)

    def cfg_nnet(x, timesteps, context):
        # classifier-free
        _cond = nnet(x, timesteps, context=context)
        if config['sample']['scale'] == 0:
            return _cond
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet(x, timesteps, context=_empty_context)
        return _cond + config['sample']['scale'] * (_cond - _uncond)

    def decode(_batch, get_logits=None, thred=config['threshod']):
        prior_data = config.get("prior_data", None)
        if prior_data is not None and config.get("sample", {}).get("use_prior", False):
            from prior_utils import apply_prior_decode_np
            hard_samples = apply_prior_decode_np(_batch, nnet, tokenizers, config)
            return _batch, hard_samples

        nnet.eval()
        hard_samples = {}
        for index, feature_key in enumerate(tokenizers.keys()):
            logits = get_logits(_batch[:,index,:],feature_key)
            array_prob = F.sigmoid(logits).detach().cpu().numpy()
            max_indices = np.argmax(array_prob, axis=1) # B

            array_prob[array_prob >= thred] = 1
            array_prob[array_prob < thred] = 0  # 优化同步
            indices = []
            for index, row in enumerate(array_prob):
                indice = (np.where(row == 1)[0]+2).tolist()
                if len(indice) <1:
                    indice = [max_indices[index]+2]
                indices.append(indice)
            tokens = tokenizers[feature_key].batch_decode_2d(indices) # indices+2
            hard_samples[feature_key] = tokens # 没有考虑缺失的比例信息。

        return _batch, hard_samples

    def get_train_generator():
        while True:
            for indices, data in train_dataset_loader:
                yield indices, data

    def get_test_generator():
        while True:
            for indices, data in test_dataset_loader:
                yield indices, data

    train_generator = get_train_generator()
    test_generator = get_test_generator()


    _betas = utils.stable_diffusion_beta_schedule()
    N = len(_betas)

    def dpm_solver_sample(_n_samples, _sample_steps, **kwargs):
        """去噪声"""

        _z_init = torch.randn(_n_samples, *config['z_shape'], device=device)  # 从正态分布中采样噪声。 # B,3,128
        z_init2 = kwargs['z_init']
        mask = kwargs['mask'].unsqueeze(dim=-1).repeat(1, 1, config['z_shape'][1])  # B,3,128
        _z_init = torch.where(mask, _z_init, z_init2)  # 从正态分布中采样噪声。 # B,3,128
        # 先加噪声，重新定义schedule
        noise_schedule = NoiseScheduleVP(schedule='discrete',
                                         betas=torch.tensor(_betas, device=device).float())  # 选一个加噪声的方式
        def model_fn(x, t_continuous):
            t = t_continuous * N
            return cfg_nnet(x, t, kwargs['context'])

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / N, T=1.)  # embedding， 感觉这里需要rounding

        return decode(_z, nnet.get_logits)

    def sample_fn_train(_n_samples):
        indices, batch = next(train_generator)
        # print("CCCBBB", indices.size(0), _n_samples)
        # assert indices.size(0) == _n_samples
        indices, sample_input = convert_input((indices, batch), tokenizers, device, config_health)
        _z, _context_emb, _, labels, hard_labels = nnet.encode_everything_test(indices, **batch)
        # return labels, hard_labels, dpm_solver_sample(_n_samples, config['sample']['sample_steps'], context=_context_emb,  z_init=_z, mask=sample_input['mask'])
        return labels, hard_labels, dpm_solver_sample(indices.size(0), config['sample']['sample_steps'], context=_context_emb, z_init=_z, mask=sample_input['mask'])

    def sample_fn(_n_samples):
        indices, batch = next(test_generator)
        assert indices.size(0) == _n_samples
        indices, sample_input = convert_input((indices, batch),tokenizers, device, config_health)
        _z, _context_emb, _, labels,hard_labels = nnet.encode_everything_test(indices, **batch)
        return labels,hard_labels, dpm_solver_sample(_n_samples, config['sample']['sample_steps'], context=_context_emb,  z_init=_z, mask=sample_input['mask'])


    logging.info(config['sample'])

    # for train
    logging.info(f'sample: n_samples={len(train_dataset)}')
    with tempfile.TemporaryDirectory() as temp_path:
        path = config['sample']['path'] + data_n + '/' + task or temp_path
        os.makedirs(path, exist_ok=True)
        logging.info(f'Samples are saved in {path}')
        aug_train_sample, aug_train_hardsample = utils.sample2dir_test(path, len(train_dataset), config['sample']['mini_batch_size'], sample_fn_train,
                         dataset.unpreprocess)

    # for test
    logging.info(f'sample: n_samples={len(test_dataset)}')
    with tempfile.TemporaryDirectory() as temp_path:
        path = config['sample']['path'] + data_n + '/' + task or temp_path
        os.makedirs(path, exist_ok=True)
        logging.info(f'Samples are saved in {path}')
        aug_test_sample, aug_test_hardsample = utils.sample2dir_test(path, len(test_dataset), config['sample']['mini_batch_size'], sample_fn,
                         dataset.unpreprocess)

    if config_health['MODEL'] == 'ours':
        soft = False
        aug_train_sample = merge_miss(hp_dataset[0], aug_train_hardsample, soft=soft)
        aug_test_sample = merge_miss(hp_dataset[1], aug_test_hardsample, soft=soft)
    elif config_health['MODEL'] == 'MedDiff':
        soft = True
        aug_train_sample = merge_miss(hp_dataset[0], aug_train_sample, soft=soft)
        aug_test_sample = merge_miss(hp_dataset[1], aug_test_sample, soft=soft)


    # 应当把D*3*d放到原有的dataset里面
    if root_to is not None:
        # 存储以便读取
        # save_pickle(aug_train_sample, root_to+'train_samples.pkl')
        save_pickle(aug_test_sample, root_to+'test_samples.pkl')
        print("All aug data save to {}.".format(path))

    return aug_test_sample
############################# test_dataloader ###########################

def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _is_view_complete(sample):
    """根据 raw test_dataset.pkl 里的 incomplete_raw 判断是否视图完整"""
    inc = sample.get("incomplete_raw")
    if inc is None:
        return True
    # 可能是 [0,0,0] 或 [[0,0,0]]
    if isinstance(inc[0], list):
        v = inc[0]
    else:
        v = inc
    return all(int(x) == 0 for x in v)


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    u = sa | sb
    i = sa & sb
    return len(i) / len(u) if len(u) > 0 else 0.0


def eval_diffusion_latent_mae_rmse1(
        nnet,
        diff_config,
        config_health,
        hp_dataset,      # [train_dataset, test_dataset]
        tokenizers,
        root_to,         # processed 目录，里面有 test_dataset.pkl
        sample_steps=50,
        use_fixed_mask=False,
):
    """
    在 test split 上评估 diffusion 重建质量：
      - 只在 raw test_dataset.pkl 中视图完整的样本上；
      - latent 空间 (z_init vs z_recon) 的 MAE / RMSE（只在 mask==1 的视图上）；
      - 离散 code 的平均 Jaccard（只在 mask==1 的视图上）。
    """
    device = torch.device('cuda:' + str(diff_config['device'])
                          if torch.cuda.is_available() else 'cpu')
    nnet.eval()

    data_n = config_health['DATASET']
    task = config_health['TASK']

    # ===== 1. 读 raw test_dataset.pkl，看哪些 visit_id 是视图完整的 =====
    raw_test_path = os.path.join(root_to, "test_dataset.pkl")
    print(f"[eval_latent] load raw test from {raw_test_path}")
    raw_test = _load_pickle(raw_test_path)
    complete_vids = {s["visit_id"] for s in raw_test if _is_view_complete(s)}
    print(f"[eval_latent] 视图完整的 test 样本数: {len(complete_vids)}")

    if len(complete_vids) == 0:
        print("[eval_latent] 没有 incomplete_raw == [0,0,0] 的样本，直接返回。")
        return

    # ===== 2. 构造 diffusion 的 dataset 和 test loader =====
    dataset = get_dataset(data_n, aug=True, samples=hp_dataset, config=diff_config)
    test_dataset = dataset.get_split(split='test', labeled=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=diff_config['sample']['mini_batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_dict,
    )

    # ===== 3. cfg_nnet + dpm_solver_sample (augment_dataset)=====
    betas = utils.stable_diffusion_beta_schedule()
    N = len(betas)

    def convert_input_test(batch_data, mask_static=True):
        # padding到相同的长度便于转为tensor
        indices, batch = batch_data

        feature_keys = ['conditions', 'drugs', 'procedures']
        for feature_key in feature_keys:
            # tokenizers
            batch[feature_key] = tokenizers[feature_key].batch_encode_2d(batch[feature_key])  # B,M
            batch[feature_key + '_hist'] = tokenizers[feature_key].batch_encode_3d(
                batch[feature_key + '_hist']
            )  # B,V,M
            origin = batch[feature_key + '_comp'].copy()
            batch[feature_key + '_comp'] = tokenizers[feature_key].batch_encode_2d(
                batch[feature_key + '_comp']
            )  # B,M

            # tensor
            batch[feature_key] = torch.tensor(batch[feature_key], dtype=torch.long, device=device)
            batch[feature_key + '_hist'] = torch.tensor(
                batch[feature_key + '_hist'], dtype=torch.long, device=device
            )
            batch[feature_key + '_comp'] = torch.tensor(
                batch[feature_key + '_comp'], dtype=torch.long, device=device
            )

            # mask
            batch[feature_key + '_mask'] = batch[feature_key] != 0  # B,M
            batch[feature_key + '_hist_mask'] = batch[feature_key + '_hist'] != 0  # B,V,M
            batch[feature_key + '_comp_mask'] = batch[feature_key + '_comp'] != 0  # B,M

            # others（保存原始 codes，后面可能调试用）
            batch[feature_key + '_comps'] = tokenizers[feature_key].batch_encode_2d(
                origin, padding=False, truncation=False
            )
            batch[feature_key + '_comps_origin'] = origin

            # note special（现在 MIII 不会走到这里，所以不用担心 pad_list）
            batch['has_note'] = False
            # if config_health['DATASET'] == 'MIV-Note':
            #     batch['note'] = torch.tensor(batch['note'], device=device)  # B,D
            #     batch['note' + '_hist'] = pad_list(batch['note' + '_hist'], device=device)  # B,T,D
            #     batch['note' + '_hist_mask'] = batch['note' + '_hist'].sum(dim=-1) != 0  # B,T
            #
            #     # 合并当前 note 到 hist
            #     batch['note' + '_hist'] = torch.cat(
            #         [batch['note' + '_hist'], batch['note'].unsqueeze(dim=1)], dim=1
            #     )  # B,T+1,D
            #     batch['note' + '_mask'] = batch['note'].sum(dim=-1) != 0  # B
            #     batch['note' + '_hist_mask'] = torch.cat(
            #         [batch['note' + '_hist_mask'], batch['note' + '_mask'].unsqueeze(dim=1)],
            #         dim=1,
            #     )  # B,T+1
            #     batch['has_note'] = True

        # mask 设置
        if mask_static:
            # 用 dataloader 里原本的 mask（即 get_dataset/augment 时生成的 mask）
            batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
        else:
            # 重新随机生成一个 mask（3 个视图），用于“随机 mask”实验
            m = 3  # 如果有 note，可以写 4
            batch['mask'] = utils.generate_mask(batch['conditions'].shape[0], m=m).to(device=device)

        # 记录 history 的 mask 状态：B,V,3
        aligned_data = [
            batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1)
            for feature_key in feature_keys
        ]
        batch['mask_hist'] = torch.cat(aligned_data, dim=-1)  # B,V,3

        return indices, batch

    def cfg_nnet(x, timesteps, context):
        _cond = nnet(x, timesteps, context=context)
        if diff_config['sample']['scale'] == 0:
            return _cond
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet(x, timesteps, context=_empty_context)
        return _cond + diff_config['sample']['scale'] * (_cond - _uncond)

    def decode_latent(z):
        """根据 latent z 解码出 hard_samples，用于 Jaccard"""
        nnet.eval()
        hard_samples = {}
        for idx, feature_key in enumerate(tokenizers.keys()):
            logits = nnet.get_logits(z[:, idx, :], feature_key)  # B, vocab_size
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            max_indices = np.argmax(prob, axis=1)

            thred = diff_config['threshod']
            prob_bin = (prob >= thred).astype(np.float32)

            indices_all = []
            for i, row in enumerate(prob_bin):
                idxs = np.where(row == 1)[0] + 2  # 跳过 pad/unk
                if len(idxs) < 1:
                    idxs = np.array([max_indices[i] + 2])
                indices_all.append(idxs.tolist())

            tokens = tokenizers[feature_key].batch_decode_2d(indices_all)
            hard_samples[feature_key] = tokens
        return hard_samples

    def dpm_solver_sample_latent(z_init, context, mask, _sample_steps):
        """
        和 augment_dataset 里的采样一样，只是多返回 z_recon 和 hard_samples：
          - z_recon: 重建后的 latent (B,3,D)
          - hard_samples: 离散解码结果
        """
        B = z_init.size(0)
        z_noisy = torch.randn(B, *diff_config['z_shape'], device=device)
        mask_latent = mask.unsqueeze(-1).repeat(1, 1, diff_config['z_shape'][1])  # B,3,D
        z_start = torch.where(mask_latent, z_noisy, z_init)

        noise_schedule = NoiseScheduleVP(
            schedule='discrete',
            betas=torch.tensor(betas, device=device).float()
        )

        def model_fn(x, t_continuous):
            t = t_continuous * N
            return cfg_nnet(x, t, context)

        dpm_solver = DPM_Solver(
            model_fn,
            noise_schedule,
            predict_x0=True,
            thresholding=False
        )

        z_recon = dpm_solver.sample(
            z_start, steps=_sample_steps, eps=1. / N, T=1.
        )
        hard_samples = decode_latent(z_recon)
        return z_recon, hard_samples

    # ===== 4. 在 test_loader 上循环，过滤掉视图不完整的样本，计算指标 =====

    sum_abs, sum_sq, count = 0.0, 0.0, 0
    jacc_dict = {"conditions": [], "procedures": [], "drugs": []}
    feature_keys = ["conditions", "procedures", "drugs"]

    for indices, batch in test_loader:
        # 4.1 先按 visit_id 过滤，只保留视图完整的那些
        vid_list = batch["visit_id"]  # list[str]
        keep = [vid in complete_vids for vid in vid_list]
        if not any(keep):
            continue

        keep_idx = torch.tensor(keep, dtype=torch.bool)
        indices = indices[keep_idx]

        def _select_field(x):
            if isinstance(x, torch.Tensor):
                return x[keep_idx.to(x.device)]
            elif isinstance(x, list):
                return [v for v, f in zip(x, keep) if f]
            else:
                return x

        batch = {k: _select_field(v) for k, v in batch.items()}
        if indices.numel() == 0:
            continue

        # 4.2 走 convert_input，得到 mask
        indices, sample_input = convert_input_test(
            (indices, batch),
            mask_static=use_fixed_mask,  # True = 用原始 mask；False = 随机新 mask
        )

        # 4.3 编码得到 z_init 和 hard_labels
        z_init, context_emb, _, _, hard_labels = nnet.encode_everything_test(
            indices, **batch
        )  # z_init: B,3,D

        # 4.4 diffusion 重建
        z_recon, hard_samples = dpm_solver_sample_latent(
            z_init, context_emb, sample_input["mask"], sample_steps
        )

        # 4.5 latent 上只在 mask==1 的位置算 MAE / RMSE
        mask_latent = sample_input["mask"].unsqueeze(-1).expand_as(z_init)  # B,3,D
        diff = z_recon - z_init
        diff_sel = diff[mask_latent]  # 展平成一维

        if diff_sel.numel() > 0:
            sum_abs += diff_sel.abs().sum().item()
            sum_sq += (diff_sel ** 2).sum().item()
            count += diff_sel.numel()

        # 4.6 离散 code 的 Jaccard，同样只看 mask==1 的视图
        mask_np = sample_input["mask"].detach().cpu().numpy()  # B,3
        for v_idx, key in enumerate(feature_keys):
            # hard_labels[key] 是 index，需要 decode
            gt_tokens_all = tokenizers[key].batch_decode_2d(hard_labels[key])
            gen_tokens_all = hard_samples[key]  # List[List[str]]

            for i in range(len(gen_tokens_all)):
                if mask_np[i, v_idx] == 0:
                    continue
                gt_codes = gt_tokens_all[i]
                gen_codes = gen_tokens_all[i]
                jacc = _jaccard(gt_codes, gen_codes)
                jacc_dict[key].append(jacc)

    if count == 0:
        print("[eval_latent] 没有用于计算的 mask==1 位置，请检查 mask 设置。")
        return

    latent_mae = sum_abs / count
    latent_rmse = float(np.sqrt(sum_sq / count))

    jacc_res = {
        k: (float(np.mean(v)) if len(v) > 0 else 0.0)
        for k, v in jacc_dict.items()
    }

    print("\n===== Test latent-space reconstruction (complete views only, masked positions) =====")
    print(f"latent MAE  = {latent_mae:.6f}")
    print(f"latent RMSE = {latent_rmse:.6f}")
    for k in feature_keys:
        print(f"Jaccard[{k}] = {jacc_res[k]:.6f}")

    return latent_mae, latent_rmse, jacc_res

def eval_diffusion_latent_mae_rmse(
        nnet,
        diff_config,
        config_health,
        hp_dataset,      # [train_dataset, test_dataset]
        tokenizers,
        root_to,         # processed 目录，里面有 test_dataset.pkl
        sample_steps=50,
        use_fixed_mask=False,
):
    """
    在 test split 上评估 diffusion 重建质量：

    - 情况 A：原始三个视图都完整的样本（incomplete_raw == [0,0,0]）；
    - 情况 B：原始至少一个视图完整的样本（排除 [1,1,1]，即 “三个视图都缺失”）。

    对每种情况都计算：
      - latent 空间 (z_init vs z_recon) 的 MAE / RMSE
        ——只在「原始视图存在且被随机 mask 掉」的位置上；
      - 离散 code 的平均 Jaccard
        ——同样只在「原始视图存在且被随机 mask 掉」的视图上。
    """
    device = torch.device('cuda:' + str(diff_config['device'])
                          if torch.cuda.is_available() else 'cpu')
    nnet.eval()

    data_n = config_health['DATASET']
    task = config_health['TASK']

    # ===== 1. 读 raw test_dataset.pkl，构建 visit_id -> incomplete_raw 映射 =====
    raw_test_path = os.path.join(root_to, "test_dataset.pkl")
    print(f"[eval_latent] load raw test from {raw_test_path}")
    raw_test = _load_pickle(raw_test_path)

    def _get_inc_vec(s):
        """统一取出 3 维的 incomplete 向量，0=完整，1=缺失。"""
        inc = s.get("incomplete_raw", s.get("incomplete", None))
        if inc is None:
            return [0, 0, 0]
        # 可能是 [0,0,0]
        if isinstance(inc, list) and len(inc) == 3 and isinstance(inc[0], int):
            return inc
        # 可能是 [[0,0,0]]
        if isinstance(inc, list) and len(inc) == 1 and isinstance(inc[0], list):
            return inc[0]
        return [0, 0, 0]

    # visit_id -> 原始 incomplete 向量
    vid2inc = {s["visit_id"]: _get_inc_vec(s) for s in raw_test}

    # 情况 A：三个视图都完整
    complete3_vids = {
        vid for vid, inc in vid2inc.items()
        if all(x == 0 for x in inc)
    }
    # 情况 B：至少一个视图完整（排除 [1,1,1]）
    valid_vids = {
        vid for vid, inc in vid2inc.items()
        if not all(x == 1 for x in inc)
    }

    print(f"[eval_latent] 原始 3 视图都完整的 test 样本数: {len(complete3_vids)}")
    print(f"[eval_latent] 原始至少 1 个视图完整的 test 样本数: {len(valid_vids)}")

    if len(valid_vids) == 0:
        print("[eval_latent] 没有原始视图存在的样本（全部 [1,1,1]），直接返回。")
        return

    # ===== 2. 构造 diffusion 的 dataset 和 test loader =====
    dataset = get_dataset(data_n, aug=True, samples=hp_dataset, config=diff_config)
    test_dataset = dataset.get_split(split='test', labeled=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=diff_config['sample']['mini_batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_dict,
    )

    # ===== 3. cfg_nnet + dpm_solver_sample (augment_dataset) =====
    betas = utils.stable_diffusion_beta_schedule()
    N = len(betas)

    def convert_input_test(batch_data, mask_static=True):
        # padding到相同的长度便于转为tensor
        indices, batch = batch_data

        feature_keys = ['conditions', 'drugs', 'procedures']
        for feature_key in feature_keys:
            # tokenizers
            batch[feature_key] = tokenizers[feature_key].batch_encode_2d(batch[feature_key])  # B,M
            batch[feature_key + '_hist'] = tokenizers[feature_key].batch_encode_3d(
                batch[feature_key + '_hist']
            )  # B,V,M
            origin = batch[feature_key + '_comp'].copy()
            batch[feature_key + '_comp'] = tokenizers[feature_key].batch_encode_2d(
                batch[feature_key + '_comp']
            )  # B,M

            # tensor
            batch[feature_key] = torch.tensor(batch[feature_key], dtype=torch.long, device=device)
            batch[feature_key + '_hist'] = torch.tensor(
                batch[feature_key + '_hist'], dtype=torch.long, device=device
            )
            batch[feature_key + '_comp'] = torch.tensor(
                batch[feature_key + '_comp'], dtype=torch.long, device=device
            )

            # mask（token 级 padding mask，不是 view 级缺失）
            batch[feature_key + '_mask'] = batch[feature_key] != 0  # B,M
            batch[feature_key + '_hist_mask'] = batch[feature_key + '_hist'] != 0  # B,V,M
            batch[feature_key + '_comp_mask'] = batch[feature_key + '_comp'] != 0  # B,M

            # others
            batch[feature_key + '_comps'] = tokenizers[feature_key].batch_encode_2d(
                origin, padding=False, truncation=False
            )
            batch[feature_key + '_comps_origin'] = origin

            # note special（MIII 不会走到这里）
            batch['has_note'] = False
            # if config_health['DATASET'] == 'MIV-Note':
            #     batch['note'] = torch.tensor(batch['note'], device=device)  # B,D
            #     batch['note' + '_hist'] = pad_list(batch['note' + '_hist'], device=device)  # B,T,D
            #     batch['note' + '_hist_mask'] = batch['note' + '_hist'].sum(dim=-1) != 0  # B,T
            #
            #     batch['note' + '_hist'] = torch.cat(
            #         [batch['note' + '_hist'], batch['note'].unsqueeze(dim=1)], dim=1
            #     )  # B,T+1,D
            #     batch['note' + '_mask'] = batch['note'].sum(dim=-1) != 0  # B
            #     batch['note' + '_hist_mask'] = torch.cat(
            #         [batch['note' + '_hist_mask'], batch['note' + '_mask'].unsqueeze(dim=1)],
            #         dim=1,
            #     )  # B,T+1
            #     batch['has_note'] = True

        # view-level mask：控制哪些视图被“当作缺失”（用于 diffusion 补全）
        if mask_static:
            batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
        else:
            m = 3  # 3 个视图
            batch['mask'] = utils.generate_mask(batch['conditions'].shape[0], m=m).to(device=device)

        # 记录 history 的 mask 状态：B,V,3
        aligned_data = [
            batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1)
            for feature_key in feature_keys
        ]
        batch['mask_hist'] = torch.cat(aligned_data, dim=-1)  # B,V,3

        return indices, batch

    def cfg_nnet(x, timesteps, context):
        _cond = nnet(x, timesteps, context=context)
        if diff_config['sample']['scale'] == 0:
            return _cond
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet(x, timesteps, context=_empty_context)
        return _cond + diff_config['sample']['scale'] * (_cond - _uncond)

    def decode_latent(z):
        """根据 latent z 解码出 hard_samples，用于 Jaccard"""
        nnet.eval()
        hard_samples = {}
        for idx, feature_key in enumerate(tokenizers.keys()):
            logits = nnet.get_logits(z[:, idx, :], feature_key)  # B, vocab_size
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            max_indices = np.argmax(prob, axis=1)

            thred = diff_config['threshod']
            prob_bin = (prob >= thred).astype(np.float32)

            indices_all = []
            for i, row in enumerate(prob_bin):
                idxs = np.where(row == 1)[0] + 2  # 跳过 pad/unk
                if len(idxs) < 1:
                    idxs = np.array([max_indices[i] + 2])
                indices_all.append(idxs.tolist())

            tokens = tokenizers[feature_key].batch_decode_2d(indices_all)
            hard_samples[feature_key] = tokens
        return hard_samples

    def dpm_solver_sample_latent(z_init, context, mask, _sample_steps):
        """
        和 augment_dataset 里的采样一样，只是多返回 z_recon 和 hard_samples：
          - z_recon: 重建后的 latent (B,3,D)
          - hard_samples: 离散解码结果
        """
        B = z_init.size(0)
        z_noisy = torch.randn(B, *diff_config['z_shape'], device=device)
        mask_latent = mask.unsqueeze(-1).repeat(1, 1, diff_config['z_shape'][1])  # B,3,D
        z_start = torch.where(mask_latent, z_noisy, z_init)

        noise_schedule = NoiseScheduleVP(
            schedule='discrete',
            betas=torch.tensor(betas, device=device).float()
        )

        def model_fn(x, t_continuous):
            t = t_continuous * N
            return cfg_nnet(x, t, context)

        dpm_solver = DPM_Solver(
            model_fn,
            noise_schedule,
            predict_x0=True,
            thresholding=False
        )

        z_recon = dpm_solver.sample(
            z_start, steps=_sample_steps, eps=1. / N, T=1.
        )
        hard_samples = decode_latent(z_recon)
        return z_recon, hard_samples

    # ===== 4. 在 test_loader 上循环，过滤掉视图“全部缺失”的样本，计算两组指标 =====

    # 情况 B（至少 1 个视图完整）：全体指标
    sum_abs_all, sum_sq_all, count_all = 0.0, 0.0, 0
    jacc_all = {"conditions": [], "procedures": [], "drugs": []}

    # 情况 A（三视图都完整）：子集指标
    sum_abs_c3, sum_sq_c3, count_c3 = 0.0, 0.0, 0
    jacc_c3 = {"conditions": [], "procedures": [], "drugs": []}

    feature_keys = ["conditions", "procedures", "drugs"]

    for indices, batch in test_loader:
        # 4.1 选出「在 raw test 里至少一个视图完整」的样本
        vid_list = batch["visit_id"]  # list[str]
        keep = [vid in valid_vids for vid in vid_list]
        if not any(keep):
            continue

        keep_idx = torch.tensor(keep, dtype=torch.bool)
        indices = indices[keep_idx]

        def _select_field(x):
            if isinstance(x, torch.Tensor):
                return x[keep_idx.to(x.device)]
            elif isinstance(x, list):
                return [v for v, f in zip(x, keep) if f]
            else:
                return x

        batch = {k: _select_field(v) for k, v in batch.items()}
        if indices.numel() == 0:
            continue

        # 对当前 batch，取出原始 incomplete 信息
        batch_vids = batch["visit_id"]  # 过滤后的
        inc_list = [vid2inc[vid] for vid in batch_vids]  # list of [c,p,d]
        inc_tensor = torch.tensor(inc_list, dtype=torch.bool, device=device)  # B,3, True 表示原始缺失
        orig_complete_view = ~inc_tensor  # B,3, True=原始有真值

        # 标记哪些样本是「三视图都完整」
        all_complete_flag = orig_complete_view.all(dim=1)  # B, True 表示 [0,0,0]

        # 4.2 走 convert_input，得到“随机 mask”或原始 mask
        indices, sample_input = convert_input_test(
            (indices, batch),
            mask_static=use_fixed_mask,  # False = eval 阶段重新随机 mask
        )

        # 4.3 编码得到 z_init 和 hard_labels
        z_init, context_emb, _, _, hard_labels = nnet.encode_everything_test(
            indices, **batch
        )  # z_init: B,3,D

        # 4.4 diffusion 重建
        z_recon, hard_samples = dpm_solver_sample_latent(
            z_init, context_emb, sample_input["mask"], sample_steps
        )

        # 4.5 latent：只在「原始视图存在 且 mask==1」的位置算误差
        mask_view = sample_input["mask"]                      # B,3
        eval_view_mask = mask_view & orig_complete_view      # B,3

        diff = z_recon - z_init                               # B,3,D
        mask_latent_all = eval_view_mask.unsqueeze(-1).expand_as(diff)  # B,3,D

        diff_sel_all = diff[mask_latent_all]  # “至少 1 视图完整” 场景

        if diff_sel_all.numel() > 0:
            sum_abs_all += diff_sel_all.abs().sum().item()
            sum_sq_all += (diff_sel_all ** 2).sum().item()
            count_all += diff_sel_all.numel()

        # 情况 A：再加一个样本级 mask，要求该样本三视图都完整
        all_complete_flag_expand = all_complete_flag.unsqueeze(-1).unsqueeze(-1)  # B,1,1
        mask_latent_c3 = mask_latent_all & all_complete_flag_expand               # B,3,D
        diff_sel_c3 = diff[mask_latent_c3]

        if diff_sel_c3.numel() > 0:
            sum_abs_c3 += diff_sel_c3.abs().sum().item()
            sum_sq_c3 += (diff_sel_c3 ** 2).sum().item()
            count_c3 += diff_sel_c3.numel()

        # 4.6 离散 code 的 Jaccard
        eval_view_np = eval_view_mask.detach().cpu().numpy()      # B,3
        all_c3_np = all_complete_flag.detach().cpu().numpy()      # B

        for v_idx, key in enumerate(feature_keys):
            gt_tokens_all = tokenizers[key].batch_decode_2d(hard_labels[key])
            gen_tokens_all = hard_samples[key]

            for i in range(len(gen_tokens_all)):
                if eval_view_np[i, v_idx] == 0:
                    # 原始缺失 或 这个视图没被 mask，不评估
                    continue
                gt_codes = gt_tokens_all[i]
                gen_codes = gen_tokens_all[i]
                jacc = _jaccard(gt_codes, gen_codes)

                # 情况 B：所有 valid_vids 的样本
                jacc_all[key].append(jacc)

                # 情况 A：只统计三视图都完整的样本
                if all_c3_np[i]:
                    jacc_c3[key].append(jacc)

    # ===== 汇总输出 =====
    if count_all == 0:
        print("[eval_latent] (至少 1 视图完整) 没有用于计算的 (原始完整 & mask==1) 位置，请检查 mask 设置。")
        return

    latent_mae_all = sum_abs_all / count_all
    latent_rmse_all = float(np.sqrt(sum_sq_all / count_all))

    jacc_res_all = {
        k: (float(np.mean(v)) if len(v) > 0 else 0.0)
        for k, v in jacc_all.items()
    }

    print("\n===== 情况 B：原始至少 1 个视图完整（排除 [1,1,1]），只在原始完整 & mask==1 位置 =====")
    print(f"[ALL] latent MAE  = {latent_mae_all:.6f}")
    print(f"[ALL] latent RMSE = {latent_rmse_all:.6f}")
    for k in feature_keys:
        print(f"[ALL] Jaccard[{k}] = {jacc_res_all[k]:.6f}")

    # 情况 A 可能没有样本，单独判断
    if count_c3 == 0:
        print("\n[WARN] 情况 A：原始 3 视图都完整的样本里，没有 (原始完整 & mask==1) 位置可用。")
        latent_mae_c3 = None
        latent_rmse_c3 = None
        jacc_res_c3 = {k: None for k in feature_keys}
    else:
        latent_mae_c3 = sum_abs_c3 / count_c3
        latent_rmse_c3 = float(np.sqrt(sum_sq_c3 / count_c3))
        jacc_res_c3 = {
            k: (float(np.mean(v)) if len(v) > 0 else 0.0)
            for k, v in jacc_c3.items()
        }

        print("\n===== 情况 A：原始 3 视图都完整 (incomplete_raw == [0,0,0])，只在原始完整 & mask==1 位置 =====")
        print(f"latent MAE  = {latent_mae_c3:.6f}")  ## [COMPLETE3]
        print(f"latent RMSE = {latent_rmse_c3:.6f}")  ## [COMPLETE3]
        for k in feature_keys:
            print(f"Jaccard[{k}] = {jacc_res_c3[k]:.6f}")  ## [COMPLETE3]

    # 返回两套结果，方便你后面画图 / 写表格
    return (
        (latent_mae_all, latent_rmse_all, jacc_res_all),
        (latent_mae_c3, latent_rmse_c3, jacc_res_c3),
    )

def eval_diffusion_latent_mae_rmse20(
        nnet,
        diff_config,
        config_health,
        hp_dataset,  # [train_dataset, test_dataset]
        tokenizers,
        root_to,  # processed 目录，里面有 test_dataset.pkl
        sample_steps=50,
        cond_ratios=(0.2, 0.5, 0.7),
        proc_ratios=(0.2, 0.5, 0.7),
        drug_ratios=(0.2, 0.5, 0.7),
):
    """
    在 test split 上做 “视图缺失比例扫描”：

    - 对 conditions：只 mask conditions 视图（procedures & drugs 不 mask），
        ratios = cond_ratios，如 0.2, 0.5, 0.7；
    - 对 procedures：只 mask procedures 视图（conditions & drugs 不 mask），
        ratios = proc_ratios；
    - 对 drugs：只 mask drugs 视图（conditions & procedures 不 mask），
        ratios = drug_ratios。

    只在 “原始视图存在（incomplete_raw == 0） && 这一轮被 mask 掉” 的位置上：
        - latent 空间 (z_init vs z_recon) 计算 MAE / RMSE；
        - 离散 code 的 Jaccard（对当前正在扫描的那一个视图）。

    返回：
        results_cond, results_proc, results_drug

    其中：
        results_xxx[ratio] = {
            "mae": ...,
            "rmse": ...,
            "jaccard": float,   # 只针对对应视图
            "count": int,       # 参与 MAE/RMSE 的元素个数
        }
    """
    device = torch.device('cuda:' + str(diff_config['device'])
                          if torch.cuda.is_available() else 'cpu')
    nnet.eval()

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    data_n = config_health['DATASET']
    task = config_health['TASK']

    # ===== 1. 读取 raw test_dataset.pkl，构建 visit_id -> incomplete_raw =====
    raw_test_path = os.path.join(root_to, "test_dataset.pkl")
    print(f"[mask_sweep] load raw test from {raw_test_path}")
    raw_test = _load_pickle(raw_test_path)

    def _get_inc_vec(s):
        """统一取出 3 维的 incomplete 向量，0=完整，1=缺失。"""
        inc = s.get("incomplete_raw", s.get("incomplete", None))
        if inc is None:
            return [0, 0, 0]
        # 可能是 [0,0,0]
        if isinstance(inc, list) and len(inc) == 3 and isinstance(inc[0], int):
            return inc
        # 可能是 [[0,0,0]]
        if isinstance(inc, list) and len(inc) == 1 and isinstance(inc[0], list):
            return inc[0]
        return [0, 0, 0]

    vid2inc = {s["visit_id"]: _get_inc_vec(s) for s in raw_test}

    # 排除三个视图都缺失 [1,1,1] 的样本（没法当 ground truth）
    valid_vids = {
        vid for vid, inc in vid2inc.items()
        if not all(x == 1 for x in inc)
    }
    print(f"[mask_sweep] 至少 1 个视图原始存在的 test 样本数: {len(valid_vids)}")

    if len(valid_vids) == 0:
        print("[mask_sweep] 没有可用样本，直接返回。")
        return {}, {}, {}

    # ===== 2. 构造 diffusion 的 dataset 和 test loader =====
    dataset = get_dataset(data_n, aug=True, samples=hp_dataset, config=diff_config)
    test_dataset = dataset.get_split(split='test', labeled=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=diff_config['sample']['mini_batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_dict,
    )

    # ===== 3. diffusion 相关准备（跟 augment_dataset 类似） =====
    betas = utils.stable_diffusion_beta_schedule()
    N = len(betas)

    def convert_input_test(batch_data, mask_static=True):
        # padding到相同的长度便于转为tensor
        indices, batch = batch_data

        feature_keys = ['conditions', 'drugs', 'procedures']
        for feature_key in feature_keys:
            # tokenizers
            batch[feature_key] = tokenizers[feature_key].batch_encode_2d(batch[feature_key])  # B,M
            batch[feature_key + '_hist'] = tokenizers[feature_key].batch_encode_3d(
                batch[feature_key + '_hist']
            )  # B,V,M
            origin = batch[feature_key + '_comp'].copy()
            batch[feature_key + '_comp'] = tokenizers[feature_key].batch_encode_2d(
                batch[feature_key + '_comp']
            )  # B,M

            # tensor
            batch[feature_key] = torch.tensor(batch[feature_key], dtype=torch.long, device=device)
            batch[feature_key + '_hist'] = torch.tensor(
                batch[feature_key + '_hist'], dtype=torch.long, device=device
            )
            batch[feature_key + '_comp'] = torch.tensor(
                batch[feature_key + '_comp'], dtype=torch.long, device=device
            )

            # mask（token 级 padding mask，不是 view 级缺失）
            batch[feature_key + '_mask'] = batch[feature_key] != 0  # B,M
            batch[feature_key + '_hist_mask'] = batch[feature_key + '_hist'] != 0  # B,V,M
            batch[feature_key + '_comp_mask'] = batch[feature_key + '_comp'] != 0  # B,M

            # others
            batch[feature_key + '_comps'] = tokenizers[feature_key].batch_encode_2d(
                origin, padding=False, truncation=False
            )
            batch[feature_key + '_comps_origin'] = origin

            # note special（MIII 不会走到这里）
            batch['has_note'] = False
            # if config_health['DATASET'] == 'MIV-Note':
            #     batch['note'] = torch.tensor(batch['note'], device=device)  # B,D
            #     batch['note' + '_hist'] = pad_list(batch['note' + '_hist'], device=device)  # B,T,D
            #     batch['note' + '_hist_mask'] = batch['note' + '_hist'].sum(dim=-1) != 0  # B,T
            #
            #     batch['note' + '_hist'] = torch.cat(
            #         [batch['note' + '_hist'], batch['note'].unsqueeze(dim=1)], dim=1
            #     )  # B,T+1,D
            #     batch['note' + '_mask'] = batch['note'].sum(dim=-1) != 0  # B
            #     batch['note' + '_hist_mask'] = torch.cat(
            #         [batch['note' + '_hist_mask'], batch['note' + '_mask'].unsqueeze(dim=1)],
            #         dim=1,
            #     )  # B,T+1
            #     batch['has_note'] = True

        # view-level mask：控制哪些视图被“当作缺失”（用于 diffusion 补全）
        if mask_static:
            batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
        else:
            m = 3  # 3 个视图
            batch['mask'] = utils.generate_mask(batch['conditions'].shape[0], m=m).to(device=device)

        # 记录 history 的 mask 状态：B,V,3
        aligned_data = [
            batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1)
            for feature_key in feature_keys
        ]
        batch['mask_hist'] = torch.cat(aligned_data, dim=-1)  # B,V,3

        return indices, batch

    def cfg_nnet(x, timesteps, context):
        _cond = nnet(x, timesteps, context=context)
        if diff_config['sample']['scale'] == 0:
            return _cond
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet(x, timesteps, context=_empty_context)
        return _cond + diff_config['sample']['scale'] * (_cond - _uncond)

    def decode_latent(z):
        """根据 latent z 解码出 hard_samples，用于 Jaccard。"""
        nnet.eval()
        hard_samples = {}
        for idx, feature_key in enumerate(tokenizers.keys()):
            logits = nnet.get_logits(z[:, idx, :], feature_key)  # B, vocab_size
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            max_indices = np.argmax(prob, axis=1)

            thred = diff_config['threshod']
            prob_bin = (prob >= thred).astype(np.float32)

            indices_all = []
            for i, row in enumerate(prob_bin):
                idxs = np.where(row == 1)[0] + 2  # skip pad/unk
                if len(idxs) < 1:
                    idxs = np.array([max_indices[i] + 2])
                indices_all.append(idxs.tolist())

            tokens = tokenizers[feature_key].batch_decode_2d(indices_all)
            hard_samples[feature_key] = tokens
        return hard_samples

    def dpm_solver_sample_latent(z_init, context, mask_view, _sample_steps):
        """
        和 augment_dataset 里的采样一样，只是：
          - 用传入的 view-level mask_view 控制哪些视图从噪声开始；
          - 返回 z_recon (B,3,D) 和 hard_samples。
        """
        B = z_init.size(0)
        z_noisy = torch.randn(B, *diff_config['z_shape'], device=device)
        mask_latent = mask_view.unsqueeze(-1).repeat(1, 1, diff_config['z_shape'][1])  # B,3,D
        z_start = torch.where(mask_latent, z_noisy, z_init)

        noise_schedule = NoiseScheduleVP(
            schedule='discrete',
            betas=torch.tensor(betas, device=device).float()
        )

        def model_fn(x, t_continuous):
            t = t_continuous * N
            return cfg_nnet(x, t, context)

        dpm_solver = DPM_Solver(
            model_fn,
            noise_schedule,
            predict_x0=True,
            thresholding=False
        )

        z_recon = dpm_solver.sample(
            z_start, steps=_sample_steps, eps=1. / N, T=1.
        )
        hard_samples = decode_latent(z_recon)
        return z_recon, hard_samples

    # ===== 4. 为每个 ratio 准备累加器 =====
    results_cond = {
        r: {"sum_abs": 0.0, "sum_sq": 0.0, "count": 0, "jacc_list": []}
        for r in cond_ratios
    }
    results_proc = {
        r: {"sum_abs": 0.0, "sum_sq": 0.0, "count": 0, "jacc_list": []}
        for r in proc_ratios
    }
    results_drug = {
        r: {"sum_abs": 0.0, "sum_sq": 0.0, "count": 0, "jacc_list": []}
        for r in drug_ratios
    }

    # ===== 5. 在 test_loader 上循环，一次性对所有 ratio 统计 =====
    for indices, batch in test_loader:
        # 5.1 过滤掉三个视图全缺失的样本
        vid_list = batch["visit_id"]  # list[str]
        keep = [vid in valid_vids for vid in vid_list]
        if not any(keep):
            continue

        keep_idx = torch.tensor(keep, dtype=torch.bool)
        indices = indices[keep_idx]

        def _select_field(x):
            if isinstance(x, torch.Tensor):
                return x[keep_idx.to(x.device)]
            elif isinstance(x, list):
                return [v for v, f in zip(x, keep) if f]
            else:
                return x

        batch = {k: _select_field(v) for k, v in batch.items()}
        if indices.numel() == 0:
            continue

        # 原始 incomplete 信息 -> 哪些视图原本存在
        batch_vids = batch["visit_id"]
        inc_list = [vid2inc[vid] for vid in batch_vids]  # list of [c,p,d]
        inc_tensor = torch.tensor(inc_list, dtype=torch.bool, device=device)  # B,3
        orig_complete_view = ~inc_tensor  # B,3, True=原始有真值

        B = len(batch_vids)

        # 5.2 走 convert_input，得到编码所需的 tensor（不改 view-level mask）
        indices, sample_input = convert_input_test((indices, batch))

        # 5.3 编码得到 z_init 和 hard_labels
        z_init, context_emb, _, _, hard_labels = nnet.encode_everything_test(
            indices, **batch
        )  # z_init: B,3,D

        # 预先 decode ground-truth tokens
        gt_tokens_cond = tokenizers["conditions"].batch_decode_2d(hard_labels["conditions"])
        gt_tokens_proc = tokenizers["procedures"].batch_decode_2d(hard_labels["procedures"])
        gt_tokens_drug = tokenizers["drugs"].batch_decode_2d(hard_labels["drugs"])

        # ===== 5.a 扫描 conditions：只 mask conditions 视图 =====
        for ratio in cond_ratios:
            if ratio <= 0.0:
                continue

            mask_view = torch.zeros(B, 3, dtype=torch.bool, device=device)
            valid_cond = orig_complete_view[:, 0]  # B
            rand = torch.rand(B, device=device)
            mask_view[:, 0] = (rand < ratio) & valid_cond  # 只 mask conditions 视图

            if mask_view.sum() == 0:
                continue

            z_recon, hard_samples = dpm_solver_sample_latent(
                z_init, context_emb, mask_view, sample_steps
            )

            # 只在 “原始有真值 & 这一轮被 mask”的位置上算误差
            eval_view_mask = mask_view & orig_complete_view  # B,3
            diff = z_recon - z_init  # B,3,D
            mask_latent = eval_view_mask.unsqueeze(-1).expand_as(diff)  # B,3,D

            diff_sel = diff[mask_latent]
            if diff_sel.numel() > 0:
                results_cond[ratio]["sum_abs"] += diff_sel.abs().sum().item()
                results_cond[ratio]["sum_sq"] += (diff_sel ** 2).sum().item()
                results_cond[ratio]["count"] += diff_sel.numel()

            # Jaccard：只看 conditions 视图（index=0）
            eval_view_np = eval_view_mask.detach().cpu().numpy()  # B,3
            gen_tokens_cond = hard_samples["conditions"]

            for i in range(B):
                if eval_view_np[i, 0] == 0:
                    continue
                gt_codes = gt_tokens_cond[i]
                gen_codes = gen_tokens_cond[i]
                jacc = _jaccard(gt_codes, gen_codes)
                results_cond[ratio]["jacc_list"].append(jacc)

        # ===== 5.b 扫描 procedures：只 mask procedures 视图 =====
        for ratio in proc_ratios:
            if ratio <= 0.0:
                continue

            mask_view = torch.zeros(B, 3, dtype=torch.bool, device=device)
            valid_proc = orig_complete_view[:, 1]  # B
            rand = torch.rand(B, device=device)
            mask_view[:, 1] = (rand < ratio) & valid_proc  # 只 mask procedures 视图

            if mask_view.sum() == 0:
                continue

            z_recon, hard_samples = dpm_solver_sample_latent(
                z_init, context_emb, mask_view, sample_steps
            )

            eval_view_mask = mask_view & orig_complete_view  # B,3
            diff = z_recon - z_init  # B,3,D
            mask_latent = eval_view_mask.unsqueeze(-1).expand_as(diff)  # B,3,D

            diff_sel = diff[mask_latent]
            if diff_sel.numel() > 0:
                results_proc[ratio]["sum_abs"] += diff_sel.abs().sum().item()
                results_proc[ratio]["sum_sq"] += (diff_sel ** 2).sum().item()
                results_proc[ratio]["count"] += diff_sel.numel()

            # Jaccard：只看 procedures 视图（index=1）
            eval_view_np = eval_view_mask.detach().cpu().numpy()  # B,3
            gen_tokens_proc = hard_samples["procedures"]

            for i in range(B):
                if eval_view_np[i, 1] == 0:
                    continue
                gt_codes = gt_tokens_proc[i]
                gen_codes = gen_tokens_proc[i]
                jacc = _jaccard(gt_codes, gen_codes)
                results_proc[ratio]["jacc_list"].append(jacc)

        # ===== 5.c 扫描 drugs：只 mask drugs 视图 =====
        for ratio in drug_ratios:
            if ratio <= 0.0:
                continue

            mask_view = torch.zeros(B, 3, dtype=torch.bool, device=device)
            valid_drug = orig_complete_view[:, 2]  # B
            rand = torch.rand(B, device=device)
            mask_view[:, 2] = (rand < ratio) & valid_drug  # 只 mask drugs 视图

            if mask_view.sum() == 0:
                continue

            z_recon, hard_samples = dpm_solver_sample_latent(
                z_init, context_emb, mask_view, sample_steps
            )

            eval_view_mask = mask_view & orig_complete_view  # B,3
            diff = z_recon - z_init  # B,3,D
            mask_latent = eval_view_mask.unsqueeze(-1).expand_as(diff)  # B,3,D

            diff_sel = diff[mask_latent]
            if diff_sel.numel() > 0:
                results_drug[ratio]["sum_abs"] += diff_sel.abs().sum().item()
                results_drug[ratio]["sum_sq"] += (diff_sel ** 2).sum().item()
                results_drug[ratio]["count"] += diff_sel.numel()

            # Jaccard：只看 drugs 视图（index=2）
            eval_view_np = eval_view_mask.detach().cpu().numpy()  # B,3
            gen_tokens_drug = hard_samples["drugs"]

            for i in range(B):
                if eval_view_np[i, 2] == 0:
                    continue
                gt_codes = gt_tokens_drug[i]
                gen_codes = gen_tokens_drug[i]
                jacc = _jaccard(gt_codes, gen_codes)
                results_drug[ratio]["jacc_list"].append(jacc)

    # ===== 6. 汇总输出 =====
    def _finalize(res_dict, view_name):
        out = {}
        print(f"\n===== {view_name} mask ratio sweep =====")
        for ratio, acc in res_dict.items():
            if acc["count"] == 0:
                print(f"[{view_name}] ratio={ratio:.2f}: 没有用于计算的 (原始存在 & mask==1) 位置。")
                out[ratio] = {"mae": None, "rmse": None, "jaccard": None, "count": 0}
                continue
            mae = acc["sum_abs"] / acc["count"]
            rmse = float(np.sqrt(acc["sum_sq"] / acc["count"]))
            jacc = float(np.mean(acc["jacc_list"])) if len(acc["jacc_list"]) > 0 else 0.0
            print(
                f"[{view_name}] ratio={ratio:.2f}: "
                f"MAE={mae:.6f}, RMSE={rmse:.6f}, Jaccard={jacc:.6f}, count={acc['count']}"
            )
            out[ratio] = {"mae": mae, "rmse": rmse, "jaccard": jacc, "count": acc["count"]}
        return out

    results_cond_final = _finalize(results_cond, "conditions")
    results_proc_final = _finalize(results_proc, "procedures")
    results_drug_final = _finalize(results_drug, "drugs")

    return results_cond_final, results_proc_final, results_drug_final

def eval_diffusion_latent_mae_rmse20all(
        nnet,
        diff_config,
        config_health,
        hp_dataset,  # [train_dataset, test_dataset]
        tokenizers,
        root_to,  # processed 目录，里面有 test_dataset.pkl
        sample_steps=50,
        cond_ratios=(0.2, 0.5, 0.7, 1.0),
        proc_ratios=(0.2, 0.5, 0.7, 1.0),
        drug_ratios=(0.2, 0.5, 0.7, 1.0),
        # seed=None,
        fully_mask_other_views=True,  # 是否把其他视图全部mask(False: 其他视图不mask; True: 其他存在真值的视图mask 补全 算指标)
):
    """
    在 test split 上做 “视图缺失比例扫描”：

    - 对 conditions：只在当前轮中需要评估的视图是 conditions；
    - 对 procedures：当前轮评估 procedures；
    - 对 drugs：当前轮评估 drugs。

    fully_mask_other_views = False（默认）:
        只随机 mask 当前视图，其它视图保持完整（不 mask），
        只在「当前视图：原始存在 & 本轮被 mask」的位置上算 latent-MAE/RMSE + Jaccard。

    fully_mask_other_views = True:
        当前视图按比例 mask，
        其它两个视图只要原始存在就全部 mask（即全部从噪声重建），
        但指标仍然只在「当前视图：原始存在 & 本轮被 mask」的位置上算。
    """
    device = torch.device('cuda:' + str(diff_config['device'])
                          if torch.cuda.is_available() else 'cpu')
    nnet.eval()

    seed = 528
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_n = config_health['DATASET']
    task = config_health['TASK']

    # ===== 1. 读取 raw test_dataset.pkl，构建 visit_id -> incomplete_raw =====
    raw_test_path = os.path.join(root_to, "test_dataset.pkl")
    print(f"[mask_sweep] load raw test from {raw_test_path}")
    raw_test = _load_pickle(raw_test_path)

    def _get_inc_vec(s):
        inc = s.get("incomplete_raw", s.get("incomplete", None))
        if inc is None:
            return [0, 0, 0]
        if isinstance(inc, list) and len(inc) == 3 and isinstance(inc[0], int):
            return inc
        if isinstance(inc, list) and len(inc) == 1 and isinstance(inc[0], list):
            return inc[0]
        return [0, 0, 0]

    vid2inc = {s["visit_id"]: _get_inc_vec(s) for s in raw_test}

    # 排除三个视图都缺失 [1,1,1] 的样本
    valid_vids = {
        vid for vid, inc in vid2inc.items()
        if not all(x == 1 for x in inc)
    }
    print(f"[mask_sweep] 至少 1 个视图原始存在的 test 样本数: {len(valid_vids)}")

    if len(valid_vids) == 0:
        print("[mask_sweep] 没有可用样本，直接返回。")
        return {}, {}, {}

    # ===== 2. 构造 diffusion 的 dataset 和 test loader =====
    dataset = get_dataset(data_n, aug=True, samples=hp_dataset, config=diff_config)
    test_dataset = dataset.get_split(split='test', labeled=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=diff_config['sample']['mini_batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_dict,
    )

    # ===== 3. diffusion 相关准备 =====
    betas = utils.stable_diffusion_beta_schedule()
    N = len(betas)

    def convert_input_test(batch_data, mask_static=True):
        indices, batch = batch_data

        feature_keys = ['conditions', 'drugs', 'procedures']
        for feature_key in feature_keys:
            batch[feature_key] = tokenizers[feature_key].batch_encode_2d(batch[feature_key])  # B,M
            batch[feature_key + '_hist'] = tokenizers[feature_key].batch_encode_3d(
                batch[feature_key + '_hist']
            )  # B,V,M
            origin = batch[feature_key + '_comp'].copy()
            batch[feature_key + '_comp'] = tokenizers[feature_key].batch_encode_2d(
                batch[feature_key + '_comp']
            )  # B,M

            batch[feature_key] = torch.tensor(batch[feature_key], dtype=torch.long, device=device)
            batch[feature_key + '_hist'] = torch.tensor(
                batch[feature_key + '_hist'], dtype=torch.long, device=device
            )
            batch[feature_key + '_comp'] = torch.tensor(
                batch[feature_key + '_comp'], dtype=torch.long, device=device
            )

            batch[feature_key + '_mask'] = batch[feature_key] != 0  # B,M
            batch[feature_key + '_hist_mask'] = batch[feature_key + '_hist'] != 0  # B,V,M
            batch[feature_key + '_comp_mask'] = batch[feature_key + '_comp'] != 0  # B,M

            batch[feature_key + '_comps'] = tokenizers[feature_key].batch_encode_2d(
                origin, padding=False, truncation=False
            )
            batch[feature_key + '_comps_origin'] = origin

            batch['has_note'] = False  # MIII 不会用到 note

        # view-level 原始 mask
        if mask_static:
            batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
        else:
            m = 3
            batch['mask'] = utils.generate_mask(batch['conditions'].shape[0], m=m).to(device=device)

        aligned_data = [
            batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1)
            for feature_key in feature_keys
        ]
        batch['mask_hist'] = torch.cat(aligned_data, dim=-1)  # B,V,3

        return indices, batch

    def cfg_nnet(x, timesteps, context):
        _cond = nnet(x, timesteps, context=context)
        if diff_config['sample']['scale'] == 0:
            return _cond
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet(x, timesteps, context=_empty_context)
        return _cond + diff_config['sample']['scale'] * (_cond - _uncond)

    def decode_latent(z):
        nnet.eval()
        hard_samples = {}
        for idx, feature_key in enumerate(tokenizers.keys()):
            logits = nnet.get_logits(z[:, idx, :], feature_key)  # B, vocab_size
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            max_indices = np.argmax(prob, axis=1)

            thred = diff_config['threshod']
            prob_bin = (prob >= thred).astype(np.float32)

            indices_all = []
            for i, row in enumerate(prob_bin):
                idxs = np.where(row == 1)[0] + 2
                if len(idxs) < 1:
                    idxs = np.array([max_indices[i] + 2])
                indices_all.append(idxs.tolist())

            tokens = tokenizers[feature_key].batch_decode_2d(indices_all)
            hard_samples[feature_key] = tokens
        return hard_samples

    def dpm_solver_sample_latent(z_init, context, mask_view, _sample_steps):
        B = z_init.size(0)
        z_noisy = torch.randn(B, *diff_config['z_shape'], device=device)
        mask_latent = mask_view.unsqueeze(-1).repeat(1, 1, diff_config['z_shape'][1])  # B,3,D
        z_start = torch.where(mask_latent, z_noisy, z_init)

        noise_schedule = NoiseScheduleVP(
            schedule='discrete',
            betas=torch.tensor(betas, device=device).float()
        )

        def model_fn(x, t_continuous):
            t = t_continuous * N
            return cfg_nnet(x, t, context)

        dpm_solver = DPM_Solver(
            model_fn,
            noise_schedule,
            predict_x0=True,
            thresholding=False
        )

        z_recon = dpm_solver.sample(
            z_start, steps=_sample_steps, eps=1. / N, T=1.
        )
        hard_samples = decode_latent(z_recon)
        return z_recon, hard_samples

    # ===== 4. 为每个 ratio 准备累加器（全视图 pooled）=====
    results_cond = {r: {"sum_abs": 0.0, "sum_sq": 0.0,
                        "count": 0, "jacc_list": []}
                    for r in cond_ratios}
    results_proc = {r: {"sum_abs": 0.0, "sum_sq": 0.0,
                        "count": 0, "jacc_list": []}
                    for r in proc_ratios}
    results_drug = {r: {"sum_abs": 0.0, "sum_sq": 0.0,
                        "count": 0, "jacc_list": []}
                    for r in drug_ratios}

    # ===== 5. 在 test_loader 上循环 =====
    for indices, batch in test_loader:
        vid_list = batch["visit_id"]
        keep = [vid in valid_vids for vid in vid_list]
        if not any(keep):
            continue

        keep_idx = torch.tensor(keep, dtype=torch.bool)
        indices = indices[keep_idx]

        def _select_field(x):
            if isinstance(x, torch.Tensor):
                return x[keep_idx.to(x.device)]
            elif isinstance(x, list):
                return [v for v, f in zip(x, keep) if f]
            else:
                return x

        batch = {k: _select_field(v) for k, v in batch.items()}
        if indices.numel() == 0:
            continue

        batch_vids = batch["visit_id"]
        inc_list = [vid2inc[vid] for vid in batch_vids]  # list of [c,p,d]
        inc_tensor = torch.tensor(inc_list, dtype=torch.bool, device=device)  # B,3
        orig_complete_view = ~inc_tensor  # B,3

        B = len(batch_vids)

        indices, sample_input = convert_input_test((indices, batch))

        z_init, context_emb, _, _, hard_labels = nnet.encode_everything_test(
            indices, **batch
        )

        gt_tokens_cond = tokenizers["conditions"].batch_decode_2d(hard_labels["conditions"])
        gt_tokens_proc = tokenizers["procedures"].batch_decode_2d(hard_labels["procedures"])
        gt_tokens_drug = tokenizers["drugs"].batch_decode_2d(hard_labels["drugs"])

        # ===== conditions 扫描（全视图 pooled）=====
        for ratio in cond_ratios:
            if ratio <= 0.0:
                continue

            mask_view = torch.zeros(B, 3, dtype=torch.bool, device=device)
            valid_cond = orig_complete_view[:, 0]
            rand = torch.rand(B, device=device)
            mask_view[:, 0] = (rand < ratio) & valid_cond  # 当前视图按比例 mask

            if fully_mask_other_views:
                # 其它两个视图：只要原始存在，就全部 mask
                mask_view[:, 1] = orig_complete_view[:, 1]
                mask_view[:, 2] = orig_complete_view[:, 2]

            if mask_view.sum() == 0:
                continue

            z_recon, hard_samples = dpm_solver_sample_latent(
                z_init, context_emb, mask_view, sample_steps
            )

            # 这里改成：所有「原始存在 & mask==1」的视图都参与
            eval_view_mask = mask_view & orig_complete_view  # B,3

            diff = z_recon - z_init  # B,3,D
            mask_latent = eval_view_mask.unsqueeze(-1).expand_as(diff)
            diff_sel = diff[mask_latent]

            if diff_sel.numel() > 0:
                results_cond[ratio]["sum_abs"] += diff_sel.abs().sum().item()
                results_cond[ratio]["sum_sq"] += (diff_sel ** 2).sum().item()
                results_cond[ratio]["count"] += diff_sel.numel()

            # Jaccard：同样对所有被评估视图算一遍并 pool
            eval_view_np = eval_view_mask.detach().cpu().numpy()  # B,3
            gen_tokens_cond = hard_samples["conditions"]
            gen_tokens_proc = hard_samples["procedures"]
            gen_tokens_drug = hard_samples["drugs"]

            for i in range(B):
                # view 0: conditions
                if eval_view_np[i, 0]:
                    jacc = _jaccard(gt_tokens_cond[i], gen_tokens_cond[i])
                    results_cond[ratio]["jacc_list"].append(jacc)
                # view 1: procedures
                if eval_view_np[i, 1]:
                    jacc = _jaccard(gt_tokens_proc[i], gen_tokens_proc[i])
                    results_cond[ratio]["jacc_list"].append(jacc)
                # view 2: drugs
                if eval_view_np[i, 2]:
                    jacc = _jaccard(gt_tokens_drug[i], gen_tokens_drug[i])
                    results_cond[ratio]["jacc_list"].append(jacc)

        # ===== procedures 扫描（全视图 pooled）=====
        for ratio in proc_ratios:
            if ratio <= 0.0:
                continue

            mask_view = torch.zeros(B, 3, dtype=torch.bool, device=device)
            valid_proc = orig_complete_view[:, 1]
            rand = torch.rand(B, device=device)
            mask_view[:, 1] = (rand < ratio) & valid_proc

            if fully_mask_other_views:
                mask_view[:, 0] = orig_complete_view[:, 0]
                mask_view[:, 2] = orig_complete_view[:, 2]

            if mask_view.sum() == 0:
                continue

            z_recon, hard_samples = dpm_solver_sample_latent(
                z_init, context_emb, mask_view, sample_steps
            )

            eval_view_mask = mask_view & orig_complete_view  # B,3

            diff = z_recon - z_init
            mask_latent = eval_view_mask.unsqueeze(-1).expand_as(diff)
            diff_sel = diff[mask_latent]

            if diff_sel.numel() > 0:
                results_proc[ratio]["sum_abs"] += diff_sel.abs().sum().item()
                results_proc[ratio]["sum_sq"] += (diff_sel ** 2).sum().item()
                results_proc[ratio]["count"] += diff_sel.numel()

            eval_view_np = eval_view_mask.detach().cpu().numpy()
            gen_tokens_cond = hard_samples["conditions"]
            gen_tokens_proc = hard_samples["procedures"]
            gen_tokens_drug = hard_samples["drugs"]

            for i in range(B):
                if eval_view_np[i, 0]:
                    jacc = _jaccard(gt_tokens_cond[i], gen_tokens_cond[i])
                    results_proc[ratio]["jacc_list"].append(jacc)
                if eval_view_np[i, 1]:
                    jacc = _jaccard(gt_tokens_proc[i], gen_tokens_proc[i])
                    results_proc[ratio]["jacc_list"].append(jacc)
                if eval_view_np[i, 2]:
                    jacc = _jaccard(gt_tokens_drug[i], gen_tokens_drug[i])
                    results_proc[ratio]["jacc_list"].append(jacc)

        # ===== drugs 扫描（全视图 pooled）=====
        for ratio in drug_ratios:
            if ratio <= 0.0:
                continue

            mask_view = torch.zeros(B, 3, dtype=torch.bool, device=device)
            valid_drug = orig_complete_view[:, 2]
            rand = torch.rand(B, device=device)
            mask_view[:, 2] = (rand < ratio) & valid_drug

            if fully_mask_other_views:
                mask_view[:, 0] = orig_complete_view[:, 0]
                mask_view[:, 1] = orig_complete_view[:, 1]

            if mask_view.sum() == 0:
                continue

            z_recon, hard_samples = dpm_solver_sample_latent(
                z_init, context_emb, mask_view, sample_steps
            )

            eval_view_mask = mask_view & orig_complete_view  # B,3

            diff = z_recon - z_init
            mask_latent = eval_view_mask.unsqueeze(-1).expand_as(diff)
            diff_sel = diff[mask_latent]

            if diff_sel.numel() > 0:
                results_drug[ratio]["sum_abs"] += diff_sel.abs().sum().item()
                results_drug[ratio]["sum_sq"] += (diff_sel ** 2).sum().item()
                results_drug[ratio]["count"] += diff_sel.numel()

            eval_view_np = eval_view_mask.detach().cpu().numpy()
            gen_tokens_cond = hard_samples["conditions"]
            gen_tokens_proc = hard_samples["procedures"]
            gen_tokens_drug = hard_samples["drugs"]

            for i in range(B):
                if eval_view_np[i, 0]:
                    jacc = _jaccard(gt_tokens_cond[i], gen_tokens_cond[i])
                    results_drug[ratio]["jacc_list"].append(jacc)
                if eval_view_np[i, 1]:
                    jacc = _jaccard(gt_tokens_proc[i], gen_tokens_proc[i])
                    results_drug[ratio]["jacc_list"].append(jacc)
                if eval_view_np[i, 2]:
                    jacc = _jaccard(gt_tokens_drug[i], gen_tokens_drug[i])
                    results_drug[ratio]["jacc_list"].append(jacc)

    # ===== 汇总 =====
    def _finalize(res_dict, view_name):
        out = {}
        print(f"\n===== {view_name} mask ratio sweep (ALL views pooled) =====")
        for ratio, acc in res_dict.items():
            if acc["count"] == 0:
                print(f"[{view_name}] ratio={ratio:.2f}: 没有用于计算的 (原始存在 & mask==1) 位置。")
                out[ratio] = {"mae": None, "rmse": None, "jaccard": None, "count": 0}
                continue
            mae = acc["sum_abs"] / acc["count"]
            rmse = float(np.sqrt(acc["sum_sq"] / acc["count"]))
            jacc = float(np.mean(acc["jacc_list"])) if len(acc["jacc_list"]) > 0 else 0.0
            print(
                f"[{view_name}] ratio={ratio:.2f}: "
                f"MAE={mae:.6f}, RMSE={rmse:.6f}, Jaccard={jacc:.6f}, count={acc['count']}"
            )
            out[ratio] = {"mae": mae, "rmse": rmse, "jaccard": jacc, "count": acc["count"]}
        return out

    results_cond_final = _finalize(results_cond, "conditions-scan")
    results_proc_final = _finalize(results_proc, "procedures-scan")
    results_drug_final = _finalize(results_drug, "drugs-scan")

    return results_cond_final, results_proc_final, results_drug_final


def eval_diffusion_latent_mae_rmse_allviews1(
        nnet,
        diff_config,
        config_health,
        hp_dataset,  # [train_dataset, test_dataset]
        tokenizers,
        root_to,  # processed 目录，里面有 test_dataset.pkl
        sample_steps=50,
        # 按缺失比例：10% 20% 30% 40% 60% 70%
        cond_ratios=(0.1, 0.2, 0.3, 0.4, 0.6, 0.7),
        proc_ratios=(0.1, 0.2, 0.3, 0.4, 0.6, 0.7),
        drug_ratios=(0.1, 0.2, 0.3, 0.4, 0.6, 0.7),
        fully_mask_other_views=True,  # True: 其他存在真值的视图全部 mask
):
    """
    只在 “三视图都原始存在 (incomplete_raw == [0,0,0])” 的样本上做视图缺失比例扫描。

    - conditions-scan：当前轮控制 conditions 的缺失比例 (cond_ratios)，
        若 fully_mask_other_views=True，procedures/drugs 只要原始存在就全部 mask。
    - procedures-scan：同理；
    - drugs-scan：同理。

    对于每一次采样 (某个 view 的某个 ratio)：
      1) latent 空间：在所有「原始存在 & 本轮 mask==1」的视图上，计算整体 MAE / RMSE；
      2) 离散空间：
         - 先对每个 (样本, 视图) 计算 Jaccard；
         - 以 err = 1 - Jaccard 作为离散误差，做 MAE / RMSE；
         - 同时输出平均 Jaccard。
    """
    device = torch.device('cuda:' + str(diff_config['device'])
                          if torch.cuda.is_available() else 'cpu')
    nnet.eval()

    # 固定随机种子，保证每次 mask 一致
    seed = 528
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_n = config_health['DATASET']
    task = config_health['TASK']

    # ===== 1. 读 raw test_dataset.pkl，构建 visit_id -> incomplete_raw =====
    raw_test_path = os.path.join(root_to, "test_dataset.pkl")
    print(f"[mask_sweep] load raw test from {raw_test_path}")
    raw_test = _load_pickle(raw_test_path)

    def _get_inc_vec(s):
        inc = s.get("incomplete_raw", s.get("incomplete", None))
        if inc is None:
            return [0, 0, 0]
        if isinstance(inc, list) and len(inc) == 3 and isinstance(inc[0], int):
            return inc
        if isinstance(inc, list) and len(inc) == 1 and isinstance(inc[0], list):
            return inc[0]
        return [0, 0, 0]

    vid2inc = {s["visit_id"]: _get_inc_vec(s) for s in raw_test}

    # *** 只保留三视图都完整的样本 ***
    complete3_vids = {
        vid for vid, inc in vid2inc.items()
        if all(x == 0 for x in inc)
    }
    print(f"[mask_sweep] 原始 3 视图都完整的 test 样本数: {len(complete3_vids)}")

    if len(complete3_vids) == 0:
        print("[mask_sweep] 没有 3 视图都完整的样本，直接返回。")
        return {}, {}, {}

    # ===== 2. diffusion dataset & dataloader =====
    dataset = get_dataset(data_n, aug=True, samples=hp_dataset, config=diff_config)
    test_dataset = dataset.get_split(split='test', labeled=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=diff_config['sample']['mini_batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_dict,
    )

    # ===== 3. diffusion 准备 =====
    betas = utils.stable_diffusion_beta_schedule()
    N = len(betas)

    def convert_input_test(batch_data, mask_static=True):
        indices, batch = batch_data
        feature_keys = ['conditions', 'drugs', 'procedures']
        for feature_key in feature_keys:
            batch[feature_key] = tokenizers[feature_key].batch_encode_2d(batch[feature_key])  # B,M
            batch[feature_key + '_hist'] = tokenizers[feature_key].batch_encode_3d(
                batch[feature_key + '_hist']
            )  # B,V,M
            origin = batch[feature_key + '_comp'].copy()
            batch[feature_key + '_comp'] = tokenizers[feature_key].batch_encode_2d(
                batch[feature_key + '_comp']
            )  # B,M

            batch[feature_key] = torch.tensor(batch[feature_key], dtype=torch.long, device=device)
            batch[feature_key + '_hist'] = torch.tensor(
                batch[feature_key + '_hist'], dtype=torch.long, device=device
            )
            batch[feature_key + '_comp'] = torch.tensor(
                batch[feature_key + '_comp'], dtype=torch.long, device=device
            )

            batch[feature_key + '_mask'] = batch[feature_key] != 0  # B,M
            batch[feature_key + '_hist_mask'] = batch[feature_key + '_hist'] != 0  # B,V,M
            batch[feature_key + '_comp_mask'] = batch[feature_key + '_comp'] != 0  # B,M

            batch[feature_key + '_comps'] = tokenizers[feature_key].batch_encode_2d(
                origin, padding=False, truncation=False
            )
            batch[feature_key + '_comps_origin'] = origin

            batch['has_note'] = False  # MIII 不会用到 note

        if mask_static:
            batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
        else:
            m = 3
            batch['mask'] = utils.generate_mask(batch['conditions'].shape[0], m=m).to(device=device)

        aligned_data = [
            batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1)
            for feature_key in feature_keys
        ]
        batch['mask_hist'] = torch.cat(aligned_data, dim=-1)  # B,V,3

        return indices, batch

    def cfg_nnet(x, timesteps, context):
        _cond = nnet(x, timesteps, context=context)
        if diff_config['sample']['scale'] == 0:
            return _cond
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet(x, timesteps, context=_empty_context)
        return _cond + diff_config['sample']['scale'] * (_cond - _uncond)

    def decode_latent(z):
        """只用于 Jaccard（tokens 层面）"""
        nnet.eval()
        hard_samples = {}
        for idx, feature_key in enumerate(tokenizers.keys()):
            logits = nnet.get_logits(z[:, idx, :], feature_key)  # B, vocab_size
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            max_indices = np.argmax(prob, axis=1)

            thred = diff_config['threshod']
            prob_bin = (prob >= thred).astype(np.float32)

            indices_all = []
            for i, row in enumerate(prob_bin):
                idxs = np.where(row == 1)[0] + 2  # 跳过 pad/unk
                if len(idxs) < 1:
                    idxs = np.array([max_indices[i] + 2])
                indices_all.append(idxs.tolist())

            tokens = tokenizers[feature_key].batch_decode_2d(indices_all)
            hard_samples[feature_key] = tokens
        return hard_samples

    def dpm_solver_sample_latent(z_init, context, mask_view, _sample_steps):
        B = z_init.size(0)
        z_noisy = torch.randn(B, *diff_config['z_shape'], device=device)
        mask_latent = mask_view.unsqueeze(-1).repeat(1, 1, diff_config['z_shape'][1])  # B,3,D
        z_start = torch.where(mask_latent, z_noisy, z_init)

        noise_schedule = NoiseScheduleVP(
            schedule='discrete',
            betas=torch.tensor(betas, device=device).float()
        )

        def model_fn(x, t_continuous):
            t = t_continuous * N
            return cfg_nnet(x, t, context)

        dpm_solver = DPM_Solver(
            model_fn,
            noise_schedule,
            predict_x0=True,
            thresholding=False
        )

        z_recon = dpm_solver.sample(
            z_start, steps=_sample_steps, eps=1. / N, T=1.
        )
        hard_samples = decode_latent(z_recon)
        return z_recon, hard_samples

    # ===== 4. 为每个 ratio 准备累加器（全视图 pooled）=====
    def _init_res(ratios):
        return {
            r: {
                "lat_sum_abs": 0.0, "lat_sum_sq": 0.0, "lat_count": 0,
                "disc_sum_abs": 0.0, "disc_sum_sq": 0.0, "disc_count": 0,
                "jacc_list": []
            }
            for r in ratios
        }

    results_cond = _init_res(cond_ratios)
    results_proc = _init_res(proc_ratios)
    results_drug = _init_res(drug_ratios)

    # ===== 5. 在 test_loader 上循环 =====
    for indices, batch in test_loader:
        # 只保留 “三视图都完整” 的样本
        vid_list = batch["visit_id"]
        keep = [vid in complete3_vids for vid in vid_list]
        if not any(keep):
            continue

        keep_idx = torch.tensor(keep, dtype=torch.bool)
        indices = indices[keep_idx]

        def _select_field(x):
            if isinstance(x, torch.Tensor):
                return x[keep_idx.to(x.device)]
            elif isinstance(x, list):
                return [v for v, f in zip(x, keep) if f]
            else:
                return x

        batch = {k: _select_field(v) for k, v in batch.items()}
        if indices.numel() == 0:
            continue

        batch_vids = batch["visit_id"]
        # 对 complete3_vids 来说，inc_tensor 全 0，orig_complete_view 全 True
        inc_list = [vid2inc[vid] for vid in batch_vids]
        inc_tensor = torch.tensor(inc_list, dtype=torch.bool, device=device)  # B,3
        orig_complete_view = ~inc_tensor  # B,3

        B = len(batch_vids)

        indices, sample_input = convert_input_test((indices, batch))

        z_init, context_emb, _, _, hard_labels = nnet.encode_everything_test(
            indices, **batch
        )

        # ground truth tokens（用于 Jaccard）
        gt_tokens_cond = tokenizers["conditions"].batch_decode_2d(hard_labels["conditions"])
        gt_tokens_proc = tokenizers["procedures"].batch_decode_2d(hard_labels["procedures"])
        gt_tokens_drug = tokenizers["drugs"].batch_decode_2d(hard_labels["drugs"])

        # ===== conditions 扫描（全视图 pooled）=====
        for ratio in cond_ratios:
            if ratio <= 0.0:
                continue

            mask_view = torch.zeros(B, 3, dtype=torch.bool, device=device)
            valid_cond = orig_complete_view[:, 0]  # B
            rand = torch.rand(B, device=device)
            mask_view[:, 0] = (rand < ratio) & valid_cond

            if fully_mask_other_views:
                mask_view[:, 1] = orig_complete_view[:, 1]
                mask_view[:, 2] = orig_complete_view[:, 2]

            if mask_view.sum() == 0:
                continue

            z_recon, hard_samples = dpm_solver_sample_latent(
                z_init, context_emb, mask_view, sample_steps
            )

            eval_view_mask = mask_view & orig_complete_view  # B,3

            # --- 连续 latent MAE/RMSE ---
            diff = z_recon - z_init  # B,3,D
            mask_latent = eval_view_mask.unsqueeze(-1).expand_as(diff)
            diff_sel = diff[mask_latent]

            if diff_sel.numel() > 0:
                results_cond[ratio]["lat_sum_abs"] += diff_sel.abs().sum().item()
                results_cond[ratio]["lat_sum_sq"] += (diff_sel ** 2).sum().item()
                results_cond[ratio]["lat_count"] += diff_sel.numel()

            # --- 离散：Jaccard & 1-Jaccard 的 MAE/RMSE（视图级）---
            eval_view_np = eval_view_mask.detach().cpu().numpy()
            gen_tokens_cond = hard_samples["conditions"]
            gen_tokens_proc = hard_samples["procedures"]
            gen_tokens_drug = hard_samples["drugs"]

            for i in range(B):
                # view 0: conditions
                if eval_view_np[i, 0]:
                    jacc = _jaccard(gt_tokens_cond[i], gen_tokens_cond[i])
                    results_cond[ratio]["jacc_list"].append(jacc)
                    err = 1.0 - jacc
                    results_cond[ratio]["disc_sum_abs"] += err
                    results_cond[ratio]["disc_sum_sq"] += err * err
                    results_cond[ratio]["disc_count"] += 1
                # view 1: procedures
                if eval_view_np[i, 1]:
                    jacc = _jaccard(gt_tokens_proc[i], gen_tokens_proc[i])
                    results_cond[ratio]["jacc_list"].append(jacc)
                    err = 1.0 - jacc
                    results_cond[ratio]["disc_sum_abs"] += err
                    results_cond[ratio]["disc_sum_sq"] += err * err
                    results_cond[ratio]["disc_count"] += 1
                # view 2: drugs
                if eval_view_np[i, 2]:
                    jacc = _jaccard(gt_tokens_drug[i], gen_tokens_drug[i])
                    results_cond[ratio]["jacc_list"].append(jacc)
                    err = 1.0 - jacc
                    results_cond[ratio]["disc_sum_abs"] += err
                    results_cond[ratio]["disc_sum_sq"] += err * err
                    results_cond[ratio]["disc_count"] += 1

        # ===== procedures 扫描 =====
        for ratio in proc_ratios:
            if ratio <= 0.0:
                continue

            mask_view = torch.zeros(B, 3, dtype=torch.bool, device=device)
            valid_proc = orig_complete_view[:, 1]
            rand = torch.rand(B, device=device)
            mask_view[:, 1] = (rand < ratio) & valid_proc

            if fully_mask_other_views:
                mask_view[:, 0] = orig_complete_view[:, 0]
                mask_view[:, 2] = orig_complete_view[:, 2]

            if mask_view.sum() == 0:
                continue

            z_recon, hard_samples = dpm_solver_sample_latent(
                z_init, context_emb, mask_view, sample_steps
            )

            eval_view_mask = mask_view & orig_complete_view  # B,3

            diff = z_recon - z_init
            mask_latent = eval_view_mask.unsqueeze(-1).expand_as(diff)
            diff_sel = diff[mask_latent]

            if diff_sel.numel() > 0:
                results_proc[ratio]["lat_sum_abs"] += diff_sel.abs().sum().item()
                results_proc[ratio]["lat_sum_sq"] += (diff_sel ** 2).sum().item()
                results_proc[ratio]["lat_count"] += diff_sel.numel()

            eval_view_np = eval_view_mask.detach().cpu().numpy()
            gen_tokens_cond = hard_samples["conditions"]
            gen_tokens_proc = hard_samples["procedures"]
            gen_tokens_drug = hard_samples["drugs"]

            for i in range(B):
                if eval_view_np[i, 0]:
                    jacc = _jaccard(gt_tokens_cond[i], gen_tokens_cond[i])
                    results_proc[ratio]["jacc_list"].append(jacc)
                    err = 1.0 - jacc
                    results_proc[ratio]["disc_sum_abs"] += err
                    results_proc[ratio]["disc_sum_sq"] += err * err
                    results_proc[ratio]["disc_count"] += 1
                if eval_view_np[i, 1]:
                    jacc = _jaccard(gt_tokens_proc[i], gen_tokens_proc[i])
                    results_proc[ratio]["jacc_list"].append(jacc)
                    err = 1.0 - jacc
                    results_proc[ratio]["disc_sum_abs"] += err
                    results_proc[ratio]["disc_sum_sq"] += err * err
                    results_proc[ratio]["disc_count"] += 1
                if eval_view_np[i, 2]:
                    jacc = _jaccard(gt_tokens_drug[i], gen_tokens_drug[i])
                    results_proc[ratio]["jacc_list"].append(jacc)
                    err = 1.0 - jacc
                    results_proc[ratio]["disc_sum_abs"] += err
                    results_proc[ratio]["disc_sum_sq"] += err * err
                    results_proc[ratio]["disc_count"] += 1

        # ===== drugs 扫描 =====
        for ratio in drug_ratios:
            if ratio <= 0.0:
                continue

            mask_view = torch.zeros(B, 3, dtype=torch.bool, device=device)
            valid_drug = orig_complete_view[:, 2]
            rand = torch.rand(B, device=device)
            mask_view[:, 2] = (rand < ratio) & valid_drug

            if fully_mask_other_views:
                mask_view[:, 0] = orig_complete_view[:, 0]
                mask_view[:, 1] = orig_complete_view[:, 1]

            if mask_view.sum() == 0:
                continue

            z_recon, hard_samples = dpm_solver_sample_latent(
                z_init, context_emb, mask_view, sample_steps
            )

            eval_view_mask = mask_view & orig_complete_view  # B,3

            diff = z_recon - z_init
            mask_latent = eval_view_mask.unsqueeze(-1).expand_as(diff)
            diff_sel = diff[mask_latent]

            if diff_sel.numel() > 0:
                results_drug[ratio]["lat_sum_abs"] += diff_sel.abs().sum().item()
                results_drug[ratio]["lat_sum_sq"] += (diff_sel ** 2).sum().item()
                results_drug[ratio]["lat_count"] += diff_sel.numel()

            eval_view_np = eval_view_mask.detach().cpu().numpy()
            gen_tokens_cond = hard_samples["conditions"]
            gen_tokens_proc = hard_samples["procedures"]
            gen_tokens_drug = hard_samples["drugs"]

            for i in range(B):
                if eval_view_np[i, 0]:
                    jacc = _jaccard(gt_tokens_cond[i], gen_tokens_cond[i])
                    results_drug[ratio]["jacc_list"].append(jacc)
                    err = 1.0 - jacc
                    results_drug[ratio]["disc_sum_abs"] += err
                    results_drug[ratio]["disc_sum_sq"] += err * err
                    results_drug[ratio]["disc_count"] += 1
                if eval_view_np[i, 1]:
                    jacc = _jaccard(gt_tokens_proc[i], gen_tokens_proc[i])
                    results_drug[ratio]["jacc_list"].append(jacc)
                    err = 1.0 - jacc
                    results_drug[ratio]["disc_sum_abs"] += err
                    results_drug[ratio]["disc_sum_sq"] += err * err
                    results_drug[ratio]["disc_count"] += 1
                if eval_view_np[i, 2]:
                    jacc = _jaccard(gt_tokens_drug[i], gen_tokens_drug[i])
                    results_drug[ratio]["jacc_list"].append(jacc)
                    err = 1.0 - jacc
                    results_drug[ratio]["disc_sum_abs"] += err
                    results_drug[ratio]["disc_sum_sq"] += err * err
                    results_drug[ratio]["disc_count"] += 1

    # ===== 汇总 =====
    # ===== 汇总 =====
    def _finalize(res_dict, view_name):
        out = {}
        print(f"\n===== {view_name} mask ratio sweep (3 views complete, ALL views pooled) =====")
        for ratio, acc in res_dict.items():
            # ratio 安全转成字符串
            try:
                ratio_str = f"{float(ratio):.2f}"
            except (TypeError, ValueError):
                ratio_str = str(ratio)

            if acc["lat_count"] == 0:
                print(f"[{view_name}] ratio={ratio_str}: 没有用于计算的 (原始存在 & mask==1) 位置。")
                out[ratio] = {
                    "latent_mae": None,
                    "latent_rmse": None,
                    "discrete_mae": None,
                    "discrete_rmse": None,
                    "jaccard": None,
                    "latent_count": 0,
                    "discrete_count": 0,
                }
                continue

            # 连续 latent 误差
            lat_mae = acc["lat_sum_abs"] / acc["lat_count"]
            lat_rmse = float(np.sqrt(acc["lat_sum_sq"] / acc["lat_count"]))

            # 离散 1-Jaccard 误差
            if acc["disc_count"] > 0:
                disc_mae = acc["disc_sum_abs"] / acc["disc_count"]
                disc_rmse = float(np.sqrt(acc["disc_sum_sq"] / acc["disc_count"]))
            else:
                disc_mae, disc_rmse = None, None

            # 平均 Jaccard
            jacc = float(np.mean(acc["jacc_list"])) if len(acc["jacc_list"]) > 0 else None

            # 先转成字符串，避免 None 和格式化冲突
            lat_mae_str = f"{lat_mae:.6f}"
            lat_rmse_str = f"{lat_rmse:.6f}"
            disc_mae_str = f"{disc_mae:.6f}" if disc_mae is not None else "NA"
            disc_rmse_str = f"{disc_rmse:.6f}" if disc_rmse is not None else "NA"
            jacc_str = f"{jacc:.6f}" if jacc is not None else "NA"

            print(
                f"[{view_name}] ratio={ratio_str}: "
                f"latent_MAE={lat_mae_str}, latent_RMSE={lat_rmse_str}, "
                f"disc_MAE={disc_mae_str}, disc_RMSE={disc_rmse_str}, "
                f"Jaccard={jacc_str}, "
                f"latent_count={acc['lat_count']}, disc_count={acc['disc_count']}"
            )

            out[ratio] = {
                "latent_mae": lat_mae,
                "latent_rmse": lat_rmse,
                "discrete_mae": disc_mae,
                "discrete_rmse": disc_rmse,
                "jaccard": jacc,
                "latent_count": acc["lat_count"],
                "discrete_count": acc["disc_count"],
            }
        return out

    results_cond_final = _finalize(results_cond, "conditions-scan")
    results_proc_final = _finalize(results_proc, "procedures-scan")
    results_drug_final = _finalize(results_drug, "drugs-scan")

    return results_cond_final, results_proc_final, results_drug_final


def plot_latent_sq_error_density(latent_sq_dict, title="", num_bins=200,
                                 max_x=None, log_x=False, log_y=False,
                                 save_path=None):
    """
    latent_sq_dict: dict[ratio -> np.ndarray]，
        其中 array 里是所有 (z_recon - z_init)^2 的标量。
    """
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))

    for ratio in sorted(latent_sq_dict.keys()):
        data = latent_sq_dict[ratio]
        if data is None or len(data) == 0:
            continue

        data = np.asarray(data).astype(float)
        if max_x is not None:
            data = data[data <= max_x]
        if data.size == 0:
            continue

        counts, bin_edges = np.histogram(data, bins=num_bins, density=True)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        plt.plot(centers, counts, label=f"ratio={ratio:.1f}")

    plt.xlabel(r"(y - $\hat{y}$)$^2$")
    plt.ylabel("Density")
    plt.title(title)
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print(f"saved to {save_path}")
    else:
        plt.show()

def plot_sq_error_hist_for_scan(abs_err_by_ratio,
                                    title='latent absolute error (multi-ratio)',
                                    use_prob=False,
                                    save_path=None):
    """
    abs_err_by_ratio: dict[ratio -> np.ndarray of |z - z_hat|]
    use_prob: False -> 纵轴 Count; True -> 纵轴比例
    """

    # 自定义区间
    bins = np.array([0, 5, 10, 15, 20, 25, 30., 35, 40, 45, 50], dtype=float)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bin_widths = (bins[1:] - bins[:-1])

    ratios = sorted(abs_err_by_ratio.keys())
    if len(ratios) == 0:
        print("[plot] abs_err_by_ratio is empty.")
        return

    plt.figure(figsize=(6, 4))

    # 并排柱：每个bin内按ratio并排
    R = len(ratios)
    # group_width = 0.85  # 每个bin内柱子整体占用比例
    group_width = 0.85  # 每个bin内柱子整体占用比例
    # 因为bin宽度都是5，这里用“相对bin宽度”的方式设置每根柱宽
    base_w = bin_widths.min()  # 这里=5
    bar_w = (base_w * group_width) / R

    # offsets: 让多根柱子围绕bin_center左右展开
    offsets = (np.arange(R) - (R - 1) / 2.0) * bar_w

    ylabel = 'Proportion' if use_prob else 'Count'

    for i, ratio in enumerate(ratios):
        abs_err = np.asarray(abs_err_by_ratio[ratio], dtype=float)
        if abs_err.size == 0:
            continue

        # 截断到 [0, 50) 防止极端值拉坏横轴
        abs_err = np.clip(abs_err, bins[0], bins[-1] - 1e-8)

        counts, _ = np.histogram(abs_err, bins=bins)

        if use_prob:
            total = counts.sum()
            y = counts / total if total > 0 else counts.astype(float)
        else:
            y = counts

        x = bin_centers + offsets[i]
        plt.bar(x, y, width=bar_w, align='center', label=f'ratio={ratio}')

    # x轴用“区间标签”更直观（你也可以改回 plt.xticks(bins)）
    interval_labels = [f'({int(bins[j])},{int(bins[j + 1])}]' for j in range(len(bins) - 1)]
    plt.xticks(bin_centers, interval_labels, rotation=30, ha='right')

    plt.xlabel(r'$|z - \hat{z}|$ bins (latent absolute error)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=2)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()

def eval_diffusion_latent_mae_rmse_allviews(
        nnet,
        diff_config,
        config_health,
        hp_dataset,  # [train_dataset, test_dataset]
        tokenizers,
        root_to,  # processed 目录，里面有 test_dataset.pkl
        sample_steps=50,
        # 按缺失比例：10% 20% 30% 40% 60% 70%
        cond_ratios=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        proc_ratios=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        drug_ratios=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        fully_mask_other_views=False,  # True: 其他存在真值的视图全部 mask
):

        device = torch.device('cuda:' + str(diff_config['device'])
                              if torch.cuda.is_available() else 'cpu')
        nnet.eval()

        # 固定随机种子，保证每次 mask 一致
        seed = 528
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        data_n = config_health['DATASET']
        task = config_health['TASK']

        # ===== 1. 读 raw test_dataset.pkl，构建 visit_id -> incomplete_raw =====
        raw_test_path = os.path.join(root_to, "test_dataset.pkl")
        print(f"[mask_sweep] load raw test from {raw_test_path}")
        raw_test = _load_pickle(raw_test_path)

        def _get_inc_vec(s):
            inc = s.get("incomplete_raw", s.get("incomplete", None))
            if inc is None:
                return [0, 0, 0]
            if isinstance(inc, list) and len(inc) == 3 and isinstance(inc[0], int):
                return inc
            if isinstance(inc, list) and len(inc) == 1 and isinstance(inc[0], list):
                return inc[0]
            return [0, 0, 0]

        vid2inc = {s["visit_id"]: _get_inc_vec(s) for s in raw_test}

        # *** 只保留三视图都完整的样本 ***
        complete3_vids = {
            vid for vid, inc in vid2inc.items()
            if all(x == 0 for x in inc)
        }
        print(f"[mask_sweep] 原始 3 视图都完整的 test 样本数: {len(complete3_vids)}")

        if len(complete3_vids) == 0:
            print("[mask_sweep] 没有 3 视图都完整的样本，直接返回。")
            return {}, {}, {}, {}, {}, {}

        # ===== 2. diffusion dataset & dataloader =====
        dataset = get_dataset(data_n, aug=True, samples=hp_dataset, config=diff_config)
        test_dataset = dataset.get_split(split='test', labeled=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=diff_config['sample']['mini_batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn_dict,
        )

        # ===== 3. diffusion 准备 =====
        betas = utils.stable_diffusion_beta_schedule()
        N = len(betas)

        def convert_input_test(batch_data, mask_static=True):
            indices, batch = batch_data
            feature_keys = ['conditions', 'drugs', 'procedures']
            for feature_key in feature_keys:
                batch[feature_key] = tokenizers[feature_key].batch_encode_2d(batch[feature_key])  # B,M
                batch[feature_key + '_hist'] = tokenizers[feature_key].batch_encode_3d(
                    batch[feature_key + '_hist']
                )  # B,V,M
                origin = batch[feature_key + '_comp'].copy()
                batch[feature_key + '_comp'] = tokenizers[feature_key].batch_encode_2d(
                    batch[feature_key + '_comp']
                )  # B,M

                batch[feature_key] = torch.tensor(batch[feature_key], dtype=torch.long, device=device)
                batch[feature_key + '_hist'] = torch.tensor(
                    batch[feature_key + '_hist'], dtype=torch.long, device=device
                )
                batch[feature_key + '_comp'] = torch.tensor(
                    batch[feature_key + '_comp'], dtype=torch.long, device=device
                )

                batch[feature_key + '_mask'] = batch[feature_key] != 0  # B,M
                batch[feature_key + '_hist_mask'] = batch[feature_key + '_hist'] != 0  # B,V,M
                batch[feature_key + '_comp_mask'] = batch[feature_key + '_comp'] != 0  # B,M

                batch[feature_key + '_comps'] = tokenizers[feature_key].batch_encode_2d(
                    origin, padding=False, truncation=False
                )
                batch[feature_key + '_comps_origin'] = origin

                batch['has_note'] = False  # MIII 不会用到 note

            if mask_static:
                batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
            else:
                m = 3
                batch['mask'] = utils.generate_mask(batch['conditions'].shape[0], m=m).to(device=device)

            aligned_data = [
                batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1)
                for feature_key in feature_keys
            ]
            batch['mask_hist'] = torch.cat(aligned_data, dim=-1)  # B,V,3

            return indices, batch

        def cfg_nnet(x, timesteps, context):
            _cond = nnet(x, timesteps, context=context)
            if diff_config['sample']['scale'] == 0:
                return _cond
            _empty_context = torch.tensor(dataset.empty_context, device=device)
            _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
            _uncond = nnet(x, timesteps, context=_empty_context)
            return _cond + diff_config['sample']['scale'] * (_cond - _uncond)

        def decode_latent(z):
            """只用于 Jaccard（tokens 层面），可选 prior boost"""
            prior_data = diff_config.get("prior_data", None)
            use_prior = bool(prior_data is not None and diff_config.get("sample", {}).get("use_prior", False))

            if not use_prior:
                nnet.eval()
                hard_samples = {}
                for idx, feature_key in enumerate(tokenizers.keys()):
                    logits = nnet.get_logits(z[:, idx, :], feature_key)
                    prob = torch.sigmoid(logits).detach().cpu().numpy()
                    max_indices = np.argmax(prob, axis=1)
                    thred = diff_config['threshod']
                    prob_bin = (prob >= thred).astype(np.float32)
                    indices_all = []
                    for i, row in enumerate(prob_bin):
                        idxs = np.where(row == 1)[0] + 2
                        if len(idxs) < 1:
                            idxs = np.array([max_indices[i] + 2])
                        indices_all.append(idxs.tolist())
                    tokens = tokenizers[feature_key].batch_decode_2d(indices_all)
                    hard_samples[feature_key] = tokens
                return hard_samples

            # --- prior 版本 ---
            from prior_utils import apply_prior_decode_np
            return apply_prior_decode_np(z, nnet, tokenizers, diff_config)

        def dpm_solver_sample_latent(z_init, context, mask_view, _sample_steps):
            B = z_init.size(0)
            z_noisy = torch.randn(B, *diff_config['z_shape'], device=device)
            mask_latent = mask_view.unsqueeze(-1).repeat(1, 1, diff_config['z_shape'][1])  # B,3,D
            z_start = torch.where(mask_latent, z_noisy, z_init)

            noise_schedule = NoiseScheduleVP(
                schedule='discrete',
                betas=torch.tensor(betas, device=device).float()
            )

            def model_fn(x, t_continuous):
                t = t_continuous * N
                return cfg_nnet(x, t, context)

            dpm_solver = DPM_Solver(
                model_fn,
                noise_schedule,
                predict_x0=True,
                thresholding=False
            )

            z_recon = dpm_solver.sample(
                z_start, steps=_sample_steps, eps=1. / N, T=1.
            )
            hard_samples = decode_latent(z_recon)
            return z_recon, hard_samples

        # ===== 4. 为每个 ratio 准备累加器（全视图 pooled）=====
        def _init_res(ratios):
            return {
                r: {
                    "lat_sum_abs": 0.0, "lat_sum_sq": 0.0, "lat_count": 0,
                    "disc_sum_abs": 0.0, "disc_sum_sq": 0.0, "disc_count": 0,
                    "jacc_list": []
                }
                for r in ratios
            }

        results_cond = _init_res(cond_ratios)
        results_proc = _init_res(proc_ratios)
        results_drug = _init_res(drug_ratios)

        # 额外保存每个 ratio 的所有 |z - z_hat|
        cond_latent_abs_err = {r: [] for r in cond_ratios}
        proc_latent_abs_err = {r: [] for r in proc_ratios}
        drug_latent_abs_err = {r: [] for r in drug_ratios}

        # ===== 5. 在 test_loader 上循环 =====
        for indices, batch in test_loader:
            # 只保留 “三视图都完整” 的样本
            vid_list = batch["visit_id"]
            keep = [vid in complete3_vids for vid in vid_list]
            if not any(keep):
                continue

            keep_idx = torch.tensor(keep, dtype=torch.bool)
            indices = indices[keep_idx]

            def _select_field(x):
                if isinstance(x, torch.Tensor):
                    return x[keep_idx.to(x.device)]
                elif isinstance(x, list):
                    return [v for v, f in zip(x, keep) if f]
                else:
                    return x

            batch = {k: _select_field(v) for k, v in batch.items()}
            if indices.numel() == 0:
                continue

            batch_vids = batch["visit_id"]
            inc_list = [vid2inc[vid] for vid in batch_vids]
            inc_tensor = torch.tensor(inc_list, dtype=torch.bool, device=device)  # B,3
            orig_complete_view = ~inc_tensor  # B,3 (这里应该全 True)

            B = len(batch_vids)

            indices, sample_input = convert_input_test((indices, batch))

            z_init, context_emb, _, _, hard_labels = nnet.encode_everything_test(
                indices, **batch
            )

            # ground truth tokens（用于 Jaccard）
            gt_tokens_cond = tokenizers["conditions"].batch_decode_2d(hard_labels["conditions"])
            gt_tokens_proc = tokenizers["procedures"].batch_decode_2d(hard_labels["procedures"])
            gt_tokens_drug = tokenizers["drugs"].batch_decode_2d(hard_labels["drugs"])

            # ===== conditions 扫描（全视图 pooled）=====
            for ratio in cond_ratios:
                if ratio <= 0.0:
                    continue

                mask_view = torch.zeros(B, 3, dtype=torch.bool, device=device)
                valid_cond = orig_complete_view[:, 0]  # B
                rand = torch.rand(B, device=device)
                mask_view[:, 0] = (rand < ratio) & valid_cond

                if fully_mask_other_views:
                    mask_view[:, 1] = orig_complete_view[:, 1]
                    mask_view[:, 2] = orig_complete_view[:, 2]

                if mask_view.sum() == 0:
                    continue

                z_recon, hard_samples = dpm_solver_sample_latent(
                    z_init, context_emb, mask_view, sample_steps
                )

                eval_view_mask = mask_view & orig_complete_view  # B,3

                # --- 连续 latent MAE/RMSE ---
                diff = z_recon - z_init  # B,3,D
                mask_latent = eval_view_mask.unsqueeze(-1).expand_as(diff)
                diff_sel = diff[mask_latent]

                if diff_sel.numel() > 0:
                    abs_err_np = diff_sel.abs().detach().cpu().numpy().reshape(-1)
                    cond_latent_abs_err[ratio].append(abs_err_np)

                    results_cond[ratio]["lat_sum_abs"] += abs_err_np.sum()
                    results_cond[ratio]["lat_sum_sq"] += (abs_err_np ** 2).sum()
                    results_cond[ratio]["lat_count"] += abs_err_np.size

                # --- 离散：Jaccard & 1-Jaccard 的 MAE/RMSE（视图级）---
                eval_view_np = eval_view_mask.detach().cpu().numpy()
                gen_tokens_cond = hard_samples["conditions"]
                gen_tokens_proc = hard_samples["procedures"]
                gen_tokens_drug = hard_samples["drugs"]

                for i in range(B):
                    # view 0: conditions
                    if eval_view_np[i, 0]:
                        jacc = _jaccard(gt_tokens_cond[i], gen_tokens_cond[i])
                        results_cond[ratio]["jacc_list"].append(jacc)
                        err = 1.0 - jacc
                        results_cond[ratio]["disc_sum_abs"] += err
                        results_cond[ratio]["disc_sum_sq"] += err * err
                        results_cond[ratio]["disc_count"] += 1
                    # view 1: procedures
                    if eval_view_np[i, 1]:
                        jacc = _jaccard(gt_tokens_proc[i], gen_tokens_proc[i])
                        results_cond[ratio]["jacc_list"].append(jacc)
                        err = 1.0 - jacc
                        results_cond[ratio]["disc_sum_abs"] += err
                        results_cond[ratio]["disc_sum_sq"] += err * err
                        results_cond[ratio]["disc_count"] += 1
                    # view 2: drugs
                    if eval_view_np[i, 2]:
                        jacc = _jaccard(gt_tokens_drug[i], gen_tokens_drug[i])
                        results_cond[ratio]["jacc_list"].append(jacc)
                        err = 1.0 - jacc
                        results_cond[ratio]["disc_sum_abs"] += err
                        results_cond[ratio]["disc_sum_sq"] += err * err
                        results_cond[ratio]["disc_count"] += 1

            # ===== procedures 扫描 =====
            for ratio in proc_ratios:
                if ratio <= 0.0:
                    continue

                mask_view = torch.zeros(B, 3, dtype=torch.bool, device=device)
                valid_proc = orig_complete_view[:, 1]
                rand = torch.rand(B, device=device)
                mask_view[:, 1] = (rand < ratio) & valid_proc

                if fully_mask_other_views:
                    mask_view[:, 0] = orig_complete_view[:, 0]
                    mask_view[:, 2] = orig_complete_view[:, 2]

                if mask_view.sum() == 0:
                    continue

                z_recon, hard_samples = dpm_solver_sample_latent(
                    z_init, context_emb, mask_view, sample_steps
                )

                eval_view_mask = mask_view & orig_complete_view  # B,3

                diff = z_recon - z_init
                mask_latent = eval_view_mask.unsqueeze(-1).expand_as(diff)
                diff_sel = diff[mask_latent]

                if diff_sel.numel() > 0:
                    abs_err_np = diff_sel.abs().detach().cpu().numpy().reshape(-1)
                    proc_latent_abs_err[ratio].append(abs_err_np)

                    results_proc[ratio]["lat_sum_abs"] += abs_err_np.sum()
                    results_proc[ratio]["lat_sum_sq"] += (abs_err_np ** 2).sum()
                    results_proc[ratio]["lat_count"] += abs_err_np.size

                eval_view_np = eval_view_mask.detach().cpu().numpy()
                gen_tokens_cond = hard_samples["conditions"]
                gen_tokens_proc = hard_samples["procedures"]
                gen_tokens_drug = hard_samples["drugs"]

                for i in range(B):
                    if eval_view_np[i, 0]:
                        jacc = _jaccard(gt_tokens_cond[i], gen_tokens_cond[i])
                        results_proc[ratio]["jacc_list"].append(jacc)
                        err = 1.0 - jacc
                        results_proc[ratio]["disc_sum_abs"] += err
                        results_proc[ratio]["disc_sum_sq"] += err * err
                        results_proc[ratio]["disc_count"] += 1
                    if eval_view_np[i, 1]:
                        jacc = _jaccard(gt_tokens_proc[i], gen_tokens_proc[i])
                        results_proc[ratio]["jacc_list"].append(jacc)
                        err = 1.0 - jacc
                        results_proc[ratio]["disc_sum_abs"] += err
                        results_proc[ratio]["disc_sum_sq"] += err * err
                        results_proc[ratio]["disc_count"] += 1
                    if eval_view_np[i, 2]:
                        jacc = _jaccard(gt_tokens_drug[i], gen_tokens_drug[i])
                        results_proc[ratio]["jacc_list"].append(jacc)
                        err = 1.0 - jacc
                        results_proc[ratio]["disc_sum_abs"] += err
                        results_proc[ratio]["disc_sum_sq"] += err * err
                        results_proc[ratio]["disc_count"] += 1

            # ===== drugs 扫描 =====
            for ratio in drug_ratios:
                if ratio <= 0.0:
                    continue

                mask_view = torch.zeros(B, 3, dtype=torch.bool, device=device)
                valid_drug = orig_complete_view[:, 2]
                rand = torch.rand(B, device=device)
                mask_view[:, 2] = (rand < ratio) & valid_drug

                if fully_mask_other_views:
                    mask_view[:, 0] = orig_complete_view[:, 0]
                    mask_view[:, 1] = orig_complete_view[:, 1]

                if mask_view.sum() == 0:
                    continue

                z_recon, hard_samples = dpm_solver_sample_latent(
                    z_init, context_emb, mask_view, sample_steps
                )

                eval_view_mask = mask_view & orig_complete_view  # B,3

                diff = z_recon - z_init
                mask_latent = eval_view_mask.unsqueeze(-1).expand_as(diff)
                diff_sel = diff[mask_latent]

                if diff_sel.numel() > 0:
                    abs_err_np = diff_sel.abs().detach().cpu().numpy().reshape(-1)
                    drug_latent_abs_err[ratio].append(abs_err_np)

                    results_drug[ratio]["lat_sum_abs"] += abs_err_np.sum()
                    results_drug[ratio]["lat_sum_sq"] += (abs_err_np ** 2).sum()
                    results_drug[ratio]["lat_count"] += abs_err_np.size

                eval_view_np = eval_view_mask.detach().cpu().numpy()
                gen_tokens_cond = hard_samples["conditions"]
                gen_tokens_proc = hard_samples["procedures"]
                gen_tokens_drug = hard_samples["drugs"]

                for i in range(B):
                    if eval_view_np[i, 0]:
                        jacc = _jaccard(gt_tokens_cond[i], gen_tokens_cond[i])
                        results_drug[ratio]["jacc_list"].append(jacc)
                        err = 1.0 - jacc
                        results_drug[ratio]["disc_sum_abs"] += err
                        results_drug[ratio]["disc_sum_sq"] += err * err
                        results_drug[ratio]["disc_count"] += 1
                    if eval_view_np[i, 1]:
                        jacc = _jaccard(gt_tokens_proc[i], gen_tokens_proc[i])
                        results_drug[ratio]["jacc_list"].append(jacc)
                        err = 1.0 - jacc
                        results_drug[ratio]["disc_sum_abs"] += err
                        results_drug[ratio]["disc_sum_sq"] += err * err
                        results_drug[ratio]["disc_count"] += 1
                    if eval_view_np[i, 2]:
                        jacc = _jaccard(gt_tokens_drug[i], gen_tokens_drug[i])
                        results_drug[ratio]["jacc_list"].append(jacc)
                        err = 1.0 - jacc
                        results_drug[ratio]["disc_sum_abs"] += err
                        results_drug[ratio]["disc_sum_sq"] += err * err
                        results_drug[ratio]["disc_count"] += 1

        # ===== 把每个 ratio 的误差 list concat 成一维 array =====
        for r in cond_ratios:
            if len(cond_latent_abs_err[r]) > 0:
                cond_latent_abs_err[r] = np.concatenate(cond_latent_abs_err[r], axis=0)
            else:
                cond_latent_abs_err[r] = np.array([], dtype=float)

        for r in proc_ratios:
            if len(proc_latent_abs_err[r]) > 0:
                proc_latent_abs_err[r] = np.concatenate(proc_latent_abs_err[r], axis=0)
            else:
                proc_latent_abs_err[r] = np.array([], dtype=float)

        for r in drug_ratios:
            if len(drug_latent_abs_err[r]) > 0:
                drug_latent_abs_err[r] = np.concatenate(drug_latent_abs_err[r], axis=0)
            else:
                drug_latent_abs_err[r] = np.array([], dtype=float)

        # ===== 汇总 =====
        def _finalize(res_dict, view_name):
            out = {}
            print(f"\n===== {view_name} mask ratio sweep (3 views complete, ALL views pooled) =====")
            for ratio, acc in res_dict.items():
                if acc["lat_count"] == 0:
                    print(f"[{view_name}] ratio={ratio:.2f}: 没有用于计算的 (原始存在 & mask==1) 位置。")
                    out[ratio] = {
                        "latent_mae": None, "latent_rmse": None,
                        "discrete_mae": None, "discrete_rmse": None,
                        "jaccard": None,
                        "latent_count": 0, "discrete_count": 0,
                    }
                    continue

                lat_mae = acc["lat_sum_abs"] / acc["lat_count"]
                lat_rmse = float(np.sqrt(acc["lat_sum_sq"] / acc["lat_count"]))

                if acc["disc_count"] > 0:
                    disc_mae = acc["disc_sum_abs"] / acc["disc_count"]
                    disc_rmse = float(np.sqrt(acc["disc_sum_sq"] / acc["disc_count"]))
                else:
                    disc_mae, disc_rmse = None, None

                jacc = float(np.mean(acc["jacc_list"])) if len(acc["jacc_list"]) > 0 else None

                disc_mae_str = f"{disc_mae:.6f}" if disc_mae is not None else "NA"
                disc_rmse_str = f"{disc_rmse:.6f}" if disc_rmse is not None else "NA"
                jacc_str = f"{jacc:.6f}" if jacc is not None else "NA"

                print(
                    f"[{view_name}] ratio={ratio:.2f}: "
                    f"latent_MAE={lat_mae:.6f}, latent_RMSE={lat_rmse:.6f}, "
                    f"disc_MAE={disc_mae_str}, disc_RMSE={disc_rmse_str}, "
                    f"Jaccard={jacc_str}, "
                    f"latent_count={acc['lat_count']}, disc_count={acc['disc_count']}"
                )

                out[ratio] = {
                    "latent_mae": lat_mae,
                    "latent_rmse": lat_rmse,
                    "discrete_mae": disc_mae,
                    "discrete_rmse": disc_rmse,
                    "jaccard": jacc,
                    "latent_count": acc["lat_count"],
                    "discrete_count": acc["disc_count"],
                }
            return out

        results_cond_final = _finalize(results_cond, "conditions-scan")
        results_proc_final = _finalize(results_proc, "procedures-scan")
        results_drug_final = _finalize(results_drug, "drugs-scan")

        return (
            results_cond_final,
            results_proc_final,
            results_drug_final,
            cond_latent_abs_err,
            proc_latent_abs_err,
            drug_latent_abs_err,
        )



## Membership Inference Attack
def _convert_input_mia(batch_data, tokenizers, device, config_health, mask_mode="train"):
    """
    把 dataloader 里读出来的 batch（list/str）转换成 nnet.encode_everything_test 可吃的 tensor，
    并构造必要的 *_mask / mask_hist / mask。

    mask_mode:
      - "train": 随机生成 mask（对齐训练 convert_input(..., mask_static=False)）
      - "data" : 使用数据自带 batch['mask']（对齐补全时固定缺失模式）
    """
    indices, batch = batch_data
    indices = indices.to(device, non_blocking=True)

    feature_keys = ['conditions', 'drugs', 'procedures']

    for fk in feature_keys:
        # 先保存原始 comp（list-of-codes）
        origin_comp = batch[fk + '_comp']  # 建议：如果它是 list，且后面可能被改动，用 copy()

        # 给 encode_everything_test 用的 hard_labels（必须有）
        batch[fk + '_comps'] = tokenizers[fk].batch_encode_2d(
            origin_comp, padding=False, truncation=False
        )
        batch[fk + '_comps_origin'] = origin_comp

        # tokenize -> ids
        batch[fk] = tokenizers[fk].batch_encode_2d(batch[fk])  # B, M
        batch[fk + '_hist'] = tokenizers[fk].batch_encode_3d(batch[fk + '_hist'])  # B, V, M
        batch[fk + '_comp'] = tokenizers[fk].batch_encode_2d(origin_comp)  # B, M

        # to tensor
        batch[fk] = torch.tensor(batch[fk], dtype=torch.long, device=device)
        batch[fk + '_hist'] = torch.tensor(batch[fk + '_hist'], dtype=torch.long, device=device)
        batch[fk + '_comp'] = torch.tensor(batch[fk + '_comp'], dtype=torch.long, device=device)

        # padding masks
        batch[fk + '_mask'] = batch[fk] != 0
        batch[fk + '_hist_mask'] = batch[fk + '_hist'] != 0
        batch[fk + '_comp_mask'] = batch[fk + '_comp'] != 0

        batch['has_note'] = False

    # 生成/使用 view mask（B,3）
    if mask_mode == "data":
        # 数据里自带 mask（确保 bool + device）
        batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
    elif mask_mode == "train":
        # 训练时的随机 mask
        B = batch['conditions'].shape[0]
        batch['mask'] = utils.generate_mask(B, m=3).to(device=device)
    else:
        raise ValueError(f"mask_mode must be 'train' or 'data', got: {mask_mode}")

    # mask_hist（有些模型会用到；跟你训练保持一致）
    aligned = [batch[fk + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1) for fk in feature_keys]  # B, V, 1
    batch['mask_hist'] = torch.cat(aligned, dim=-1)  # B, V, 3

    return indices, batch


@torch.no_grad()
def mia_attack_p_audit(
    nnet,
    diff_config,
    config_health,
    hp_dataset,
    tokenizers,
    max_batches=50,
    repeats=2,
    mia_alphas=(0.05, 0.01, 0.001),
    mask_mode="train",
    use_train_encoder=False,
):
    """
    Ye(CCS'22) Attack-P（loss-threshold）快速审计：用 non-member(test) 的 loss 分位数定阈值，控制 FPR=alpha，
    然后看 member(train) 的 TPR。

    分数(score) = mean over views of LSimple.loss_mse  -> [B]

    参数：
      - max_batches: 每个 split 最多取多少个 batch 做快速体检（想更稳就增大甚至跑全量）
      - repeats: 每条样本重复计算几次（LSimple 内部本身有随机 n/eps；repeats 越大分数越稳）
      - mia_alphas: 你想看的 FPR 档位
      - mask_mode: "train"（对齐训练随机mask） or "data"（对齐真实缺失mask）
      - use_train_encoder: True 则用 nnet.encode_everything（更对齐训练）；False 用 encode_everything_test（更像推理）
    """
    nnet.eval()
    device = torch.device('cuda:' + str(diff_config['device']) if torch.cuda.is_available() else 'cpu')

    data_n = config_health['DATASET']
    dataset = get_dataset(data_n, aug=True, samples=hp_dataset, config=diff_config)

    batch_size = diff_config['train']['batch_size']
    train_dataset = dataset.get_split(split='train', labeled=True)
    test_dataset  = dataset.get_split(split='test',  labeled=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=collate_fn_dict
    )
    test_loader = DataLoader(
        test_dataset, batch_size=diff_config['sample']['mini_batch_size'], shuffle=False, drop_last=False,
        num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=collate_fn_dict
    )

    # 与训练完全一致的 schedule
    _betas = utils.stable_diffusion_beta_schedule()
    schedule = utils.Schedule(_betas)

    def _encode(indices, batch):
        # 输出 z0 和 context（与你训练/推理的接口对齐）
        if use_train_encoder and hasattr(nnet, "encode_everything"):
            z0, context, *_ = nnet.encode_everything(indices, **batch)
            return z0, context
        else:
            z0, context, *_ = nnet.encode_everything_test(indices, **batch)
            return z0, context

    @torch.no_grad()
    def _to_per_sample(loss_mse: torch.Tensor) -> torch.Tensor:
        # 目标：返回 [B]
        if loss_mse.dim() == 1:  # [B]
            return loss_mse
        elif loss_mse.dim() == 2:  # [B,3] 或 [B,K]
            return loss_mse.mean(dim=1)
        else:  # [B, ...]
            return loss_mse.flatten(1).mean(dim=1)

    @torch.no_grad()
    def per_sample_score_lsimple(z0, context, batch, repeats=2):
        scores = []
        for _ in range(repeats):
            loss_mse, _, _ = utils.LSimple(z0, nnet, schedule, context=context, **batch)
            scores.append(_to_per_sample(loss_mse))
        return torch.stack(scores, dim=0).mean(dim=0)  # [B]

    @torch.no_grad()
    def collect_scores(loader, max_batches=50, repeats=2):
        all_s = []
        for bi, (indices, batch) in enumerate(loader):
            if bi >= max_batches:
                break

            indices, batch = _convert_input_mia((indices, batch), tokenizers, device, config_health, mask_mode=mask_mode)
            z0, context = _encode(indices, batch)
            s = per_sample_score_lsimple(z0, context, batch, repeats=repeats)
            all_s.append(s.detach().cpu())

        if len(all_s) == 0:
            raise RuntimeError("collect_scores got 0 batches. Check max_batches or dataloader.")

        return torch.cat(all_s, dim=0).numpy()

    # 1) 收集 member / non-member 的分数
    train_scores = collect_scores(train_loader, max_batches=max_batches, repeats=repeats)  # member
    test_scores  = collect_scores(test_loader,  max_batches=max_batches, repeats=repeats)  # non-member

    print("N_train_scores:", len(train_scores), "N_test_scores:", len(test_scores))

    # 2) Attack-P：用 non-member 的 alpha 分位数设阈值（score 越小越像 member）
    res = {}
    for a in mia_alphas:
        c = float(np.quantile(test_scores, a))
        tpr = float((train_scores <= c).mean())
        fpr = float((test_scores  <= c).mean())
        res[a] = {"threshold": c, "TPR": tpr, "FPR": fpr}

    logging.info(f"[MIA Attack-P] mask_mode={mask_mode}, repeats={repeats}, max_batches={max_batches}, res={res}")
    return res


@torch.no_grad()
def mia_to_per_sample_loss(loss_tensor: torch.Tensor) -> torch.Tensor:
    """Convert loss tensor to shape [B]."""
    if loss_tensor.dim() == 1:
        return loss_tensor
    return loss_tensor.flatten(1).mean(dim=1)


@torch.no_grad()
def mia_prepare_batch_for_unitrans(batch_data, tokenizers, device, config_health, mask_mode="train"):
    """
    Prepare batch dict for nnet.encode_everything_test / utils.LSimple.
    Adds required keys: *_comps, *_comps_origin, has_note, mask_hist, mask.
    """
    indices, batch = batch_data
    indices = indices.to(device, non_blocking=True)

    feature_keys = ['conditions', 'drugs', 'procedures']

    for fk in feature_keys:
        # --- required by encode_everything_test: fk+'_comps'
        origin_comp = batch[fk + '_comp']
        batch[fk + '_comps'] = tokenizers[fk].batch_encode_2d(origin_comp, padding=False, truncation=False)
        batch[fk + '_comps_origin'] = origin_comp

        # --- tokenize main fields
        batch[fk] = tokenizers[fk].batch_encode_2d(batch[fk])                     # B,M
        batch[fk + '_hist'] = tokenizers[fk].batch_encode_3d(batch[fk + '_hist']) # B,V,M
        batch[fk + '_comp'] = tokenizers[fk].batch_encode_2d(origin_comp)         # B,M

        # --- to tensor
        batch[fk] = torch.tensor(batch[fk], dtype=torch.long, device=device)
        batch[fk + '_hist'] = torch.tensor(batch[fk + '_hist'], dtype=torch.long, device=device)
        batch[fk + '_comp'] = torch.tensor(batch[fk + '_comp'], dtype=torch.long, device=device)

        # --- padding masks (same as training convert_input)
        batch[fk + '_mask'] = batch[fk] != 0
        batch[fk + '_hist_mask'] = batch[fk + '_hist'] != 0
        batch[fk + '_comp_mask'] = batch[fk + '_comp'] != 0

    # required by unitrans.encode_everything_test
    batch['has_note'] = False

    # mask: B,3
    if mask_mode == "data":
        batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
    elif mask_mode == "train":
        B = batch['conditions'].shape[0]
        batch['mask'] = utils.generate_mask(B, m=3).to(device=device)
    else:
        raise ValueError(f"mask_mode must be 'train' or 'data', got {mask_mode}")

    # mask_hist: B,V,3
    aligned = [batch[fk + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1) for fk in feature_keys]
    batch['mask_hist'] = torch.cat(aligned, dim=-1)

    return indices, batch


@torch.no_grad()
def mia_collect_scores_lsimple(nnet, schedule, loader, tokenizers, device, config_health,
                              mask_mode="train", max_batches=50, repeats=2, agg="mean"):
    """
    Collect per-sample scores using training-aligned utils.LSimple.
    agg: "mean" | "min" | "q10"
    Return: numpy array [N]
    """
    nnet.eval()
    scores_all = []

    for bi, (indices, batch) in enumerate(loader):
        if bi >= max_batches:
            break

        indices, batch = mia_prepare_batch_for_unitrans((indices, batch), tokenizers, device, config_health, mask_mode)

        # encode (use test encoder path; aligns with your earlier pipeline)
        z0, context, *_ = nnet.encode_everything_test(indices, **batch)

        # repeats over random (n, eps) inside schedule.sample
        vals = []
        for _ in range(repeats):
            loss_mse, _, _ = utils.LSimple(z0, nnet, schedule, context=context, **batch)
            vals.append(mia_to_per_sample_loss(loss_mse))  # [B]
        vals = torch.stack(vals, dim=0)  # [R,B]

        if agg == "mean":
            s = vals.mean(dim=0)
        elif agg == "min":
            s = vals.min(dim=0).values
        elif agg == "q10":
            s = torch.quantile(vals, 0.10, dim=0)
        else:
            raise ValueError(f"agg must be mean/min/q10, got {agg}")

        scores_all.append(s.detach().cpu())

    if len(scores_all) == 0:
        raise RuntimeError("mia_collect_scores_lsimple got 0 batches. Increase max_batches or check dataloader.")

    return torch.cat(scores_all, dim=0).numpy()


def mia_attackP_threshold_lsimple_mean(nnet, diff_config, config_health, hp_dataset, tokenizers,
                                       mia_fprs=(0.05, 0.01, 0.001),
                                       mask_mode="train", max_batches=50, repeats=2):
    """
    Attack-P with LSimple score, agg=mean over repeats.
    Return dict: {fpr: {threshold, TPR, FPR, N_train, N_test}}
    """
    device = torch.device('cuda:' + str(diff_config['device']) if torch.cuda.is_available() else 'cpu')
    dataset = get_dataset(config_health['DATASET'], aug=True, samples=hp_dataset, config=diff_config)

    train_ds = dataset.get_split(split='train', labeled=True)
    test_ds  = dataset.get_split(split='test',  labeled=True)

    train_loader = DataLoader(train_ds, batch_size=diff_config['train']['batch_size'], shuffle=False,
                              drop_last=False, num_workers=8, pin_memory=True, persistent_workers=True,
                              collate_fn=collate_fn_dict)
    test_loader  = DataLoader(test_ds, batch_size=diff_config['sample']['mini_batch_size'], shuffle=False,
                              drop_last=False, num_workers=8, pin_memory=True, persistent_workers=True,
                              collate_fn=collate_fn_dict)

    schedule = utils.Schedule(utils.stable_diffusion_beta_schedule())

    train_scores = mia_collect_scores_lsimple(nnet, schedule, train_loader, tokenizers, device, config_health,
                                              mask_mode=mask_mode, max_batches=max_batches, repeats=repeats, agg="mean")
    test_scores  = mia_collect_scores_lsimple(nnet, schedule, test_loader, tokenizers, device, config_health,
                                              mask_mode=mask_mode, max_batches=max_batches, repeats=repeats, agg="mean")

    Ntr, Nts = len(train_scores), len(test_scores)
    min_fpr = 1.0 / Nts

    res = {}
    for a in mia_fprs:
        if a < min_fpr:
            # 样本太少，无法实现该FPR粒度
            continue
        thr = float(np.quantile(test_scores, a))
        tpr = float((train_scores <= thr).mean())
        fpr = float((test_scores  <= thr).mean())
        res[a] = {"threshold": thr, "TPR": tpr, "FPR": fpr, "N_train": Ntr, "N_test": Nts}

    logging.info(f"[AttackP-mean] mask_mode={mask_mode}, repeats={repeats}, max_batches={max_batches}, res={res}")
    return res


def mia_attackP_threshold_lsimple_min(nnet, diff_config, config_health, hp_dataset, tokenizers,
                                      mia_fprs=(0.05, 0.01, 0.001),
                                      mask_mode="train", max_batches=50, repeats=16):
    """
    Attack-P++: use agg=min over repeats (stronger attack; use repeats>=16).
    """
    device = torch.device('cuda:' + str(diff_config['device']) if torch.cuda.is_available() else 'cpu')
    dataset = get_dataset(config_health['DATASET'], aug=True, samples=hp_dataset, config=diff_config)

    train_ds = dataset.get_split(split='train', labeled=True)
    test_ds  = dataset.get_split(split='test',  labeled=True)

    train_loader = DataLoader(train_ds, batch_size=diff_config['train']['batch_size'], shuffle=False,
                              drop_last=False, num_workers=8, pin_memory=True, persistent_workers=True,
                              collate_fn=collate_fn_dict)
    test_loader  = DataLoader(test_ds, batch_size=diff_config['sample']['mini_batch_size'], shuffle=False,
                              drop_last=False, num_workers=8, pin_memory=True, persistent_workers=True,
                              collate_fn=collate_fn_dict)

    schedule = utils.Schedule(utils.stable_diffusion_beta_schedule())

    train_scores = mia_collect_scores_lsimple(nnet, schedule, train_loader, tokenizers, device, config_health,
                                              mask_mode=mask_mode, max_batches=max_batches, repeats=repeats, agg="min")
    test_scores  = mia_collect_scores_lsimple(nnet, schedule, test_loader, tokenizers, device, config_health,
                                              mask_mode=mask_mode, max_batches=max_batches, repeats=repeats, agg="min")

    Ntr, Nts = len(train_scores), len(test_scores)
    min_fpr = 1.0 / Nts

    res = {}
    for a in mia_fprs:
        if a < min_fpr:
            continue
        thr = float(np.quantile(test_scores, a))
        tpr = float((train_scores <= thr).mean())
        fpr = float((test_scores  <= thr).mean())
        res[a] = {"threshold": thr, "TPR": tpr, "FPR": fpr, "N_train": Ntr, "N_test": Nts}

    logging.info(f"[AttackP-min] mask_mode={mask_mode}, repeats={repeats}, max_batches={max_batches}, res={res}")
    return res


def mia_attackR_residual_threshold_lsimple_mean(target_nnet, ref_nnet, diff_config, config_health, hp_dataset, tokenizers,
                                                mia_fprs=(0.05, 0.01),
                                                mask_mode="train", max_batches=50, repeats=2):
    """
    Attack-R (Residual): score = score_target - score_ref, then threshold like Attack-P.
    ref_nnet: reference model (different seed / earlier ckpt / disjoint data).
    """
    device = torch.device('cuda:' + str(diff_config['device']) if torch.cuda.is_available() else 'cpu')
    dataset = get_dataset(config_health['DATASET'], aug=True, samples=hp_dataset, config=diff_config)

    train_ds = dataset.get_split(split='train', labeled=True)
    test_ds  = dataset.get_split(split='test',  labeled=True)

    train_loader = DataLoader(train_ds, batch_size=diff_config['train']['batch_size'], shuffle=False,
                              drop_last=False, num_workers=8, pin_memory=True, persistent_workers=True,
                              collate_fn=collate_fn_dict)
    test_loader  = DataLoader(test_ds, batch_size=diff_config['sample']['mini_batch_size'], shuffle=False,
                              drop_last=False, num_workers=8, pin_memory=True, persistent_workers=True,
                              collate_fn=collate_fn_dict)

    schedule = utils.Schedule(utils.stable_diffusion_beta_schedule())

    # collect raw scores
    tr_t = mia_collect_scores_lsimple(target_nnet, schedule, train_loader, tokenizers, device, config_health,
                                      mask_mode=mask_mode, max_batches=max_batches, repeats=repeats, agg="mean")
    te_t = mia_collect_scores_lsimple(target_nnet, schedule, test_loader, tokenizers, device, config_health,
                                      mask_mode=mask_mode, max_batches=max_batches, repeats=repeats, agg="mean")

    tr_r = mia_collect_scores_lsimple(ref_nnet, schedule, train_loader, tokenizers, device, config_health,
                                      mask_mode=mask_mode, max_batches=max_batches, repeats=repeats, agg="mean")
    te_r = mia_collect_scores_lsimple(ref_nnet, schedule, test_loader, tokenizers, device, config_health,
                                      mask_mode=mask_mode, max_batches=max_batches, repeats=repeats, agg="mean")

    # residual scores (smaller => more like member)
    train_scores = tr_t - tr_r
    test_scores  = te_t - te_r

    Ntr, Nts = len(train_scores), len(test_scores)
    min_fpr = 1.0 / Nts

    res = {}
    for a in mia_fprs:
        if a < min_fpr:
            continue
        thr = float(np.quantile(test_scores, a))
        tpr = float((train_scores <= thr).mean())
        fpr = float((test_scores  <= thr).mean())
        res[a] = {"threshold": thr, "TPR": tpr, "FPR": fpr, "N_train": Ntr, "N_test": Nts}

    logging.info(f"[AttackR-residual] mask_mode={mask_mode}, repeats={repeats}, max_batches={max_batches}, res={res}")
    return res




#############
# ===========================
# Leak audit for released imputed dataset (MIMIC/MIMIC-III style)
# - NN Jaccard / Weighted Jaccard (rare-token aware)
# - new_* triple exact match (when each view has 1 token)
# ===========================

import numpy as np
import math
import json, ast
from collections import defaultdict
from collections.abc import Mapping


# ---------- 0) Robust helpers ----------

def leak_unwrap_dict(sample):
    """sample can be dict OR tuple/list that contains a dict; return dict."""
    if isinstance(sample, Mapping):
        return sample
    if isinstance(sample, (tuple, list)):
        for x in sample:
            if isinstance(x, Mapping):
                return x
    raise TypeError(f"Cannot unwrap to dict. type={type(sample)} head={str(sample)[:120]}")


def leak_to_list(v):
    """Normalize v to Python list. Supports list/tuple, torch/numpy (.tolist), and string-encoded list."""
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)

    # torch / numpy
    if hasattr(v, "tolist") and not isinstance(v, str):
        try:
            vv = v.tolist()
            return vv if isinstance(vv, list) else [vv]
        except Exception:
            pass

    # string list: "['51881','5070']" or '["51881","5070"]'
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return []
        try:
            vv = json.loads(s)
            return vv if isinstance(vv, list) else [vv]
        except Exception:
            try:
                vv = ast.literal_eval(s)
                return vv if isinstance(vv, list) else [vv]
            except Exception:
                return [s]

    return [v]


def leak_norm_token(tok):
    """stringify, strip, remove existing view prefix if present."""
    s = str(tok).strip()
    if s.startswith(("c:", "p:", "d:")):
        s = s[2:]
    return s


def leak_flatten_once(seq):
    """Flatten one level if elements are list/tuple/set."""
    out = []
    for t in seq:
        if isinstance(t, (list, tuple, set)):
            out.extend(list(t))
        else:
            out.append(t)
    return out


# ---------- 1) Build token sets ----------

def leak_make_prefixed_set(sample_dict,
                           drop_empty=True,
                           drop_tokens=("0", 0, None, "", "PAD", "[PAD]")):
    """
    sample_dict expects keys: conditions/procedures/drugs (list-like).
    Return a set of tokens with view prefix: c:/p:/d:
    """
    out = set()

    def _add(prefix, seq):
        seq = leak_to_list(seq)
        seq = leak_flatten_once(seq)
        for t in seq:
            if drop_empty and (t in drop_tokens):
                continue
            if t is None:
                continue
            out.add(prefix + leak_norm_token(t))

    _add("c:", sample_dict.get("conditions", []))
    _add("p:", sample_dict.get("procedures", []))
    _add("d:", sample_dict.get("drugs", []))
    return out


# ---------- 2) Packing (train real / release new / release full) ----------

def leak_pack_train_real(ds_split):
    """
    Build train real record dicts (conditions/procedures/drugs).
    IMPORTANT: prefer original code strings fields FIRST (conditions/procedures/drugs),
    then fall back to comp/comps if needed.
    """
    def _pick(d, keys):
        for k in keys:
            if k not in d:
                continue
            v = leak_to_list(d.get(k))
            v = leak_flatten_once(v)
            if len(v) > 0:
                return v
        return []

    packed = []
    for i in range(len(ds_split)):
        s = leak_unwrap_dict(ds_split[i])

        packed.append({
            "conditions": _pick(s, ["conditions", "conditions_comps_origin", "conditions_comp", "conditions_comps"]),
            "procedures": _pick(s, ["procedures", "procedures_comps_origin", "procedures_comp", "procedures_comps"]),
            "drugs":      _pick(s, ["drugs", "drugs_comps_origin", "drugs_comp", "drugs_comps"]),
        })
    return packed


def leak_pack_release_new(ds_like):
    """Use only new_* (the generated/imputed part)."""
    # dict-of-lists support
    if isinstance(ds_like, dict):
        N = len(ds_like.get("new_conditions", []))
        out = []
        for i in range(N):
            out.append({
                "conditions": ds_like.get("new_conditions", [])[i],
                "procedures": ds_like.get("new_procedures", [])[i],
                "drugs":      ds_like.get("new_drugs", [])[i],
            })
        return out

    # list[dict] or dataset-like
    out = []
    for i in range(len(ds_like)):
        s = leak_unwrap_dict(ds_like[i])
        out.append({
            "conditions": leak_to_list(s.get("new_conditions")),
            "procedures": leak_to_list(s.get("new_procedures")),
            "drugs":      leak_to_list(s.get("new_drugs")),
        })
    return out


def leak_pack_release_full(ds_like):
    """Use final released record: observed + new_ (complete imputed record)."""
    # dict-of-lists support
    if isinstance(ds_like, dict):
        N = len(ds_like.get("conditions", []))
        out = []
        for i in range(N):
            out.append({
                "conditions": leak_to_list(ds_like.get("conditions", [])[i]) + leak_to_list(ds_like.get("new_conditions", [])[i]),
                "procedures": leak_to_list(ds_like.get("procedures", [])[i]) + leak_to_list(ds_like.get("new_procedures", [])[i]),
                "drugs":      leak_to_list(ds_like.get("drugs", [])[i])      + leak_to_list(ds_like.get("new_drugs", [])[i]),
            })
        return out

    # list[dict] or dataset-like
    out = []
    for i in range(len(ds_like)):
        s = leak_unwrap_dict(ds_like[i])
        out.append({
            "conditions": leak_to_list(s.get("conditions")) + leak_to_list(s.get("new_conditions")),
            "procedures": leak_to_list(s.get("procedures")) + leak_to_list(s.get("new_procedures")),
            "drugs":      leak_to_list(s.get("drugs"))      + leak_to_list(s.get("new_drugs")),
        })
    return out


# ---------- 3) Similarities + indexing ----------

def leak_jaccard(a_set, b_set):
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0


def leak_build_inverted_index(train_sets):
    inv = defaultdict(list)
    for i, st in enumerate(train_sets):
        for tok in st:
            inv[tok].append(i)
    return inv


def leak_build_idf_weights(train_sets):
    """IDF weights on prefixed tokens."""
    N = len(train_sets)
    df = defaultdict(int)
    for st in train_sets:
        for tok in st:
            df[tok] += 1
    # smooth idf
    w = {tok: math.log((N + 1.0) / (dfv + 1.0)) for tok, dfv in df.items()}
    return w, df


def leak_weighted_jaccard(a_set, b_set, w):
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    inter = a_set & b_set
    union = a_set | b_set
    num = 0.0
    den = 0.0
    for t in union:
        wt = w.get(t, 0.0)
        den += wt
        if t in inter:
            num += wt
    return (num / den) if den > 0 else 0.0


def leak_nn_best(query_set, train_sets, inv, sim_mode="jaccard", w=None):
    """Return (best_score, best_train_index). Candidate pruning via inverted index."""
    cand = set()
    for tok in query_set:
        cand.update(inv.get(tok, []))

    if not cand:
        return 0.0, -1

    best = 0.0
    best_j = -1
    for j in cand:
        if sim_mode == "wjaccard":
            sim = leak_weighted_jaccard(query_set, train_sets[j], w)
        else:
            sim = leak_jaccard(query_set, train_sets[j])

        if sim > best:
            best = sim
            best_j = j
            if best >= 0.999999:
                break
    return best, best_j


def leak_audit_nn(train_dicts, query_dicts, sim_mode="jaccard"):
    train_sets = [leak_make_prefixed_set(x) for x in train_dicts]
    inv = leak_build_inverted_index(train_sets)
    w, df = leak_build_idf_weights(train_sets) if sim_mode == "wjaccard" else (None, None)

    scores = np.zeros(len(query_dicts), dtype=np.float32)
    nn_idx = np.full(len(query_dicts), -1, dtype=np.int32)

    for i, q in enumerate(query_dicts):
        qset = leak_make_prefixed_set(q)
        s, j = leak_nn_best(qset, train_sets, inv, sim_mode=sim_mode, w=w)
        scores[i] = s
        nn_idx[i] = j
    return scores, nn_idx, df


def leak_report(scores, title,
                quantiles=(0.50, 0.90, 0.99, 0.999),
                near_dup_thr=0.90, exact_thr=0.999999):
    scores = np.asarray(scores, dtype=np.float64)
    print(f"\n===== {title} =====")
    print("N =", scores.size)
    if scores.size == 0:
        return
    for q in quantiles:
        print(f"quantile {q:.3f} =", float(np.quantile(scores, q)))
    print("mean =", float(scores.mean()))
    print(">= near-dup", near_dup_thr, "rate =", float((scores >= near_dup_thr).mean()))
    print(">= exact", exact_thr, "rate =", float((scores >= exact_thr).mean()))


def leak_show_topk(train_dicts, query_dicts, scores, nn_idx, k=10):
    scores = np.asarray(scores)
    idx = np.argsort(-scores)[:k]
    print(f"\n--- TOP-{k} by score ---")
    for r, qi in enumerate(idx):
        j = int(nn_idx[qi])
        print(f"[{r}] query={qi} score={float(scores[qi]):.6f} nn_train={j}")
        if j >= 0:
            qset = leak_make_prefixed_set(query_dicts[qi])
            tset = leak_make_prefixed_set(train_dicts[j])
            inter = list(qset & tset)[:20]
            print("  overlap(head):", inter)


# ---------- 4) new_* triple exact match audit (when each view has exactly 1 token) ----------

def leak_new_triple_audit(train_dicts, release_new_dicts):
    """
    If release_new has exactly 1 condition + 1 procedure + 1 drug,
    check whether the triple exists in training data. Also count unique triples (df=1).
    """
    # Build train triple multiset
    triple2count = defaultdict(int)
    triple2example = {}

    for i, d in enumerate(train_dicts):
        c = leak_to_list(d.get("conditions"))
        p = leak_to_list(d.get("procedures"))
        dr = leak_to_list(d.get("drugs"))
        # only use triples where each view has exactly 1 token (avoid cartesian explosion)
        if len(c) == 1 and len(p) == 1 and len(dr) == 1:
            t = (leak_norm_token(c[0]), leak_norm_token(p[0]), leak_norm_token(dr[0]))
            triple2count[t] += 1
            if t not in triple2example:
                triple2example[t] = i

    # Query triples
    hit = 0
    hit_unique = 0
    total = 0
    for q in release_new_dicts:
        c = leak_to_list(q.get("conditions"))
        p = leak_to_list(q.get("procedures"))
        dr = leak_to_list(q.get("drugs"))
        if len(c) == 1 and len(p) == 1 and len(dr) == 1:
            total += 1
            t = (leak_norm_token(c[0]), leak_norm_token(p[0]), leak_norm_token(dr[0]))
            if t in triple2count:
                hit += 1
                if triple2count[t] == 1:
                    hit_unique += 1

    print("\n===== new_* triple exact-match audit =====")
    print("train triples (1/1/1 only) =", len(triple2count))
    print("release triples (1/1/1 only) =", total)
    if total > 0:
        print("hit rate =", hit / total)
        print("unique-hit rate (train df=1) =", hit_unique / total)
    else:
        print("No (1,1,1) triples in release_new to audit.")
    return {
        "train_triples": len(triple2count),
        "release_triples": total,
        "hit_rate": (hit / total) if total else 0.0,
        "unique_hit_rate": (hit_unique / total) if total else 0.0,
    }


# ---------- 5) One-call runner ----------

def leak_audit_run(train_ds, release_ds_like, title_prefix="Release vs Train", topk=20):
    """
    train_ds: dataset split for training real data (ds[i] can be tuple/dict).
    release_ds_like: what you will publish (Subset/Dataset OR list[dict] OR dict-of-lists).
                    It should contain keys conditions/procedures/drugs/new_conditions/new_procedures/new_drugs.
    """
    # 1) pack train real
    train_real = leak_pack_train_real(train_ds)

    # 2) pack release (new-only and full)
    release_new = leak_pack_release_new(release_ds_like)
    release_new = [x for x in release_new if (len(x["conditions"]) + len(x["procedures"]) + len(x["drugs"])) > 0]

    release_full = leak_pack_release_full(release_ds_like)
    release_full = [x for x in release_full if (len(x["conditions"]) + len(x["procedures"]) + len(x["drugs"])) > 0]

    print("Sanity lens:")
    print("  train_real[0] lens:", len(train_real[0]["conditions"]), len(train_real[0]["procedures"]), len(train_real[0]["drugs"]))
    if release_new:
        print("  release_new[0] lens:", len(release_new[0]["conditions"]), len(release_new[0]["procedures"]), len(release_new[0]["drugs"]))
    if release_full:
        print("  release_full[0] lens:", len(release_full[0]["conditions"]), len(release_full[0]["procedures"]), len(release_full[0]["drugs"]))
    print("  N_release_new_nonempty =", len(release_new))
    print("  N_release_full_nonempty =", len(release_full))

    # 3) NN Jaccard
    s_new_j, nn_new_j, _ = leak_audit_nn(train_real, release_new, sim_mode="jaccard")
    leak_report(s_new_j, f"{title_prefix} | NN-Jaccard (new_* only)")
    leak_show_topk(train_real, release_new, s_new_j, nn_new_j, k=topk)

    s_full_j, nn_full_j, _ = leak_audit_nn(train_real, release_full, sim_mode="jaccard")
    leak_report(s_full_j, f"{title_prefix} | NN-Jaccard (full record)")
    leak_show_topk(train_real, release_full, s_full_j, nn_full_j, k=topk)

    # 4) NN Weighted-Jaccard
    s_new_w, nn_new_w, _ = leak_audit_nn(train_real, release_new, sim_mode="wjaccard")
    leak_report(s_new_w, f"{title_prefix} | NN-WeightedJaccard (new_* only)")
    leak_show_topk(train_real, release_new, s_new_w, nn_new_w, k=topk)

    s_full_w, nn_full_w, _ = leak_audit_nn(train_real, release_full, sim_mode="wjaccard")
    leak_report(s_full_w, f"{title_prefix} | NN-WeightedJaccard (full record)")
    leak_show_topk(train_real, release_full, s_full_w, nn_full_w, k=topk)

    # 5) triple audit on new_ (when 1/1/1)
    triple_stats = leak_new_triple_audit(train_real, release_new)

    return {
        "new_jaccard_scores": s_new_j,
        "full_jaccard_scores": s_full_j,
        "new_wjaccard_scores": s_new_w,
        "full_wjaccard_scores": s_full_w,
        "triple_stats": triple_stats,
    }




###### 多步采样
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
            # note 逻辑保持注释

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


# ------------------------------------------------------------
# (1) Select only samples with ALL 3 views having GT tokens
#     (NEW name)
# ------------------------------------------------------------
def seed528_complete3v_subset_v1(ds_split, min_len=1, keys_map=None):
    """
    ds_split[i] might be dict OR tuple(..., dict).
    Keep i if ALL 3 views have length>=min_len for chosen GT keys.
    """
    if keys_map is None:
        # 默认用 *_comp 作为 GT（你随机mask前的真值）
        keys_map = {
            "conditions": ["conditions_comp", "conditions_comps", "conditions"],
            "procedures": ["procedures_comp", "procedures_comps", "procedures"],
            "drugs":      ["drugs_comp", "drugs_comps", "drugs"],
        }

    def _unwrap(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, tuple):
            # find last dict in tuple
            for t in reversed(x):
                if isinstance(t, dict):
                    return t
        return x  # unknown type

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


# ------------------------------------------------------------
# (2) Seeded random view-mask (mask exactly 1 view) (NEW name)
# ------------------------------------------------------------
def seed528_viewmask_one_v1(B, device, seed):
    """
    Return mask [B,3] bool: True means "mask this view".
    Deterministic given seed.
    """
    cuda_devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        ridx = torch.randint(0, 3, (B,), device=device)
        m = torch.zeros(B, 3, dtype=torch.bool, device=device)
        m[torch.arange(B, device=device), ridx] = True
        return m


# ------------------------------------------------------------
# (3) DPM impute then compute masked-only latent MSE (NEW name)
# ------------------------------------------------------------
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
    """
    z0_true: [B,3,D]
    view_mask_b3: [B,3] bool
    score: masked-only MSE in latent space (per-sample) [B]
    """
    device = z0_true.device
    B, M, D = z0_true.shape

    mask_b3 = view_mask_b3.to(device)
    mask_b3d = mask_b3.unsqueeze(-1).expand(B, M, D)              # [B,3,D]
    denom = mask_b3d.float().sum(dim=(1, 2)).clamp_min(1.0)       # [B]

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
        # deterministic noise per repeat
        cuda_devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(seed + 10007 * r)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + 10007 * r)
            noise = torch.randn_like(z0_true)

        z_init = torch.where(mask_b3d, noise, z0_true)
        z_hat = dpm.sample(z_init, steps=sample_steps, eps=1.0 / N, T=1.0)

        diff2 = (z_hat - z0_true) ** 2
        mse_masked = (diff2 * mask_b3d.float()).sum(dim=(1, 2)) / denom  # [B]
        scores.append(mse_masked)

    scores = torch.stack(scores, dim=0)  # [R,B]
    return scores.min(dim=0).values if agg == "min" else scores.mean(dim=0)


# ------------------------------------------------------------
# (4) Collect losses for a split (NEW name)
#     strict_context=True avoids leaking the masked view into context
# ------------------------------------------------------------
@torch.no_grad()
def seed528_collect_losses_complete3v_dpm_v1(
    loader,
    nnet,
    convert_input_fn,
    sample_steps=20,
    repeats=4,
    agg="mean",
    strict_context=True,
    max_batches=None,
    seed_base=528,
    verbose_every=0
):
    feature_keys = ["conditions", "drugs", "procedures"]
    all_scores = []

    for bi, (indices, batch) in enumerate(loader):
        if (max_batches is not None) and (bi >= max_batches):
            break

        indices, batch = convert_input_fn((indices, batch), mask_static=True)

        # 1) GT latent + full context (GT latent needs full input)
        z0_true, ctx_full, *_ = nnet.encode_everything_test(indices, **batch)
        B = z0_true.size(0)

        # 2) deterministic view-mask (exactly 1 view masked)
        view_mask = seed528_viewmask_one_v1(
            B=B,
            device=z0_true.device,
            seed=seed_base + bi * 1000 + 17
        )

        # 3) recompute context with masked view hidden (recommended)
        if strict_context:
            batch2 = {}
            for k, v in batch.items():
                batch2[k] = v.clone() if torch.is_tensor(v) else v

            for j, fk in enumerate(feature_keys):
                m = view_mask[:, j]
                if m.any():
                    batch2[fk][m] = 0
                    batch2[fk + "_mask"] = (batch2[fk] != 0)

            batch2["mask"] = view_mask
            _, ctx, *_ = nnet.encode_everything_test(indices, **batch2)
        else:
            ctx = ctx_full

        # 4) DPM impute then masked-loss
        s = seed528_dpm_masked_latent_mse_v1(
            nnet=nnet,
            z0_true=z0_true,
            context=ctx,
            view_mask_b3=view_mask,
            sample_steps=sample_steps,
            repeats=repeats,
            agg=agg,
            seed=seed_base + bi * 1000
        )

        all_scores.append(s.detach().cpu())

        if verbose_every and (bi % verbose_every == 0):
            print(f"[seed528-audit] batch={bi} B={B} mean={float(s.mean()):.6f} strict_context={strict_context}")

    if len(all_scores) == 0:
        return np.array([], dtype=np.float32)
    return torch.cat(all_scores, dim=0).numpy()


# ------------------------------------------------------------
# (5) End-to-end: build complete subsets, run loss audit (NEW name)
# ------------------------------------------------------------
def seed528_run_complete3v_randommask_dpm_loss_v1(
    nnet,
    diff_config,
    config_health,
    hp_dataset,
    tokenizers,
    convert_input_fn,
    keys_map=None,              # if your GT is new_*, pass keys_map below
    min_len=1,
    sample_steps=20,
    repeats=4,
    agg="mean",
    strict_context=True,
    max_batches_train=None,
    max_batches_test=None,
    seed_base=528,
    quantiles=(0.50, 0.90, 0.99, 0.999)
):
    """
    Returns:
      {
        "train_losses": np.ndarray,
        "test_losses":  np.ndarray,
        "summary": {...}
      }
    """
    device = torch.device("cuda:" + str(diff_config["device"]) if torch.cuda.is_available() else "cpu")
    nnet.eval().to(device)

    # build dataset & splits
    dataset = get_dataset(config_health["DATASET"], aug=True, samples=hp_dataset, config=diff_config)
    train_ds = dataset.get_split(split="train", labeled=True)
    test_ds  = dataset.get_split(split="test",  labeled=True)

    # select only 3-view-complete samples
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

    # collect losses
    train_losses = seed528_collect_losses_complete3v_dpm_v1(
        train_loader, nnet, convert_input_fn,
        sample_steps=sample_steps, repeats=repeats, agg=agg,
        strict_context=strict_context,
        max_batches=max_batches_train,
        seed_base=seed_base
    )
    test_losses = seed528_collect_losses_complete3v_dpm_v1(
        test_loader, nnet, convert_input_fn,
        sample_steps=sample_steps, repeats=repeats, agg=agg,
        strict_context=strict_context,
        max_batches=max_batches_test,
        seed_base=seed_base + 100000   # use different stream for test
    )

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
        "train": _stats(train_losses),
        "test": _stats(test_losses),
    }

    print("\n===== seed528 audit: complete3v -> randommask(1 view) -> DPM -> masked-loss =====")
    print("N_train_complete =", summary["N_train"], " N_test_complete =", summary["N_test"])
    print("seed =", summary["seed"], "steps =", summary["steps"], "repeats =", summary["repeats"],
          "agg =", summary["agg"], "strict_context =", summary["strict_context"])
    print("train mean =", summary["train"]["mean"], "test mean =", summary["test"]["mean"])
    for q in quantiles:
        print(f"train q{q} =", summary["train"][f"q{q}"], " | test q{q} =", summary["test"][f"q{q}"])

    return {"train_losses": train_losses, "test_losses": test_losses, "summary": summary}


# ------------------------------------------------------------
# (6) Attack-P from losses (NEW name)
# ------------------------------------------------------------
def seed528_attackP_from_maskedloss_v1(train_losses, test_losses, alphas=(0.2, 0.1, 0.05, 0.01, 0.001)):
    """
    member if loss <= threshold (threshold is alpha-quantile of test_losses).
    """
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

# ------------------------------------------------------------
# (7) Run complete3v->randommask->DPM-loss then Attack-P (NEW name)
# ------------------------------------------------------------
def seed528_run_complete3v_dpm_then_attackP_v1(
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
    # Attack-P params
    alphas=(0.2, 0.1, 0.05, 0.01, 0.001),
    return_losses=False,
):
    """
    Returns:
      - mia_res: dict(alpha -> {...})
      - optionally also returns train_losses/test_losses/summary if return_losses=True
    """
    out = seed528_run_complete3v_randommask_dpm_loss_v1(
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


# ------------------------------------------------------------
# (8) Pretty print helper (NEW name)
# ------------------------------------------------------------
def seed528_print_attackP_table_v1(mia_res):
    """
    Print in a consistent, readable format.
    """
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


############# 罕见疾病隐私攻击(缺失用new_*、不缺失用真实值) ################
def rare_privacy_audit(
    train_samples,
    test_samples_aug,
    K=10,
    sim_th=0.9,
    rare_from_feat="conditions",
    audit_feats=("conditions",),
    use_imp_for_test=True,
    use_imp_for_train=False,
    return_cases=False,          # 新增：是否返回高风险case列表
    max_cases=None,              # 新增：最多返回多少个case（按相似度降序）
    use_candidate_index=True,    # 新增：用“罕见code倒排索引”加速找最近邻
):
    """
    罕见code隐私风险审计（Jaccard-Nearest-Neighbor 风格）：
    1) 在训练集里定义罕见 code：频次 <= K（通常用真实训练集）
    2) 在测试/增强样本中挑出包含罕见 code 的样本（按 rare_from_feat）
    3) 对这些样本，计算其与训练集中样本的最大 Jaccard 相似度（按 audit_feats）
    4) 汇总统计；可选返回 ≥sim_th 的样本ID与最相似训练样本

    train_samples: 通常是 raw_train_samples（真实训练样本，没被new替换）
    test_samples_aug: augment_dataset 输出的 test_samples_aug（含 new_*，且原字段可能被你挖空）
    """

    from collections import Counter, defaultdict
    import numpy as np

    VIEW2IDX = {"conditions": 0, "procedures": 1, "drugs": 2}

    def _is_list_of_list(x):
        return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)

    def flatten_codes(view_val):
        if view_val is None:
            return []
        if _is_list_of_list(view_val):
            out = []
            for vv in view_val:
                out.extend(vv)
            return out
        return list(view_val)

    def get_incomplete_raw(sample):
        inc = sample.get("incomplete_raw", None)
        if isinstance(inc, list) and len(inc) == 3 and not isinstance(inc[0], list):
            return [1 if int(x) != 0 else 0 for x in inc]

        inc2 = sample.get("incomplete", None)
        if isinstance(inc2, list) and len(inc2) > 0 and isinstance(inc2[0], list):
            out = [0, 0, 0]
            for row in inc2:
                if isinstance(row, list) and len(row) == 3:
                    out = [max(out[j], 1 if int(row[j]) != 0 else 0) for j in range(3)]
            return out

        return [0, 0, 0]

    def build_view(sample, feat, use_imp):
        """use_imp=True：缺失才用 new_*（与 loader 的 merge_new 一致）"""
        if not use_imp:
            return sample.get(feat, None)

        inc_raw = get_incomplete_raw(sample)
        j = VIEW2IDX[feat]
        orig = sample.get(feat, None)
        newv = sample.get("new_" + feat, None)

        if inc_raw[j] == 1 and newv is not None:
            return newv
        return orig

    def jaccard(a_set, b_set):
        if not a_set and not b_set:
            return 1.0
        inter = len(a_set & b_set)
        union = len(a_set | b_set)
        return inter / union if union > 0 else 0.0

    def _sid(s):
        return {
            "patient_id": s.get("patient_id", None),
            "visit_id": s.get("visit_id", None),
            "visit_id_hist": s.get("visit_id_hist", None),
        }

    # ---------- 1) 统计训练集里的罕见 code（按 rare_from_feat） ----------
    cnt = Counter()
    for s in train_samples:
        cnt.update(flatten_codes(build_view(s, rare_from_feat, use_imp_for_train)))
    rare = {c for c, f in cnt.items() if f <= K}

    # ---------- 2) 构建 train_sets（按 audit_feats）+ 倒排索引（可选） ----------
    train_sets = []
    rare_to_train_idx = defaultdict(set)  # rare code -> set(train_idx)

    for i, s in enumerate(train_samples):
        codes_all = []
        for feat in audit_feats:
            codes_all.extend(flatten_codes(build_view(s, feat, use_imp_for_train)))
        S_train = set(codes_all)
        train_sets.append(S_train)

        if use_candidate_index:
            # 候选用 rare_from_feat 的罕见 code 建索引（更小更快）
            codes_rare_src = set(flatten_codes(build_view(s, rare_from_feat, use_imp_for_train)))
            for c in (codes_rare_src & rare):
                rare_to_train_idx[c].add(i)

    # ---------- 3) 遍历 test：对“含罕见code”的样本算 max_sim ----------
    max_sims = []
    triggered = 0
    high_risk_cases = []

    for ti, s in enumerate(test_samples_aug):
        # 触发判断用 rare_from_feat（test 用 imp 更贴近“下游实际输入”）
        rare_view = build_view(s, rare_from_feat, use_imp_for_test)
        rare_codes_in_sample = set(flatten_codes(rare_view)) & rare
        if not rare_codes_in_sample:
            continue

        triggered += 1

        # 构建 test 的集合（按 audit_feats）
        codes_all = []
        for feat in audit_feats:
            codes_all.extend(flatten_codes(build_view(s, feat, use_imp_for_test)))
        S_test = set(codes_all)

        # 候选训练样本集合（用倒排索引加速）
        if use_candidate_index:
            cand = set()
            for c in rare_codes_in_sample:
                cand |= rare_to_train_idx.get(c, set())
            cand_list = list(cand) if cand else list(range(len(train_sets)))
        else:
            cand_list = list(range(len(train_sets)))

        # 找最相似训练样本
        best_sim = -1.0
        best_j = None
        for j in cand_list:
            sim = jaccard(S_test, train_sets[j])
            if sim > best_sim:
                best_sim = sim
                best_j = j

        max_sims.append(best_sim)

        if return_cases and best_sim >= sim_th and best_j is not None:
            high_risk_cases.append({
                "test_index": ti,
                "train_index": best_j,
                "max_sim": float(best_sim),
                "hit_rare_codes": list(sorted(rare_codes_in_sample)),
                "test_id": _sid(s),
                "train_id": _sid(train_samples[best_j]),
            })

    # ---------- 4) 汇总 ----------
    if triggered == 0:
        summary = {
            "triggered": 0,
            "note": "no rare samples under this definition",
            "rare_K": int(K),
            "sim_threshold": float(sim_th),
            "rare_from_feat": rare_from_feat,
            "audit_feats": tuple(audit_feats),
            "use_imp_for_test": bool(use_imp_for_test),
            "use_imp_for_train": bool(use_imp_for_train),
            "rare_vocab_size": int(len(rare)),
        }
        return (summary, []) if return_cases else summary

    max_sims = np.array(max_sims, dtype=float)
    summary = {
        "triggered": int(triggered),
        "mean_max_sim": float(max_sims.mean()),
        "p95_max_sim": float(np.quantile(max_sims, 0.95)),
        "risk_rate_ge_th": float((max_sims >= sim_th).mean()),
        "rare_K": int(K),
        "sim_threshold": float(sim_th),
        "rare_from_feat": rare_from_feat,
        "audit_feats": tuple(audit_feats),
        "use_imp_for_test": bool(use_imp_for_test),
        "use_imp_for_train": bool(use_imp_for_train),
        "rare_vocab_size": int(len(rare)),
    }

    if return_cases:
        # 按相似度降序
        high_risk_cases.sort(key=lambda x: -x["max_sim"])
        if max_cases is not None:
            high_risk_cases = high_risk_cases[:max_cases]
        return summary, high_risk_cases

    return summary


def print_cases_with_translation(
    train_samples,
    test_samples_aug,
    cases,
    audit_feats=("conditions","procedures","drugs"),
    use_imp_for_test=True,
    max_print=50,
    max_codes_each=30,
):
    """
    把 ≥阈值 的case打印出来，并尝试翻译 code->名称。
    翻译依赖 PyHealth：pyhealth.medcode.InnerMap（没装就只打印code）
    """
    # --------- 翻译器（可选）---------
    translators = {}
    try:
        from pyhealth.medcode import InnerMap
        translators["conditions"]  = InnerMap.load("ICD9CM")
        translators["procedures"]  = InnerMap.load("ICD9PROC")
        translators["drugs"]       = InnerMap.load("ATC")
    except Exception as e:
        print("[WARN] PyHealth InnerMap not available, will print raw codes only.", e)

    VIEW2IDX = {"conditions": 0, "procedures": 1, "drugs": 2}

    def _is_list_of_list(x):
        return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)

    def flatten_codes(view_val):
        if view_val is None:
            return []
        if _is_list_of_list(view_val):
            out = []
            for vv in view_val:
                out.extend(vv)
            return out
        return list(view_val)

    def get_incomplete_raw(sample):
        inc = sample.get("incomplete_raw", None)
        if isinstance(inc, list) and len(inc) == 3 and not isinstance(inc[0], list):
            return [1 if int(x) != 0 else 0 for x in inc]
        inc2 = sample.get("incomplete", None)
        if isinstance(inc2, list) and len(inc2) > 0 and isinstance(inc2[0], list):
            out = [0, 0, 0]
            for row in inc2:
                if isinstance(row, list) and len(row) == 3:
                    out = [max(out[j], 1 if int(row[j]) != 0 else 0) for j in range(3)]
            return out
        return [0, 0, 0]

    def build_view(sample, feat, use_imp):
        if not use_imp:
            return sample.get(feat, None)
        inc_raw = get_incomplete_raw(sample)
        j = VIEW2IDX[feat]
        orig = sample.get(feat, None)
        newv = sample.get("new_" + feat, None)
        if inc_raw[j] == 1 and newv is not None:
            return newv
        return orig

    def jaccard(a_set, b_set):
        if not a_set and not b_set:
            return 1.0
        inter = len(a_set & b_set)
        union = len(a_set | b_set)
        return inter / union if union > 0 else 0.0

    def translate_list(codes, feat):
        mapper = translators.get(feat, None)
        out = []
        for c in list(codes)[:max_codes_each]:
            name = None
            if mapper is not None:
                try:
                    name = mapper.lookup(str(c))
                except Exception:
                    name = None
            out.append(f"{c}: {name}" if name else f"{c}")
        return out

    for rank, c in enumerate(cases[:max_print], start=1):
        ti = c["test_index"]
        tr_i = c["train_index"]
        sim = c["max_sim"]

        ts = test_samples_aug[ti]
        tr = train_samples[tr_i]

        print("\n" + "="*90)
        print(f"[{rank}] max_sim={sim:.4f}")
        print("TEST :", c["test_id"])
        print("TRAIN:", c["train_id"])
        print("hit_rare_codes:", c.get("hit_rare_codes", [])[:30])

        # 分视图检查“几乎完全一致”
        for feat in audit_feats:
            S_test = set(flatten_codes(build_view(ts, feat, use_imp_for_test)))
            S_tr   = set(flatten_codes(tr.get(feat, None)))  # train通常用真实即可
            inter  = S_test & S_tr
            sim_f  = jaccard(S_test, S_tr)
            print(f"\n  [{feat}] jaccard={sim_f:.4f}  overlap={len(inter)}  test_size={len(S_test)}  train_size={len(S_tr)}")

            # 只打印 overlap 的翻译（最能看出是不是“复刻”）
            for line in translate_list(sorted(inter), feat):
                print("   -", line)


############# 罕见疾病隐私攻击(测试样本全部用new_*) ################
def max_jaccard_all_new_vs_train(
    train_samples_real,
    test_samples_aug,
    feats=("conditions", "procedures", "drugs"),
    sim_th=0.7,
    top_print=20,
    return_pairs=True,
    use_inverted_index=True,     # 是否用倒排索引加速候选集
    print_details=False,         # <<< 新增：是否打印每个高风险case的分视图细节
    code_name_maps=None,         # <<< 新增：可选 {"conditions":{code:name}, "procedures":{...}, "drugs":{...}}
    max_overlap_print=60,        # <<< 新增：每个视图最多打印多少个 overlap code（None=全打印）
):
    """
    目标：把 test 的输入强制设为 new_*（全生成），然后对每个 test 样本
    找到与所有 train 样本的最大 Jaccard（最近邻）。

    train_samples_real: list[dict] 真实训练样本（用原字段 conditions/procedures/drugs）
    test_samples_aug:   list[dict] augment 后测试样本（含 new_conditions/new_procedures/new_drugs）
    """

    from collections import defaultdict
    import numpy as np

    def _is_list_of_list(x):
        return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)

    def flatten_codes(view_val):
        """把 list-of-list(按visit) 或 list(样本级) 展平成一维 code list"""
        if view_val is None:
            return []
        if _is_list_of_list(view_val):
            out = []
            for vv in view_val:
                out.extend(vv)
            return out
        return list(view_val)

    def build_code_set(sample, feats=("conditions", "procedures", "drugs"), prefix=None):
        """
        prefix=None  -> 用原字段：conditions/procedures/drugs
        prefix="new_"-> 用 new_conditions/new_procedures/new_drugs
        """
        all_codes = []
        for feat in feats:
            key = (prefix + feat) if prefix is not None else feat
            all_codes.extend(flatten_codes(sample.get(key, None)))
        return set(all_codes)

    def build_code_set_one_view(sample, feat, prefix=None):
        key = (prefix + feat) if prefix is not None else feat
        return set(flatten_codes(sample.get(key, None)))

    def jaccard_set(A, B):
        if not A and not B:
            return 1.0
        if not A or not B:
            return 0.0
        inter = len(A & B)
        union = len(A | B)
        return inter / union if union > 0 else 0.0

    def _sid(s):
        return {
            "patient_id": s.get("patient_id", None),
            "visit_id": s.get("visit_id", None),
            "visit_id_hist": s.get("visit_id_hist", None),
        }

    def _name(feat, code):
        if not code_name_maps:
            return None
        m = code_name_maps.get(feat, None)
        if not m:
            return None
        return m.get(code, None)

    # 1) 训练集真实 set
    train_sets = [build_code_set(s, feats=feats, prefix=None) for s in train_samples_real]

    # 2) 倒排索引：code -> train_idx（加速候选集）
    inv = defaultdict(list)
    if use_inverted_index:
        for i, S in enumerate(train_sets):
            for c in S:
                inv[c].append(i)

    # 3) 遍历 test：用 new_* 建 set，然后找最大相似度 train
    results = []
    high = []

    for ti, s in enumerate(test_samples_aug):
        Snew = build_code_set(s, feats=feats, prefix="new_")  # <<< 全用 new_*

        # 候选集：共享至少一个 code 的 train
        if use_inverted_index:
            cand = set()
            for c in Snew:
                cand.update(inv.get(c, []))
            cand_list = list(cand)
        else:
            cand_list = list(range(len(train_sets)))

        best_sim = 0.0
        best_j = None
        if cand_list:
            for j in cand_list:
                sim = jaccard_set(Snew, train_sets[j])
                if sim > best_sim:
                    best_sim = sim
                    best_j = j

        rec = {
            "test_index": ti,
            "max_sim": float(best_sim),
            "best_train_index": int(best_j) if best_j is not None else None,
            "test_id": _sid(s),
        }
        results.append(rec)
        if best_sim >= sim_th:
            high.append(rec)

    # 4) 汇总统计
    sims = np.array([r["max_sim"] for r in results], dtype=float)
    summary = {
        "N_test": int(len(results)),
        "feats": tuple(feats),
        "test_use": "new_* (ALL generated)",
        "train_use": "real/original",
        "mean_max_sim": float(sims.mean()) if len(sims) else 0.0,
        "p95_max_sim": float(np.quantile(sims, 0.95)) if len(sims) else 0.0,
        "p99_max_sim": float(np.quantile(sims, 0.99)) if len(sims) else 0.0,
        "risk_rate_ge_th": float((sims >= sim_th).mean()) if len(sims) else 0.0,
        "sim_threshold": float(sim_th),
        "high_risk_cases": int(len(high)),
    }

    # 5) 打印 top 高风险 +（可选）分视图细节
    if top_print and len(high) > 0:
        high_sorted = sorted(high, key=lambda x: x["max_sim"], reverse=True)[:top_print]
        print("\n==== [ALL-new vs TRAIN] high-risk top cases ====")

        for k, h in enumerate(high_sorted, 1):
            j = h["best_train_index"]
            print(f"[{k}] max_sim={h['max_sim']:.4f}")
            print(" TEST :", h["test_id"])
            if j is not None:
                t = train_samples_real[j]
                print(" TRAIN:", _sid(t))
            else:
                print(" TRAIN: None")
            print("-" * 88)

            # >>> 分视图细节
            if print_details and j is not None:
                test_s = test_samples_aug[h["test_index"]]
                train_s = train_samples_real[j]

                for feat in feats:
                    A = build_code_set_one_view(test_s, feat, prefix="new_")   # test: new_*
                    B = build_code_set_one_view(train_s, feat, prefix=None)    # train: real
                    inter = A & B
                    jac = jaccard_set(A, B)

                    print(f"  [{feat}] jaccard={jac:.4f}  overlap={len(inter)}  test_size={len(A)}  train_size={len(B)}")

                    if len(inter) > 0:
                        inter_list = sorted(list(inter))
                        if max_overlap_print is not None:
                            inter_list = inter_list[:max_overlap_print]

                        for code in inter_list:
                            nm = _name(feat, code)
                            if nm:
                                print(f"   - {code}: {nm}")
                            else:
                                print(f"   - {code}")

                        if max_overlap_print is not None and len(inter) > max_overlap_print:
                            print(f"   ... ({len(inter) - max_overlap_print} more overlap codes)")

                print("-" * 88)

    return (summary, results, high) if return_pairs else summary


############################## 生成数据中训练集覆盖程度(不看Jaccard) ##############################
import numpy as np
from collections import defaultdict

def _is_list_of_list(x):
    return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)

def _flatten_codes(v):
    if v is None:
        return []
    if _is_list_of_list(v):
        out = []
        for vv in v:
            out.extend(vv)
        return out
    return list(v)

def _get_feat_value(sample, feat):
    """
    兼容你的数据：
      - conditions/procedures: sample['conditions'] / ['procedures'] (list[list[str]])
      - drugs: 优先用 drugs_hist（更像输入视图），没有再用 drugs
    """
    if feat == "drugs":
        if "drugs_hist" in sample and sample["drugs_hist"] is not None:
            return sample["drugs_hist"]
        return sample.get("drugs", None)
    return sample.get(feat, None)

def _get_new_feat_value(sample, feat):
    return sample.get("new_" + feat, None)

def _build_set(sample, feats=("conditions","procedures","drugs"), use_new=False):
    codes = []
    for feat in feats:
        v = _get_new_feat_value(sample, feat) if use_new else _get_feat_value(sample, feat)
        codes.extend(_flatten_codes(v))
    return set(codes)

def _make_idf_weights(train_sets, smooth=1.0):
    """
    IDF based on TRAIN(real) only.
    w(c) = log((N+smooth)/(df+smooth)) + 1  (always > 0)
    """
    N = len(train_sets)
    df = defaultdict(int)
    for S in train_sets:
        for c in S:
            df[c] += 1
    w = {}
    for c, f in df.items():
        w[c] = float(np.log((N + smooth) / (f + smooth)) + 1.0)
    return w

def _sum_w(S, w):
    return float(sum(w.get(c, 0.0) for c in S))

def _inter_w(A, B, w):
    I = A & B
    return float(sum(w.get(c, 0.0) for c in I)), I

def _union_w(A, B, w):
    U = A | B
    return float(sum(w.get(c, 0.0) for c in U))

def audit_gen_leakage_against_train(
    train_samples_real,
    query_samples,               # 你可以传 test_samples 或 train_samples（它们里都有 new_*）
    feats=("conditions","procedures","drugs"),
    query_use_new=True,          # True: 用 new_* 当作生成；False: 用原始字段
    train_use_new=False,         # 通常 False：训练库用真实
    score_key="wrec",            # "wrec" 更适合找“复刻训练”的泄露；也可 "wj"/"wprec"
    topk=30,
    sim_th_wrec=0.9,
    sim_th_wprec=0.9,
    sim_th_wj=0.9,
    print_details=True,
):
    """
    返回：
      summary: dict
      rows: list[dict] (每个 query 的 best_train 及指标)
    """
    # --- build train(real) sets ---
    train_sets = [_build_set(s, feats=feats, use_new=train_use_new) for s in train_samples_real]
    Ntr = len(train_sets)

    # --- weights & inverted index (TRAIN) ---
    w = _make_idf_weights(train_sets)
    inv = defaultdict(list)
    for i, S in enumerate(train_sets):
        for c in S:
            inv[c].append(i)

    def _metrics(Sg, Str):
        # weighted
        iw, I = _inter_w(Sg, Str, w)
        uw = _union_w(Sg, Str, w)
        wj = iw / uw if uw > 0 else 0.0
        tr_w = _sum_w(Str, w) + 1e-12
        ge_w = _sum_w(Sg, w) + 1e-12
        wrec  = iw / tr_w      # train 被覆盖多少（复刻程度）
        wprec = iw / ge_w      # gen 有多少来自 train（抄袭比例）

        # unweighted（可选）
        inter = len(Sg & Str)
        uni = len(Sg | Str)
        j = inter / uni if uni > 0 else 0.0
        rec  = inter / (len(Str) + 1e-12)
        prec = inter / (len(Sg) + 1e-12)

        return {
            "wj": float(wj), "wrec": float(wrec), "wprec": float(wprec),
            "j": float(j), "rec": float(rec), "prec": float(prec),
            "overlap": sorted(list(I)),
            "overlap_n": int(len(I)),
            "gen_n": int(len(Sg)),
            "train_n": int(len(Str)),
        }

    # --- loop queries ---
    rows = []
    for qi, s in enumerate(query_samples):
        Sg = _build_set(s, feats=feats, use_new=query_use_new)

        # candidates: share at least 1 code
        cand = set()
        for c in Sg:
            cand.update(inv.get(c, []))

        if not cand:
            rows.append({
                "query_idx": qi, "best_train_idx": None,
                "wj": 0.0, "wrec": 0.0, "wprec": 0.0, "j": 0.0, "rec": 0.0, "prec": 0.0,
                "gen_n": int(len(Sg)), "train_n": 0, "overlap_n": 0, "overlap": [],
            })
            continue

        best = None
        best_j = None
        best_val = -1.0
        for j in cand:
            m = _metrics(Sg, train_sets[j])
            val = m[score_key]
            if val > best_val:
                best_val = val
                best = m
                best_j = j

        best.update({
            "query_idx": qi,
            "best_train_idx": int(best_j),
        })
        rows.append(best)

    # --- summary ---
    def arr(k):
        return np.asarray([r[k] for r in rows], dtype=float)

    summary = {
        "N_query": int(len(rows)),
        "N_train": int(Ntr),
        "feats": tuple(feats),
        "query_use": "new_*" if query_use_new else "raw",
        "train_use": "new_*" if train_use_new else "raw",
        "score_key": score_key,

        "wj_mean": float(arr("wj").mean()),
        "wj_q50": float(np.quantile(arr("wj"), 0.50)),
        "wj_q90": float(np.quantile(arr("wj"), 0.90)),
        "wj_q99": float(np.quantile(arr("wj"), 0.99)),
        "wrec_mean": float(arr("wrec").mean()),
        "wrec_q99": float(np.quantile(arr("wrec"), 0.99)),
        "wprec_mean": float(arr("wprec").mean()),
        "wprec_q99": float(np.quantile(arr("wprec"), 0.99)),

        # “泄露告警率”：你可以按更严格/更宽松调整
        "rate_wj_ge_th": float((arr("wj") >= sim_th_wj).mean()),
        "rate_wrec_ge_th": float((arr("wrec") >= sim_th_wrec).mean()),
        "rate_wprec_ge_th": float((arr("wprec") >= sim_th_wprec).mean()),
        "th_wj": float(sim_th_wj),
        "th_wrec": float(sim_th_wrec),
        "th_wprec": float(sim_th_wprec),
    }

    if print_details:
        print("\n===== Gen(new_*) vs Train(real) leakage audit =====")
        print("summary =", summary)

        # 你最关心“覆盖训练样本多少” -> 按 wrec 排
        top_wrec = sorted(rows, key=lambda r: r["wrec"], reverse=True)[:topk]
        print(f"\n--- TOP-{topk} by wrec (train coverage /复刻程度) ---")
        for k, r in enumerate(top_wrec):
            print(f"[{k}] q={r['query_idx']} wrec={r['wrec']:.6f} wprec={r['wprec']:.6f} wj={r['wj']:.6f} nn_train={r['best_train_idx']}")
            print("  overlap(head):", r["overlap"][:20])

        # “生成大多来自某个训练样本” -> 按 wprec 排
        top_wprec = sorted(rows, key=lambda r: r["wprec"], reverse=True)[:topk]
        print(f"\n--- TOP-{topk} by wprec (gen-from-train /抄袭比例) ---")
        for k, r in enumerate(top_wprec):
            print(f"[{k}] q={r['query_idx']} wprec={r['wprec']:.6f} wrec={r['wrec']:.6f} wj={r['wj']:.6f} nn_train={r['best_train_idx']}")
            print("  overlap(head):", r["overlap"][:20])

    return summary, rows

def train_covered_by_generated(
    train_samples_real,
    gen_samples_pool,            # e.g. train_samples + test_samples
    feats=("conditions","procedures","drugs"),
    gen_use_new=True,            # 用 new_* 当生成池
    topk=30,
    th_cover=0.9,
    print_details=True,
):
    """
    反向：train(real) -> gen(new_*) 最近邻
    看“训练集有多少条被生成池覆盖/复刻”
    """
    train_sets = [_build_set(s, feats=feats, use_new=False) for s in train_samples_real]
    gen_sets   = [_build_set(s, feats=feats, use_new=gen_use_new) for s in gen_samples_pool]

    # weights based on TRAIN
    w = _make_idf_weights(train_sets)

    # inverted index on GEN pool
    inv = defaultdict(list)
    for i, S in enumerate(gen_sets):
        for c in S:
            inv[c].append(i)

    rows = []
    for ti, Str in enumerate(train_sets):
        cand = set()
        for c in Str:
            cand.update(inv.get(c, []))
        if not cand:
            rows.append({"train_idx": ti, "best_gen_idx": None, "wrec": 0.0, "wj": 0.0, "wprec": 0.0, "overlap_n": 0})
            continue

        best = None
        best_g = None
        best_wrec = -1.0
        for gi in cand:
            Sg = gen_sets[gi]
            iw, I = _inter_w(Sg, Str, w)
            uw = _union_w(Sg, Str, w)
            wj = iw / uw if uw > 0 else 0.0
            tr_w = _sum_w(Str, w) + 1e-12
            ge_w = _sum_w(Sg, w) + 1e-12
            wrec  = iw / tr_w
            wprec = iw / ge_w
            if wrec > best_wrec:
                best_wrec = wrec
                best = {"wrec": float(wrec), "wj": float(wj), "wprec": float(wprec), "overlap": sorted(list(I)), "overlap_n": int(len(I))}
                best_g = gi

        best.update({"train_idx": ti, "best_gen_idx": int(best_g)})
        rows.append(best)

    wrec_arr = np.asarray([r["wrec"] for r in rows], dtype=float)
    summary = {
        "N_train": int(len(rows)),
        "N_gen_pool": int(len(gen_sets)),
        "feats": tuple(feats),
        "gen_use": "new_*" if gen_use_new else "raw",
        "wrec_mean": float(wrec_arr.mean()),
        "wrec_q99": float(np.quantile(wrec_arr, 0.99)),
        "rate_train_covered_ge_th": float((wrec_arr >= th_cover).mean()),
        "th_cover": float(th_cover),
    }

    if print_details:
        print("\n===== Train(real) covered by Gen(new_*) =====")
        print("summary =", summary)
        top = sorted(rows, key=lambda r: r["wrec"], reverse=True)[:topk]
        print(f"\n--- TOP-{topk} train by wrec (most covered) ---")
        for k, r in enumerate(top):
            print(f"[{k}] train={r['train_idx']} wrec={r['wrec']:.6f} wprec={r['wprec']:.6f} wj={r['wj']:.6f} nn_gen={r['best_gen_idx']}")
            print("  overlap(head):", r["overlap"][:20])

    return summary, rows


from collections import Counter
def leakage_audit_gen_vs_train(
    train_samples,
    query_samples,
    feats=("conditions", "procedures", "drugs"),
    query_use="new_*",      # "new_*" or "raw"
    train_use="raw",        # "raw" or "new_*" (一般用 raw)
    score_key="wrec",       # "wj" | "wprec" | "wrec"
    topk=20,
    # thresholds for rates
    th_wj=0.9,
    th_wprec=0.9,
    th_wrec=0.9,
    # NEW: rare/high-weight definition
    rare_weight_quantile=0.90,   # 取全体 code 权重的 0.9 分位作为“高权重/稀有”阈值
    # NEW: detail printing / return all
    print_details=False,
    return_all=False,
    # NEW: 如果 query_samples 可能就是 train_samples，本项避免“同一个样本自己匹配自己”
    exclude_self=False,
    id_keys=("patient_id", "visit_id"),
):
    """
    生成(new_*) vs 训练(raw) 泄露审计（带频率/IDF加权的 wj/wprec/wrec）
    并新增 debug：
      - |G|、|T|
      - overlap 里高权重(稀有)码占比（权重占比 + 数量占比）
    """

    # ------------------ helpers ------------------
    def _is_list_of_list(x):
        return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)

    def flatten_codes(view_val):
        if view_val is None:
            return []
        if _is_list_of_list(view_val):
            out = []
            for vv in view_val:
                out.extend(vv)
            return out
        return list(view_val)

    def get_view(sample, feat, use):
        # use == "new_*" -> new_conditions/new_procedures/new_drugs
        if use == "new_*":
            return sample.get("new_" + feat, None)
        else:
            return sample.get(feat, None)

    def make_code_set(sample, feats, use):
        codes = []
        for feat in feats:
            # 建议加前缀避免不同体系冲突（你之前输出里就有 d:/p: 这种）
            prefix = feat[0] + ":"  # c:/p:/d: 简化
            codes.extend([prefix + str(x) for x in flatten_codes(get_view(sample, feat, use))])
        return set(codes)

    def sid(sample):
        return tuple(sample.get(k, None) for k in id_keys)

    # ------------------ build train sets ------------------
    train_sets = []
    train_ids = []
    df = Counter()  # document frequency over train

    for s in train_samples:
        S = make_code_set(s, feats, train_use)
        train_sets.append(S)
        train_ids.append(sid(s))
        for c in S:
            df[c] += 1

    Ntr = len(train_sets)
    # ---- weight: IDF-like (rare -> large weight) ----
    # 你如果已经有自己的权重表，就把这里替换掉即可
    w = {}
    for c, f in df.items():
        # 平滑版 idf
        w[c] = np.log((Ntr + 1.0) / (f + 1.0)) + 1.0

    # NEW: 高权重阈值（代表“稀有”）
    all_w = np.array(list(w.values()), dtype=float)
    high_w_thr = float(np.quantile(all_w, rare_weight_quantile)) if len(all_w) else 1e9

    def wsum(codes_set):
        return float(sum(w.get(c, 0.0) for c in codes_set))

    def weighted_jaccard(A, B):
        I = A & B
        U = A | B
        denom = wsum(U)
        return 0.0 if denom <= 0 else wsum(I) / denom

    def weighted_precision(A, B):
        # A=G (generated), B=T (train)
        I = A & B
        denom = wsum(A)
        return 0.0 if denom <= 0 else wsum(I) / denom

    def weighted_recall(A, B):
        # A=G (generated), B=T (train)
        I = A & B
        denom = wsum(B)
        return 0.0 if denom <= 0 else wsum(I) / denom

    # ------------------ loop queries ------------------
    rows = []
    scores_for_summary = {"wj": [], "wprec": [], "wrec": []}

    for qi, qs in enumerate(query_samples):
        G = make_code_set(qs, feats, query_use)
        qid = sid(qs)

        best = {"idx": None, "wj": -1, "wprec": -1, "wrec": -1}

        # 朴素扫一遍（你原来如果有倒排索引加速，可以保留你的 cand_list）
        for j, T in enumerate(train_sets):
            if exclude_self and (qid is not None) and (qid == train_ids[j]):
                continue

            wj  = weighted_jaccard(G, T)
            wp  = weighted_precision(G, T)
            wr  = weighted_recall(G, T)

            s = {"wj": wj, "wprec": wp, "wrec": wr}[score_key]
            best_s = {"wj": best["wj"], "wprec": best["wprec"], "wrec": best["wrec"]}[score_key]
            if s > best_s:
                best = {"idx": j, "wj": wj, "wprec": wp, "wrec": wr}

        if best["idx"] is None:
            continue

        j = best["idx"]
        T = train_sets[j]
        inter = G & T

        # NEW: debug columns
        G_size = len(G)
        T_size = len(T)
        inter_size = len(inter)

        # overlap 中“高权重/稀有”占比（权重占比 + 数量占比）
        inter_w = wsum(inter)
        inter_high = [c for c in inter if w.get(c, 0.0) >= high_w_thr]
        inter_high_w = wsum(inter_high)

        rare_w_ratio = 0.0 if inter_w <= 0 else float(inter_high_w / (inter_w + 1e-12))
        rare_n_ratio = 0.0 if inter_size <= 0 else float(len(inter_high) / inter_size)

        row = {
            "query_index": qi,
            "nn_train": j,
            "wj": float(best["wj"]),
            "wprec": float(best["wprec"]),
            "wrec": float(best["wrec"]),
            "G_size": int(G_size),
            "T_size": int(T_size),
            "overlap_n": int(inter_size),
            "overlap_w": float(inter_w),
            "rare_overlap_n": int(len(inter_high)),
            "rare_overlap_wratio": float(rare_w_ratio),
            "rare_overlap_nratio": float(rare_n_ratio),
            "overlap_head": sorted(list(inter))[:20],  # 你之前的 overlap(head)
            "query_id": qid,
            "train_id": train_ids[j],
        }
        rows.append(row)

        scores_for_summary["wj"].append(row["wj"])
        scores_for_summary["wprec"].append(row["wprec"])
        scores_for_summary["wrec"].append(row["wrec"])

    # ------------------ summary ------------------
    def q(x, p):
        return float(np.quantile(np.asarray(x, dtype=float), p)) if len(x) else 0.0

    summary = {
        "N_query": int(len(query_samples)),
        "N_train": int(len(train_samples)),
        "feats": tuple(feats),
        "query_use": query_use,
        "train_use": train_use,
        "score_key": score_key,

        "wj_mean": float(np.mean(scores_for_summary["wj"])) if scores_for_summary["wj"] else 0.0,
        "wj_q50": q(scores_for_summary["wj"], 0.50),
        "wj_q90": q(scores_for_summary["wj"], 0.90),
        "wj_q99": q(scores_for_summary["wj"], 0.99),

        "wprec_mean": float(np.mean(scores_for_summary["wprec"])) if scores_for_summary["wprec"] else 0.0,
        "wprec_q99": q(scores_for_summary["wprec"], 0.99),

        "wrec_mean": float(np.mean(scores_for_summary["wrec"])) if scores_for_summary["wrec"] else 0.0,
        "wrec_q99": q(scores_for_summary["wrec"], 0.99),

        "th_wj": float(th_wj),
        "th_wprec": float(th_wprec),
        "th_wrec": float(th_wrec),

        "rate_wj_ge_th": float(np.mean(np.asarray(scores_for_summary["wj"]) >= th_wj)) if scores_for_summary["wj"] else 0.0,
        "rate_wprec_ge_th": float(np.mean(np.asarray(scores_for_summary["wprec"]) >= th_wprec)) if scores_for_summary["wprec"] else 0.0,
        "rate_wrec_ge_th": float(np.mean(np.asarray(scores_for_summary["wrec"]) >= th_wrec)) if scores_for_summary["wrec"] else 0.0,

        # NEW: 稀有阈值信息（便于解释）
        "rare_weight_quantile": float(rare_weight_quantile),
        "high_w_threshold": float(high_w_thr),
    }

    # ------------------ print_details ------------------
    if print_details:
        print("\n===== Gen(new_*) vs Train(real) leakage audit (with debug) =====")
        print("summary =", summary)

        # 按 score_key 排序输出 TOP-K
        rows_sorted = sorted(rows, key=lambda r: -r[score_key])
        print(f"\n--- TOP-{topk} by {score_key} ---")
        for rank, r in enumerate(rows_sorted[:topk], start=1):
            print(f"[{rank}] query={r['query_index']} score({score_key})={r[score_key]:.6f} nn_train={r['nn_train']}")
            print(f"  ids: query_id={r['query_id']} train_id={r['train_id']}")
            print(f"  sizes: |G|={r['G_size']} |T|={r['T_size']} overlap_n={r['overlap_n']} overlap_w={r['overlap_w']:.4f}")
            print(f"  rare_overlap: n={r['rare_overlap_n']} "
                  f"nratio={r['rare_overlap_nratio']:.3f} wratio={r['rare_overlap_wratio']:.3f} "
                  f"(high_w>=q{int(rare_weight_quantile*100)})")
            print("  overlap(head):", r["overlap_head"])

    if return_all:
        return summary, rows
    return summary

import numpy as np
from collections import Counter

def leakage_audit_gen_vs_train_viewwise1(
    train_samples,
    query_samples,                      # 建议传 test_samples
    feats=("conditions", "procedures", "drugs"),
    query_use="new_*",                  # query 用 new_*
    train_use="raw",                    # train 用原始视图
    match_score_key="wrec",             # 用整体的哪个指标选最近邻：wrec/wj/wprec
    topk=20,
    # threshold: containment rate
    th_wrec=0.9,                        # “近似包含”阈值
    strict_contain=True,                # 是否额外统计严格集合包含 T ⊆ G
    # rare/high-weight
    rare_weight_quantile=0.90,
    print_details=False,
    return_all=False,
    exclude_self=False,
    id_keys=("patient_id", "visit_id"),
    w_default_mode="high",              # 生成中出现但训练df没见过的code权重：'high'更保守
):
    """
    1) 用三视图整体(含前缀)在 train 中找每个 query 的最近邻（按 match_score_key）
    2) 对同一个 (query, nn_train) 配对，分别计算：
       - overall 的 wj/wprec/wrec + contain
       - 每个单视图的 wj/wprec/wrec + contain
    3) 输出：三视图分别 + 整体 的 summary
    """

    # ------------------ helpers ------------------
    def _is_list_of_list(x):
        return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)

    def flatten_codes(view_val):
        if view_val is None:
            return []
        if _is_list_of_list(view_val):
            out = []
            for vv in view_val:
                out.extend(vv)
            return out
        return list(view_val)

    def get_view(sample, feat, use):
        if use == "new_*":
            return sample.get("new_" + feat, None)
        else:
            return sample.get(feat, None)

    def make_code_set(sample, feat_list, use, with_prefix=True):
        codes = []
        for feat in feat_list:
            vals = flatten_codes(get_view(sample, feat, use))
            if with_prefix:
                prefix = feat[0] + ":"   # c:/p:/d:
                codes.extend([prefix + str(x) for x in vals])
            else:
                codes.extend([str(x) for x in vals])
        return set(codes)

    def sid(sample):
        return tuple(sample.get(k, None) for k in id_keys)

    # ------------------ build train sets ------------------
    train_ids = []
    train_sets_overall = []
    train_sets_by_view = {f: [] for f in feats}

    df = Counter()   # df over train (overall, with prefix)

    for s in train_samples:
        train_ids.append(sid(s))

        # overall set (prefixed)
        S_all = make_code_set(s, feats, train_use, with_prefix=True)
        train_sets_overall.append(S_all)
        for c in S_all:
            df[c] += 1

        # per-view set (also prefixed for consistency)
        for f in feats:
            Sv = make_code_set(s, (f,), train_use, with_prefix=True)
            train_sets_by_view[f].append(Sv)

    Ntr = len(train_samples)

    # ---- IDF-like weight (rare -> large) ----
    w = {}
    for c, f in df.items():
        w[c] = np.log((Ntr + 1.0) / (f + 1.0)) + 1.0

    all_w = np.array(list(w.values()), dtype=float)
    high_w_thr = float(np.quantile(all_w, rare_weight_quantile)) if len(all_w) else 1.0
    max_w = float(all_w.max()) if len(all_w) else 1.0

    if w_default_mode == "high":
        w_default = high_w_thr
    elif w_default_mode == "max":
        w_default = max_w
    else:
        w_default = 1.0

    def wsum(codes_set):
        return float(sum(w.get(c, w_default) for c in codes_set))

    def weighted_jaccard(A, B):
        I = A & B
        U = A | B
        denom = wsum(U)
        return 0.0 if denom <= 0 else wsum(I) / denom

    def weighted_precision(A, B):
        I = A & B
        denom = wsum(A)
        return 0.0 if denom <= 0 else wsum(I) / denom

    def weighted_recall(A, B):
        I = A & B
        denom = wsum(B)
        if denom <= 0:
            return 0.0
        val = wsum(I) / denom
        # 轻微数值溢出保护
        return float(min(1.0, val + 1e-12))

    def strict_contains(A, B):
        # 是否严格集合包含：B ⊆ A（训练样本是否被生成样本完全包含）
        return (len(B - A) == 0)

    # ------------------ loop queries ------------------
    rows = []

    for qi, qs in enumerate(query_samples):
        qid = sid(qs)

        # query overall / per-view sets (prefixed)
        G_all = make_code_set(qs, feats, query_use, with_prefix=True)
        G_by_view = {f: make_code_set(qs, (f,), query_use, with_prefix=True) for f in feats}

        # 1) choose best nn_train by OVERALL score
        best_j = None
        best_scores = {"wj": -1, "wprec": -1, "wrec": -1}

        for j, T_all in enumerate(train_sets_overall):
            if exclude_self and (qid is not None) and (qid == train_ids[j]):
                continue

            wj  = weighted_jaccard(G_all, T_all)
            wp  = weighted_precision(G_all, T_all)
            wr  = weighted_recall(G_all, T_all)

            cur = {"wj": wj, "wprec": wp, "wrec": wr}[match_score_key]
            best = {"wj": best_scores["wj"], "wprec": best_scores["wprec"], "wrec": best_scores["wrec"]}[match_score_key]
            if cur > best:
                best_j = j
                best_scores = {"wj": wj, "wprec": wp, "wrec": wr}

        if best_j is None:
            continue

        # 2) compute viewwise metrics against the SAME nn_train
        j = best_j

        out = {
            "query_index": qi,
            "nn_train": j,
            "query_id": qid,
            "train_id": train_ids[j],
            # overall
            "overall_wj": float(best_scores["wj"]),
            "overall_wprec": float(best_scores["wprec"]),
            "overall_wrec": float(best_scores["wrec"]),
            "overall_G_size": int(len(G_all)),
            "overall_T_size": int(len(train_sets_overall[j])),
            "overall_overlap_n": int(len(G_all & train_sets_overall[j])),
        }
        if strict_contain:
            out["overall_contain"] = bool(strict_contains(G_all, train_sets_overall[j]))

        # per-view
        for f in feats:
            Gv = G_by_view[f]
            Tv = train_sets_by_view[f][j]
            out[f"{f}_wj"] = float(weighted_jaccard(Gv, Tv))
            out[f"{f}_wprec"] = float(weighted_precision(Gv, Tv))
            out[f"{f}_wrec"] = float(weighted_recall(Gv, Tv))
            out[f"{f}_G_size"] = int(len(Gv))
            out[f"{f}_T_size"] = int(len(Tv))
            out[f"{f}_overlap_n"] = int(len(Gv & Tv))
            if strict_contain:
                out[f"{f}_contain"] = bool(strict_contains(Gv, Tv))

        rows.append(out)

    # ------------------ summary ------------------
    def _q(arr, p):
        arr = np.asarray(arr, dtype=float)
        return float(np.quantile(arr, p)) if arr.size else 0.0

    def summarize(key_prefix):
        wrec_list = [r[f"{key_prefix}_wrec"] for r in rows]
        wj_list   = [r[f"{key_prefix}_wj"] for r in rows]
        wp_list   = [r[f"{key_prefix}_wprec"] for r in rows]

        summ = {
            "N_query": int(len(query_samples)),
            "N_train": int(len(train_samples)),
            "match_score_key": match_score_key,
            "th_wrec": float(th_wrec),
            f"{key_prefix}_wrec_mean": float(np.mean(wrec_list)) if wrec_list else 0.0,
            f"{key_prefix}_wrec_q50": _q(wrec_list, 0.50),
            f"{key_prefix}_wrec_q90": _q(wrec_list, 0.90),
            f"{key_prefix}_wrec_q99": _q(wrec_list, 0.99),
            f"{key_prefix}_wj_mean": float(np.mean(wj_list)) if wj_list else 0.0,
            f"{key_prefix}_wj_q99": _q(wj_list, 0.99),
            f"{key_prefix}_wprec_mean": float(np.mean(wp_list)) if wp_list else 0.0,
            f"{key_prefix}_wprec_q99": _q(wp_list, 0.99),
            f"{key_prefix}_rate_wrec_ge_th": float(np.mean(np.asarray(wrec_list) >= th_wrec)) if wrec_list else 0.0,
        }
        if strict_contain:
            contain_list = [r.get(f"{key_prefix}_contain", False) for r in rows]
            summ[f"{key_prefix}_rate_strict_contain"] = float(np.mean(np.asarray(contain_list, dtype=float))) if contain_list else 0.0
        return summ

    summary = {
        "overall": summarize("overall"),
        "conditions": summarize("conditions"),
        "procedures": summarize("procedures"),
        "drugs": summarize("drugs"),
        "rare_weight_quantile": float(rare_weight_quantile),
        "high_w_threshold": float(high_w_thr),
        "w_default_mode": w_default_mode,
    }

    # ------------------ print_details ------------------
    if print_details:
        print("\n===== Gen(new_*) vs Train(raw) VIEWWISE leakage audit =====")
        print("summary =", summary)

        # top-K by overall_wrec (or chosen match key)
        sort_key = f"overall_{match_score_key}"
        rows_sorted = sorted(rows, key=lambda r: -r[sort_key])
        print(f"\n--- TOP-{topk} by {sort_key} ---")
        for rank, r in enumerate(rows_sorted[:topk], start=1):
            print(f"[{rank}] query={r['query_index']} nn_train={r['nn_train']} ids: query_id={r['query_id']} train_id={r['train_id']}")
            print(f"  OVERALL: wrec={r['overall_wrec']:.4f} wprec={r['overall_wprec']:.4f} wj={r['overall_wj']:.4f} "
                  f"|G|={r['overall_G_size']} |T|={r['overall_T_size']} overlap_n={r['overall_overlap_n']}"
                  + (f" strict_contain={r.get('overall_contain', False)}" if strict_contain else ""))
            for f in feats:
                print(f"  {f}: wrec={r[f'{f}_wrec']:.4f} wprec={r[f'{f}_wprec']:.4f} wj={r[f'{f}_wj']:.4f} "
                      f"|G|={r[f'{f}_G_size']} |T|={r[f'{f}_T_size']} overlap_n={r[f'{f}_overlap_n']}"
                      + (f" strict_contain={r.get(f'{f}_contain', False)}" if strict_contain else ""))

    if return_all:
        return summary, rows
    return summary

def leakage_audit_gen_vs_train_viewwise(
    train_samples,
    query_samples,                      # 建议传 test_samples
    feats=("conditions", "procedures", "drugs"),
    query_use="new_*",                  # query 用 new_*
    train_use="raw",                    # train 用原始视图
    match_score_key="wrec",             # 用整体的哪个指标选最近邻：wrec/wj/wprec
    topk=20,
    # threshold: containment rate
    th_wrec=0.9,                        # “近似包含”阈值
    strict_contain=True,                # 是否额外统计严格集合包含 T ⊆ G
    # rare/high-weight
    rare_weight_quantile=0.90,          # 罕见码=IDF权重 top (1-quantile)
    print_details=False,
    return_all=False,
    exclude_self=False,
    id_keys=("patient_id", "visit_id"),
    w_default_mode="high",              # 生成中出现但训练df没见过的code权重：'high'更保守
    # ---------------- NEW: 正常子集过滤 ----------------
    normal_keep_frac=None,              # 例如 0.5 => 只保留 union unique 最小的50%
    normal_keep_by="union",             # "union" or 指定某视图名，如 "conditions"/"procedures"/"drugs"
    normal_max_union_unique=None,       # 硬阈值：union unique > 该值的样本剔除
    normal_max_view_unique=None,        # dict: {"conditions":200, "procedures":80, "drugs":200}
):
    """
    1) (可选) 先在 query 上做“正常子集”过滤：去掉极端大集合样本
    2) 用三视图整体(含前缀)在 train 中找每个 query 的最近邻（按 match_score_key）
    3) 对同一个 (query, nn_train) 配对，分别计算 overall + 每个视图：
       - wj/wprec/wrec
       - strict contain: T ⊆ G
       - rare stats: 是否命中罕见码、罕见码是否被完全包含、罕见占比
    """

    # ------------------ helpers ------------------
    def _is_list_of_list(x):
        return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)

    def flatten_codes(view_val):
        if view_val is None:
            return []
        if _is_list_of_list(view_val):
            out = []
            for vv in view_val:
                out.extend(vv)
            return out
        return list(view_val)

    def get_view(sample, feat, use):
        if use == "new_*":
            return sample.get("new_" + feat, None)
        else:
            return sample.get(feat, None)

    def make_code_set(sample, feat_list, use, with_prefix=True):
        codes = []
        for feat in feat_list:
            vals = flatten_codes(get_view(sample, feat, use))
            if with_prefix:
                prefix = feat[0] + ":"   # c:/p:/d:
                codes.extend([prefix + str(x) for x in vals])
            else:
                codes.extend([str(x) for x in vals])
        return set(codes)

    def sid(sample):
        return tuple(sample.get(k, None) for k in id_keys)

    # ------------------ NEW: compute query sizes & filter ------------------
    # 计算每个 query 的 per-view unique 和 union unique（不带前缀）
    q_view_unique = {f: [] for f in feats}
    q_union_unique = []

    for qs in query_samples:
        sets = []
        for f in feats:
            Sv = make_code_set(qs, (f,), query_use, with_prefix=False)
            q_view_unique[f].append(len(Sv))
            sets.append(Sv)
        q_union_unique.append(len(set().union(*sets)))

    q_union_unique = np.asarray(q_union_unique, dtype=int)
    for f in feats:
        q_view_unique[f] = np.asarray(q_view_unique[f], dtype=int)

    keep_mask = np.ones(len(query_samples), dtype=bool)

    # 硬阈值过滤（union）
    if normal_max_union_unique is not None:
        keep_mask &= (q_union_unique <= int(normal_max_union_unique))

    # 硬阈值过滤（per-view）
    if isinstance(normal_max_view_unique, dict):
        for f, thr in normal_max_view_unique.items():
            if f in q_view_unique:
                keep_mask &= (q_view_unique[f] <= int(thr))

    # 按分位/比例保留（例如保留 union unique 最小的50%）
    if normal_keep_frac is not None:
        frac = float(normal_keep_frac)
        frac = min(max(frac, 0.0), 1.0)
        # 选择按 union 或某视图大小排序
        if normal_keep_by == "union":
            score = q_union_unique.copy()
        elif normal_keep_by in feats:
            score = q_view_unique[normal_keep_by].copy()
        else:
            score = q_union_unique.copy()

        # 只在当前 keep_mask 为 True 的集合里再取最小 frac
        idx_all = np.where(keep_mask)[0]
        if len(idx_all) > 0 and frac < 1.0:
            idx_sorted = idx_all[np.argsort(score[idx_all])]
            k = int(np.floor(len(idx_sorted) * frac))
            k = max(k, 1)  # 至少保留1个
            keep2 = np.zeros(len(query_samples), dtype=bool)
            keep2[idx_sorted[:k]] = True
            keep_mask &= keep2

    # ------------------ build train sets ------------------
    train_ids = []
    train_sets_overall = []
    train_sets_by_view = {f: [] for f in feats}

    df = Counter()   # df over train (overall, with prefix)

    for s in train_samples:
        train_ids.append(sid(s))

        # overall set (prefixed)
        S_all = make_code_set(s, feats, train_use, with_prefix=True)
        train_sets_overall.append(S_all)
        for c in S_all:
            df[c] += 1

        # per-view set (also prefixed for consistency)
        for f in feats:
            Sv = make_code_set(s, (f,), train_use, with_prefix=True)
            train_sets_by_view[f].append(Sv)

    Ntr = len(train_samples)

    # ---- IDF-like weight (rare -> large) ----
    w = {}
    for c, f in df.items():
        w[c] = np.log((Ntr + 1.0) / (f + 1.0)) + 1.0

    all_w = np.array(list(w.values()), dtype=float)
    high_w_thr = float(np.quantile(all_w, rare_weight_quantile)) if len(all_w) else 1.0
    max_w = float(all_w.max()) if len(all_w) else 1.0

    if w_default_mode == "high":
        w_default = high_w_thr
    elif w_default_mode == "max":
        w_default = max_w
    else:
        w_default = 1.0

    def wsum(codes_set):
        return float(sum(w.get(c, w_default) for c in codes_set))

    def weighted_jaccard(A, B):
        I = A & B
        U = A | B
        denom = wsum(U)
        return 0.0 if denom <= 0 else wsum(I) / denom

    def weighted_precision(A, B):
        I = A & B
        denom = wsum(A)
        return 0.0 if denom <= 0 else wsum(I) / denom

    def weighted_recall(A, B):
        I = A & B
        denom = wsum(B)
        if denom <= 0:
            return 0.0
        val = wsum(I) / denom
        return float(min(1.0, val + 1e-12))

    def strict_contains(A, B):
        return (len(B - A) == 0)

    # ------------------ NEW: rare statistics ------------------
    def rare_stats(G, T):
        """
        rare codes 定义为：T 中权重 >= high_w_thr 的那些 code（只看训练集合中的 rare）
        返回：rare_any（交集是否命中rare），rare_valid（T里是否有rare），rare_contain（T_rare ⊆ G）
              rare_nratio / rare_wratio：交集里rare占比（数量/权重）
        """
        I = G & T
        if len(T) == 0:
            return False, False, False, 0.0, 0.0

        T_rare = {c for c in T if w.get(c, w_default) >= high_w_thr}
        rare_valid = (len(T_rare) > 0)
        I_rare = I & T_rare
        rare_any = (len(I_rare) > 0)
        rare_contain = bool(len(T_rare - G) == 0) if rare_valid else False

        inter_n = len(I)
        inter_w = wsum(I)
        rare_nratio = 0.0 if inter_n == 0 else float(len(I_rare) / inter_n)
        rare_wratio = 0.0 if inter_w <= 0 else float(wsum(I_rare) / (inter_w + 1e-12))
        return rare_any, rare_valid, rare_contain, rare_nratio, rare_wratio

    # ------------------ loop queries ------------------
    rows = []
    kept_query_indices = np.where(keep_mask)[0]

    for qi in kept_query_indices:
        qs = query_samples[qi]
        qid = sid(qs)

        # query overall / per-view sets (prefixed)
        G_all = make_code_set(qs, feats, query_use, with_prefix=True)
        G_by_view = {f: make_code_set(qs, (f,), query_use, with_prefix=True) for f in feats}

        # 1) choose best nn_train by OVERALL score
        best_j = None
        best_scores = {"wj": -1, "wprec": -1, "wrec": -1}

        for j, T_all in enumerate(train_sets_overall):
            if exclude_self and (qid is not None) and (qid == train_ids[j]):
                continue

            wj  = weighted_jaccard(G_all, T_all)
            wp  = weighted_precision(G_all, T_all)
            wr  = weighted_recall(G_all, T_all)

            cur = {"wj": wj, "wprec": wp, "wrec": wr}[match_score_key]
            best = {"wj": best_scores["wj"], "wprec": best_scores["wprec"], "wrec": best_scores["wrec"]}[match_score_key]
            if cur > best:
                best_j = j
                best_scores = {"wj": wj, "wprec": wp, "wrec": wr}

        if best_j is None:
            continue

        # 2) compute viewwise metrics against the SAME nn_train
        j = best_j
        T_all = train_sets_overall[j]

        out = {
            "query_index": int(qi),  # 保留原始index
            "nn_train": int(j),
            "query_id": qid,
            "train_id": train_ids[j],

            # overall
            "overall_wj": float(best_scores["wj"]),
            "overall_wprec": float(best_scores["wprec"]),
            "overall_wrec": float(best_scores["wrec"]),
            "overall_G_size": int(len(G_all)),
            "overall_T_size": int(len(T_all)),
            "overall_overlap_n": int(len(G_all & T_all)),
        }
        if strict_contain:
            out["overall_contain"] = bool(strict_contains(G_all, T_all))

        # overall rare
        ra, rv, rc, rnr, rwr = rare_stats(G_all, T_all)
        out["overall_rare_any"] = bool(ra)
        out["overall_rare_valid"] = bool(rv)
        out["overall_rare_contain"] = bool(rc)
        out["overall_rare_nratio"] = float(rnr)
        out["overall_rare_wratio"] = float(rwr)

        # per-view
        for f in feats:
            Gv = G_by_view[f]
            Tv = train_sets_by_view[f][j]
            out[f"{f}_wj"] = float(weighted_jaccard(Gv, Tv))
            out[f"{f}_wprec"] = float(weighted_precision(Gv, Tv))
            out[f"{f}_wrec"] = float(weighted_recall(Gv, Tv))
            out[f"{f}_G_size"] = int(len(Gv))
            out[f"{f}_T_size"] = int(len(Tv))
            out[f"{f}_overlap_n"] = int(len(Gv & Tv))
            if strict_contain:
                out[f"{f}_contain"] = bool(strict_contains(Gv, Tv))

            # view rare
            ra, rv, rc, rnr, rwr = rare_stats(Gv, Tv)
            out[f"{f}_rare_any"] = bool(ra)
            out[f"{f}_rare_valid"] = bool(rv)
            out[f"{f}_rare_contain"] = bool(rc)
            out[f"{f}_rare_nratio"] = float(rnr)
            out[f"{f}_rare_wratio"] = float(rwr)

        rows.append(out)

    # ------------------ summary ------------------
    def _q(arr, p):
        arr = np.asarray(arr, dtype=float)
        return float(np.quantile(arr, p)) if arr.size else 0.0

    def summarize(key_prefix):
        wrec_list = np.asarray([r[f"{key_prefix}_wrec"] for r in rows], dtype=float)
        wj_list   = np.asarray([r[f"{key_prefix}_wj"] for r in rows], dtype=float)
        wp_list   = np.asarray([r[f"{key_prefix}_wprec"] for r in rows], dtype=float)

        # rare
        rare_any   = np.asarray([r.get(f"{key_prefix}_rare_any", False) for r in rows], dtype=float)
        rare_valid = np.asarray([r.get(f"{key_prefix}_rare_valid", False) for r in rows], dtype=bool)
        rare_cont  = np.asarray([r.get(f"{key_prefix}_rare_contain", False) for r in rows], dtype=float)
        rare_wratio= np.asarray([r.get(f"{key_prefix}_rare_wratio", 0.0) for r in rows], dtype=float)

        summ = {
            "N_query_total": int(len(query_samples)),
            "N_query_kept": int(keep_mask.sum()),
            "N_train": int(len(train_samples)),
            "match_score_key": match_score_key,
            "th_wrec": float(th_wrec),

            f"{key_prefix}_wrec_mean": float(wrec_list.mean()) if len(wrec_list) else 0.0,
            f"{key_prefix}_wrec_q50": _q(wrec_list, 0.50),
            f"{key_prefix}_wrec_q90": _q(wrec_list, 0.90),
            f"{key_prefix}_wrec_q99": _q(wrec_list, 0.99),

            f"{key_prefix}_wj_mean": float(wj_list.mean()) if len(wj_list) else 0.0,
            f"{key_prefix}_wj_q99": _q(wj_list, 0.99),

            f"{key_prefix}_wprec_mean": float(wp_list.mean()) if len(wp_list) else 0.0,
            f"{key_prefix}_wprec_q99": _q(wp_list, 0.99),

            f"{key_prefix}_rate_wrec_ge_th": float((wrec_list >= th_wrec).mean()) if len(wrec_list) else 0.0,

            # rare summaries
            f"{key_prefix}_rate_rare_any": float(rare_any.mean()) if len(rare_any) else 0.0,
            f"{key_prefix}_rare_contain_rate_given_valid": float(rare_cont[rare_valid].mean()) if rare_valid.any() else 0.0,
            f"{key_prefix}_rare_wratio_mean": float(rare_wratio.mean()) if len(rare_wratio) else 0.0,
        }
        if strict_contain:
            contain_list = np.asarray([r.get(f"{key_prefix}_contain", False) for r in rows], dtype=float)
            summ[f"{key_prefix}_rate_strict_contain"] = float(contain_list.mean()) if len(contain_list) else 0.0
        return summ

    summary = {
        "overall": summarize("overall"),
        "conditions": summarize("conditions"),
        "procedures": summarize("procedures"),
        "drugs": summarize("drugs"),

        # rare config
        "rare_weight_quantile": float(rare_weight_quantile),
        "high_w_threshold": float(high_w_thr),
        "w_default_mode": w_default_mode,

        # normal filter config + stats
        "normal_filter": {
            "normal_keep_frac": normal_keep_frac,
            "normal_keep_by": normal_keep_by,
            "normal_max_union_unique": normal_max_union_unique,
            "normal_max_view_unique": normal_max_view_unique,
            "kept_union_unique_max": int(q_union_unique[keep_mask].max()) if keep_mask.any() else None,
            "kept_union_unique_p90": float(np.quantile(q_union_unique[keep_mask], 0.90)) if keep_mask.any() else None,
        }
    }

    # ------------------ print_details ------------------
    if print_details:
        print("\n===== Gen(new_*) vs Train(raw) VIEWWISE leakage audit (NORMAL subset enabled) =====")
        print("summary =", summary)

        sort_key = f"overall_{match_score_key}"
        rows_sorted = sorted(rows, key=lambda r: -r[sort_key])
        print(f"\n--- TOP-{topk} by {sort_key} ---")
        for rank, r in enumerate(rows_sorted[:topk], start=1):
            print(f"[{rank}] query={r['query_index']} nn_train={r['nn_train']} ids: query_id={r['query_id']} train_id={r['train_id']}")
            print(f"  OVERALL: wrec={r['overall_wrec']:.4f} wprec={r['overall_wprec']:.4f} wj={r['overall_wj']:.4f} "
                  f"|G|={r['overall_G_size']} |T|={r['overall_T_size']} overlap_n={r['overall_overlap_n']}"
                  + (f" strict_contain={r.get('overall_contain', False)}" if strict_contain else "")
                  + f" rare_any={r['overall_rare_any']} rare_contain={r['overall_rare_contain']} rare_wratio={r['overall_rare_wratio']:.3f}")
            for f in feats:
                print(f"  {f}: wrec={r[f'{f}_wrec']:.4f} wprec={r[f'{f}_wprec']:.4f} wj={r[f'{f}_wj']:.4f} "
                      f"|G|={r[f'{f}_G_size']} |T|={r[f'{f}_T_size']} overlap_n={r[f'{f}_overlap_n']}"
                      + (f" strict_contain={r.get(f'{f}_contain', False)}" if strict_contain else "")
                      + f" rare_any={r[f'{f}_rare_any']} rare_contain={r[f'{f}_rare_contain']} rare_wratio={r[f'{f}_rare_wratio']:.3f}")

    if return_all:
        return summary, rows
    return summary
