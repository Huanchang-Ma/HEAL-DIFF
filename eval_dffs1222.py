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
from torch.utils.data import DataLoader
from diffuser.dpm import NoiseScheduleVP, DPM_Solver
from diffuser.diff_dataset import merge_miss
from utils import pad_list
import pickle
import random
import matplotlib.pyplot as plt


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

    def decode(_batch, get_logits=None, thred=config['threshod']): # 可以进一步rounding
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
            # indices = [(np.where(row == 1)[0]+2).tolist() for row in array_prob] # 加上padding项
            tokens = tokenizers[feature_key].batch_decode_2d(indices) # indices+2
            hard_samples[feature_key] = tokens # 没有考虑缺失的比例信息。
        # print(hard_samples['conditions'])
        # print("AAAAAAA", hard_samples.keys(), len(hard_samples['conditions']))
        # print("BBBBB", hard_samples['conditions'][0])

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
        assert indices.size(0) == _n_samples
        indices, sample_input = convert_input((indices, batch), tokenizers, device, config_health)
        _z, _context_emb, _, labels, hard_labels = nnet.encode_everything_test(indices, **batch)
        return labels, hard_labels, dpm_solver_sample(_n_samples, config['sample']['sample_steps'], context=_context_emb,  z_init=_z, mask=sample_input['mask'])

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
        save_pickle(aug_test_sample, root_to+'test_samples.pkl')
        print("All aug data save to {}.".format(path))

    return aug_train_sample, aug_test_sample

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

            # note special（你现在 MIII 不会走到这里，所以不用担心 pad_list）
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
    save_path = "/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/src/MAE/"
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

    plt.figure(figsize=(5, 4))

    # 并排柱：每个bin内按ratio并排
    R = len(ratios)
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