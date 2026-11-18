# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train_dffs.py
# Time       ：11/11/2024 4:05 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：仿照CV重建
"""

import os
import torch
import tempfile
import einops
import wandb
import torch.nn.functional as F
import numpy as np
from absl import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from diffuser import diff_utils as utils
from diffuser.diff_dataset import get_dataset, collate_fn_dict
from diffuser.dpm import NoiseScheduleVP, DPM_Solver
from utils import set_random_seed
from utils import pad_list
from sklearn.metrics import roc_auc_score, roc_curve
import csv
import json


def run_diffusion_DP(config, config_health, hp_dataset, tokenizers, exp_num='0'):
    """
    :param config: diffusion config
    :param config_health: healthcare config
    :param hp_dataset: healthcare train dataset
    :param tokenizers: healthcare tokenizers
    :return:
    """
    print("run_diffusion")
    set_random_seed(config_health['SEED'])

    data_n = config_health['DATASET']
    task = config_health['TASK']

    # utils.calculate_quality(config['sample']['path'] + data_n)

    # 加速
    # accelerator = accelerate.Accelerator() # CUDA_VISIBLE_DEVICES=4  accelerate launch main.py
    # device = accelerator.device
    # accelerate.utils.set_seed(config['seed'], device_specific=True)
    device = torch.device('cuda:' + str(config['device']) if torch.cuda.is_available() else 'cpu')

    # logging.info(f'Process {accelerator.process_index} using device: {device}')
    logging.info(f'using device: {device}')

    # config['mixed_precision'] = accelerator.mixed_precision  # 混合精度
    # assert config['train']['batch_size'] % accelerator.num_processes == 0
    batch_size = config['train']['batch_size'] #// accelerator.num_processes

    # if accelerator.is_main_process:  # 主进程用作模型加载，记录日志等功能
    os.makedirs(config['ckpt_root'] + data_n + '/' + task, exist_ok=True)
    os.makedirs(config['sample_dir'] + data_n + '/' + task, exist_ok=True)

    # accelerator.wait_for_everyone()

    # if accelerator.is_main_process:
    wandb.init(dir=os.path.abspath(config['workdir']), project=f'mmht_{data_n}_{task}', config=config,
               name=config['hparams'], job_type='train', mode='offline')  # 离线存储
    utils.set_logger(log_level='info', fname=os.path.join(config['workdir'], data_n +'-'+ task +'-' + exp_num + '-output.log'))
    logging.info(config)
    # else:
    #     utils.set_logger(log_level='error')  # 禁用打印
    #     builtins.print = lambda *args: None
    # logging.info(f'Run on {accelerator.num_processes} devices')

    # dataset
    dataset = get_dataset(data_n, aug=False, samples=hp_dataset, config=config) # training 所以aug设置为False
    # assert os.path.exists(dataset.fid_stat)  # fid起到啥作用？label的作用

    # train_dataset = dataset.get_split(split='train', labeled=True)
    # train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
    #                                   num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=collate_fn_dict)
    # test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    # test_dataset_loader = DataLoader(test_dataset, batch_size=config['sample']['mini_batch_size'], shuffle=True,
    #                                  drop_last=False,
    #                                  num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=collate_fn_dict)
    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                      num_workers=1, pin_memory=True, persistent_workers=True,
                                      collate_fn=collate_fn_dict)
    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config['sample']['mini_batch_size'], shuffle=True,
                                     drop_last=False,
                                     num_workers=1, pin_memory=True, persistent_workers=True,
                                     collate_fn=collate_fn_dict)

    print("Input for diffusion", train_dataset[0])
    print("AAAAAAAA", (len(train_dataset)//batch_size +1)*batch_size, len(train_dataset)) # 如果dropout为False，就不要+1

    # model define
    train_state = utils.initialize_train_state(config, device, tokenizers, len(train_dataset)) # 多一个batch
    # nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = accelerator.prepare(
    #     train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader)
    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config['ckpt_root'] + data_n + '/' + task)  # load latest ckpt

    ###################### DP-SGD #######################
    from contextlib import nullcontext

    privacy_engine = None
    use_dp = bool(config['train'].get('dp_sgd', False))

    if use_dp:
        from opacus import PrivacyEngine
        from opacus.utils.batch_memory_manager import BatchMemoryManager

        privacy_engine = PrivacyEngine(
            accountant=config['train'].get('dp_accountant', 'prv'),
            secure_mode=bool(config['train'].get('dp_secure_mode', False)),
        )

        # 先做兼容化，再挂 DP
        nnet = privacy_engine.get_compatible_module(nnet)

        # 按目标 epsilon 包装
        nnet, optimizer, train_dataset_loader = privacy_engine.make_private_with_epsilon(
            module=nnet,
            optimizer=optimizer,
            data_loader=train_dataset_loader,
            target_epsilon=float(config['train']['dp_target_epsilon']),
            target_delta=float(config['train']['dp_delta']),
            epochs=int(config['train']['n_epochs']),
            max_grad_norm=float(config['train']['dp_max_grad_norm']),
            poisson_sampling=bool(config['train'].get('dp_poisson_sampling', True)),
            grad_sample_mode=str(config['train'].get('dp_grad_sample_mode', 'functorch')),
            clipping=str(config['train'].get('dp_clipping', 'flat')),
        )

        # 同步回 train_state
        train_state.nnet = nnet


    # @ torch.cuda.amp.autocast() # 半精度

    def decode(_batch, get_logits=None, thred=config['threshod']): # 可以进一步rounding
        nnet.eval()
        hard_samples = {}
        for index, feature_key in enumerate(tokenizers.keys()):
            logits = get_logits(_batch[:,index,:], feature_key)
            array_prob = F.sigmoid(logits).detach().cpu().numpy()
            max_indices = np.argmax(array_prob, axis=1) # B
            if feature_key == 'conditions':
                truncate_indices = np.argsort(array_prob, axis=1)[:, -20:]+2 # depends on dataset
                tokens = tokenizers[feature_key].batch_decode_2d(truncate_indices.tolist()) # indices+2
            else:
                array_prob[array_prob >= thred] = 1
                array_prob[array_prob < thred] = 0  # 优化同步
                indices = []
                for index, row in enumerate(array_prob):
                    indice = (np.where(row == 1)[0]+2).tolist()
                    if len(indice) <1: # 如果没有预测到，就取最大的作为预测
                        indice = [max_indices[index]+2] # 某一行最大的那个
                    indices.append(indice)
            # indices = [(np.where(row == 1)[0]+2).tolist() for row in array_prob] # 加上padding项
                tokens = tokenizers[feature_key].batch_decode_2d(indices) # indices+2
            hard_samples[feature_key] = tokens

        return _batch, hard_samples

    def get_data_generator():
        while True:
            for indices, data in tqdm(train_dataset_loader, desc='epoch'): # disable=not accelerator.is_main_process
                yield indices, data

    data_generator = get_data_generator()  # 转为生成器

    def get_test_generator():
        while True:
            for indices, data in test_dataset_loader:
                yield indices, data

    test_generator = get_test_generator()

    # diffusion prepare
    _betas = utils.stable_diffusion_beta_schedule()
    _schedule = utils.Schedule(_betas) # forward add noise schedule
    logging.info(f'use {_schedule}')

    def cfg_nnet(x, timesteps, context):
        # classifier-free config
        _cond = nnet_ema(x, timesteps, context=context)
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet_ema(x, timesteps, context=_empty_context)
        return _cond #+ config['sample']['scale'] * (_cond - _uncond) # classifier free 要用unconditional的表征进行放缩。

    def train_step(indices, _batch, epoch=None):
        _metrics = dict()
        optimizer.zero_grad()
        # _z, _z_mask, _z_his = encode(_batch[0], _batch[1], batch[2]) # B, 3, D; B, 3, D
        # _zc, loss_c = cluster(_z) # B, 1, D, near cluster

        _z, _context, neigh_loss, _, _ = nnet.encode_everything(indices, **_batch) # B，3，D； B,9,D； 对比损失

        loss_mse, loss_round, T_loss = utils.LSimple(_z, nnet, _schedule,
                             context=_context,  **_batch)  # currently only support the extracted feature version
        _metrics['loss_mse'] = loss_mse.detach().mean() #accelerator.gather(loss.detach()).mean()
        _metrics['loss_c'] = neigh_loss.detach().mean() #accelerator.gather(neigh_loss.detach()).mean()
        _metrics['loss_rounding'] = loss_round.detach().mean() # batch
        _metrics['T_loss'] = T_loss.detach().mean() # batch
        total_loss = loss_mse.mean() + neigh_loss.mean() + loss_round.mean() + T_loss.mean() if epoch > config['warm'] else loss_mse.mean() + loss_round.mean() + T_loss.mean()
        total_loss.backward()
        # accelerator.backward(loss.mean() + neigh_loss.mean())

        if use_dp and train_state.step == 0:
            missing_grad_sample = []
            for name, p in nnet.named_parameters():
                if p.requires_grad and getattr(p, "grad_sample", None) is None:
                    missing_grad_sample.append(name)
            if missing_grad_sample:
                raise RuntimeError(f"Missing grad_sample for parameters: {missing_grad_sample}")

        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return _z, dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def dpm_solver_sample(_n_samples, _sample_steps, **kwargs):
        _z_init = torch.randn(_n_samples, *config['z_shape'], device=device) # 从正态分布中采样噪声。 # B,3,128
        z_init2 = kwargs['z_init']
        mask = kwargs['mask'].unsqueeze(dim=-1).repeat(1,1, config['z_shape'][1]) # B,3,128
        _z_init = torch.where(mask, _z_init, z_init2) # 从正态分布中采样噪声。 # B,3,128
        # 先加噪声，重新定义schedule
        noise_schedule = NoiseScheduleVP(schedule='discrete',
                                         betas=torch.tensor(_betas, device=device).float())  # 选一个加噪声的方式

        def model_fn(x, t_continuous):
            t = t_continuous * _schedule.N
            return cfg_nnet(x, t ,kwargs['context'])

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1.)  # embedding， 感觉这里需要rounding
        return decode(_z, nnet.get_logits)


    def eval_step(n_samples, sample_steps):
        """这里需要做case study"""
        # 逐步采样。这里用上了DPM加速
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=dpm_solver, '
                     f'mini_batch_size={config["sample"]["mini_batch_size"]}')

        def sample_fn(_n_samples):
            indices, batch = next(test_generator)

            assert indices.size(0) == _n_samples
            indices, sample_input = convert_input((indices, batch))
            _z, _context_emb, _, labels, hard_labels = nnet.encode_everything_test(indices, **batch)
            # _context_emb = encode(_contexts)
            return labels, hard_labels, dpm_solver_sample(_n_samples, sample_steps, context=_context_emb, z_init=_z, mask=sample_input['mask'])

        with tempfile.TemporaryDirectory() as temp_path:  # 创建临时目录，进行文件管理
            path = config['sample']['path'] + data_n + '/' + task or temp_path
            work_dir = config['workdir'] + data_n + '/' + task
            # if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
            os.makedirs(work_dir, exist_ok=True)
            utils.sample2dir(path, n_samples, config['sample']['mini_batch_size'], sample_fn, tokenizers,
                             dataset.unpreprocess) # 看看要不要改成test，因为这里是test

            _quality = 0
            # if accelerator.is_main_process:  # 采样完成后计算fid和原图的差异。metrics,可以用于可视化
            # _fid = None
            _quality, jaccard_quality = utils.calculate_quality(path)
            logging.info(f'step={train_state.step} quality-{n_samples}={_quality}')
            logging.info(f'step={train_state.step} jaccard_quality-{n_samples}={jaccard_quality}')
            with open(os.path.join(config['workdir'] + data_n + '/' + task, 'eval.log'), 'a') as f:
                print(f'step={train_state.step} quality-{n_samples}={_quality} jaccard_quality-{n_samples}={jaccard_quality}', file=f)
            wandb.log({f'quality_mse-{n_samples}': _quality}, step=train_state.step)
            _quality = torch.tensor(_quality, device=device)
            # _fid = accelerator.reduce(_fid, reduction='sum')

        return _quality.item(), jaccard_quality # 这里存储的是embeddings，一旦不进行decode就是mediffusion。

    # logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config["mixed_precision"]}')

    logging.info(f'Start fitting, step={train_state.step}')

    def convert_input(batch_data, mask_static=True):
        # padding到相同的长度便于转为tensor
        indices, batch = batch_data
        # print("batch_data 0", batch['mask'])

        feature_keys = ['conditions', 'drugs', 'procedures']
        # batch_new = {}
        for feature_key in feature_keys:
            # tokenizers
            batch[feature_key] = tokenizers[feature_key].batch_encode_2d(batch[feature_key]) # B,M,  m——feature没有用。
            batch[feature_key + '_hist'] = tokenizers[feature_key].batch_encode_3d(batch[feature_key + '_hist']) # B,V,M
            origin = batch[feature_key + '_comp'].copy()
            batch[feature_key + '_comp'] = tokenizers[feature_key].batch_encode_2d(batch[feature_key + '_comp']) # B,M
            # tensor
            batch[feature_key] = torch.tensor(batch[feature_key], dtype=torch.long, device=device)
            batch[feature_key + '_hist'] = torch.tensor(batch[feature_key + '_hist'], dtype=torch.long, device=device)
            batch[feature_key + '_comp'] = torch.tensor(batch[feature_key + '_comp'], dtype=torch.long, device=device)
            # mask
            batch[feature_key + '_mask'] = batch[feature_key] != 0 # 为True的就是mask掉的, # B,M
            batch[feature_key + '_hist_mask'] = batch[feature_key + '_hist'] != 0 # 这个是为了padding, # B,V,M
            batch[feature_key + '_comp_mask'] = batch[feature_key + '_comp'] != 0 # 这个是为了padding # B,M

            # others
            batch[feature_key + '_comps'] = tokenizers[feature_key].batch_encode_2d(origin, padding=False,  truncation=False) # 不需要减去2,只需要能看清楚是什么就行，因为decode的时候会加上2
            batch[feature_key + '_comps_origin'] = origin # origin ID

            # note special
            batch['has_note'] = False
            ## 20251210注释掉"if config_health['DATASET'] == 'MIV-Note':"
            # if config_health['DATASET'] == 'MIV-Note': # LOS MIV修改下。
            #     batch['note'] = torch.tensor(batch['note'], device=device) # cur, B,D; 这里不需要对他进行重建，因为他可能变为pad字符串
            #     batch['note' + '_hist'] = pad_list(batch['note' + '_hist'], device=device)# hist, B,T,D
            #     batch['note' + '_hist_mask'] = batch['note' + '_hist'].sum(dim=-1) !=0 # B,T
            #
            #     # 作为context,需要合并
            #     batch['note' + '_hist'] = torch.cat([batch['note' + '_hist'], batch['note'].unsqueeze(dim=1)], dim=1)  # B,T+1,D
            #     batch['note' + '_mask'] = batch['note'].sum(dim=-1) != 0  # B,
            #     batch['note' + '_hist_mask'] = torch.cat(
            #         [batch['note' + '_hist_mask'], batch['note' + '_mask'].unsqueeze(dim=1)], dim=1)  # B,T+1
            #
            #     batch['has_note'] = True

        # mask设置为固定
        if mask_static:
            batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
        # mask不固定, evaldffs固定就好, diverse
        else:
            m = 3 # 4 if batch['has_note'] else 3
            batch['mask'] = utils.generate_mask(batch['conditions'].shape[0],m=m).to(device=device)


        aligned_data = [batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1) for feature_key in feature_keys] # B,V,1
        batch['mask_hist'] = torch.cat(aligned_data, dim=-1)  # B, V, 3，记录history mask状态

        return indices, batch


    ########## 加入训练集结果统计 #########
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=config['sample']['mini_batch_size'],  # 用mini_batch_size评估更省
        shuffle=False,  # 固定顺序更稳定可比
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_dict
    )

    def get_train_eval_generator():
        while True:
            for indices, data in train_eval_loader:
                yield indices, data

    train_eval_generator = get_train_eval_generator()

    def train_epoch_metrics(sample_steps, num_batches=2, mask_static=True):
        """
        在 train split 上抽 num_batches 个 batch，计算平均 MSE 和 (Jc,Jp,Jd)
        """
        nnet.eval()
        mse_list = []
        jc_list, jp_list, jd_list = [], [], []

        with torch.no_grad():
            for _ in range(num_batches):
                idx, b = next(train_eval_generator)
                idx, b = convert_input((idx, b), mask_static=mask_static)

                # encode + GT
                z0, contexts, _, labels, hard_labels = nnet.encode_everything_test(idx, **b)

                # sample + decode
                samples, hard_samples = dpm_solver_sample(
                    _n_samples=idx.size(0),
                    _sample_steps=sample_steps,
                    context=contexts,
                    z_init=z0,
                    mask=b['mask']
                )

                # MSE（你中途输出那段也是这样算的）
                mse_list.append(F.mse_loss(samples, labels.squeeze()).item())

                # Jaccard（三视图）
                gt_cond = tokenizers['conditions'].batch_decode_2d(hard_labels['conditions'])
                gt_proc = tokenizers['procedures'].batch_decode_2d(hard_labels['procedures'])
                gt_drug = tokenizers['drugs'].batch_decode_2d(hard_labels['drugs'])

                jc_list.append(utils.calculate_average_jaccard(hard_samples['conditions'], gt_cond))
                jp_list.append(utils.calculate_average_jaccard(hard_samples['procedures'], gt_proc))
                jd_list.append(utils.calculate_average_jaccard(hard_samples['drugs'], gt_drug))

        nnet.train()

        train_mse = float(np.mean(mse_list))
        train_jacc = (float(np.mean(jc_list)), float(np.mean(jp_list)), float(np.mean(jd_list)))
        return train_mse, train_jacc
    ######################################################
    step_quality = []
    jaccard_quality = []
    # while train_state.step < config['train']['n_steps']:
    # max_n_steps = (len(train_dataset) // config['train']['batch_size']) + 1 if len(train_dataset) % config['train']['batch_size'] !=0 else len(train_dataset) // config['train']['batch_size'] # 不对，如果正好整除的话，就不需要+1
    max_n_steps = len(train_dataset_loader)
    for epoch in range(config['train']['n_epochs']):
        if epoch == 0:
            nnet.e_step()

        total_visit_embs = []
        total_indices = []

        if use_dp:
            loader_cm = BatchMemoryManager(
                data_loader=train_dataset_loader,
                max_physical_batch_size=int(config['train']['dp_physical_batch_size']),
                optimizer=optimizer,
            )
        else:
            loader_cm = nullcontext(train_dataset_loader)

        with loader_cm as epoch_loader:
            for indices, raw_batch in epoch_loader:
                nnet.train()
                indices, batch = convert_input((indices, raw_batch), mask_static=False)
                _z, metrics = train_step(indices, batch, epoch)

                total_visit_embs.append(_z.view(_z.shape[0], -1))
                total_indices.append(indices)

                nnet.eval()
                # 记录metrics, 输出train_state.step, 以及config.train.log_interval
                # if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
                if train_state.step % config['train']['log_interval'] == 0:
                    logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
                    logging.info(config['workdir'])
                    wandb.log(metrics, step=train_state.step) # 不同step

                # 中途输出
                # if train_state.step % config['train']['eval_interval'] == 0:  # 任何记录日志，保存模型都要使用
                #     torch.cuda.empty_cache()
                    logging.info('Save a grid of {} samples for training...'.format(config['train_val_num']))
                    sample_input = dataset.get_contexts(num=config['train_val_num'])
                    indices = torch.arange(config['train_val_num']).to(device)
                    indices, sample_input = convert_input((indices, sample_input))
                    _z, contexts, _, labels, hard_labels = nnet.encode_everything_test(indices, **sample_input)

                    # contexts = torch.tensor(dataset.contexts, device=device)[: 2 * 5] # 选取前10个context
                    samples, hard_samples = dpm_solver_sample(_n_samples=config['train_val_num'], _sample_steps=50, context=contexts, z_init=_z, mask=sample_input['mask'])
                    # print("generate", hard_samples['conditions'][0])
                    # print("origin", tokenizers['conditions'].batch_decode_2d(hard_labels['conditions'])[0]) # 没啥问题，encode, decode用同一套
                    # print("generate", hard_samples['procedures'][0])
                    # print("origin", tokenizers['procedures'].batch_decode_2d(hard_labels['procedures'])[0])  # 没啥问题，encode, decode用同一套
                    # print("generate", hard_samples['drugs'][0])
                    # print("origin", tokenizers['drugs'].batch_decode_2d(hard_labels['drugs'])[0])  # 没啥问题，encode, decode用同一套
                    # samples = make_grid(dataset.unpreprocess(samples), 5)
                    # save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
                    # wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
                    jaccard_condition = utils.calculate_average_jaccard(hard_samples['conditions'], tokenizers['conditions'].batch_decode_2d(hard_labels['conditions']))
                    jaccard_procedure = utils.calculate_average_jaccard(hard_samples['procedures'], tokenizers['procedures'].batch_decode_2d(hard_labels['procedures']))
                    jaccard_drug = utils.calculate_average_jaccard(hard_samples['drugs'], tokenizers['drugs'].batch_decode_2d(hard_labels['drugs']))
                    mse_loss = F.mse_loss(samples, labels.squeeze())
                    logging.info({'mask num': sample_input['mask'].sum(dim=0), 'val samples mse': mse_loss, 'jaccard': (jaccard_condition,jaccard_procedure,jaccard_drug)})
                    # wandb.log({'samples': mse_loss }, step=train_state.step)
                    # torch.cuda.empty_cache()
                # accelerator.wait_for_everyone()

            # save ckpt， 这一块代码检查当前进程是否为主进程（accelerator.is_main_process），如果是则保存模型
        # if train_state.step % config['train']['save_interval'] == 0 or train_state.step == config['train']['n_steps']:
        # torch.cuda.empty_cache()

        ###################
        train_mse_ep, train_jacc_ep = train_epoch_metrics(
            sample_steps=config['sample']['sample_steps'],
            num_batches=config['train'].get('train_eval_batches', 2),
            mask_static=True  # 更稳定；想贴近训练mask可设False
        )

        logging.info(f'[TRAIN EVAL] epoch={epoch} train_mse={train_mse_ep} train_jacc={train_jacc_ep}')
        wandb.log({
            'train_epoch/mse': train_mse_ep,
            'train_epoch/jaccard_conditions': train_jacc_ep[0],
            'train_epoch/jaccard_procedures': train_jacc_ep[1],
            'train_epoch/jaccard_drugs': train_jacc_ep[2],
            'train_epoch/jaccard_mean': (train_jacc_ep[0] + train_jacc_ep[1] + train_jacc_ep[2]) / 3.0,
        }, step=train_state.step)
        ###################

        logging.info(f'Save and eval checkpoint {epoch}...')
        # if accelerator.local_process_index == 0:
        train_state.save(os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch}.ckpt'))
        # accelerator.wait_for_everyone()
        if epoch >=0 or (epoch>=5 and epoch % config['train']['eval_epoch'] == 0):
            # quality, jaccard_tuple = eval_step(n_samples=len(test_dataset), sample_steps=50)  # calculate fid of the saved checkpoint
            quality, jaccard_tuple = eval_step(n_samples=len(test_dataset), sample_steps=config['sample']['sample_steps'])
        else:
            quality = 10000 # 因为靠mse取最小的mse
            jaccard_tuple = (1,1,1)
        step_quality.append((epoch, quality))
        jaccard_quality.append((epoch, jaccard_tuple))
        # torch.cuda.empty_cache()

        # e step
        total_visit_embs = torch.cat(total_visit_embs, dim=0)
        total_indices = torch.cat(total_indices, dim=0)
        total_visit_embs = total_visit_embs[total_indices.argsort()] # 重新排序， 这里需要check下对不对。
        print("CCCCCCCC", total_visit_embs.shape)
        nnet.update_train_visit_emb(total_visit_embs)
        visit_centroids, _ = nnet.e_step()
        nnet.update_centroid_emb(visit_centroids)

        if use_dp and privacy_engine is not None:
            eps = privacy_engine.get_epsilon(float(config['train']['dp_delta']))
            logging.info(
                f"[DP] epoch={epoch} epsilon={eps:.6f} delta={config['train']['dp_delta']}"
            )
            wandb.log({"dp/epsilon": eps}, step=train_state.step)



    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_quality: {step_quality}')
    logging.info(f'step_quality: {jaccard_quality}')
    epoch_best = sorted(step_quality, key=lambda x: x[1])[0][0] # 最小那个mse
    logging.info(f'step_best: {epoch_best}')
    train_state.load(os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch_best}.ckpt'))
    del metrics
    # accelerator.wait_for_everyone()
    eval_step(n_samples=len(test_dataset), sample_steps=config['sample']['sample_steps'])  # 选取最好的ckpt

    nnet_path = os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch_best}.ckpt/nnet.pth')
    return nnet_path, train_state.nnet, train_state.nnet.total_visit_emb.shape[0]




def run_diffusion(config, config_health, hp_dataset, tokenizers, exp_num='0'):
    """
    :param config: diffusion config
    :param config_health: healthcare config
    :param hp_dataset: healthcare train dataset
    :param tokenizers: healthcare tokenizers
    :return:
    """
    print("run_diffusion")
    set_random_seed(config_health['SEED'])

    data_n = config_health['DATASET']
    task = config_health['TASK']

    # utils.calculate_quality(config['sample']['path'] + data_n)

    # 加速
    # accelerator = accelerate.Accelerator() # CUDA_VISIBLE_DEVICES=4  accelerate launch main.py
    # device = accelerator.device
    # accelerate.utils.set_seed(config['seed'], device_specific=True)
    device = torch.device('cuda:' + str(config['device']) if torch.cuda.is_available() else 'cpu')

    # logging.info(f'Process {accelerator.process_index} using device: {device}')
    logging.info(f'using device: {device}')

    # config['mixed_precision'] = accelerator.mixed_precision  # 混合精度
    # assert config['train']['batch_size'] % accelerator.num_processes == 0
    batch_size = config['train']['batch_size'] #// accelerator.num_processes

    # if accelerator.is_main_process:  # 主进程用作模型加载，记录日志等功能
    os.makedirs(config['ckpt_root'] + data_n + '/' + task, exist_ok=True)
    os.makedirs(config['sample_dir'] + data_n + '/' + task, exist_ok=True)

    # accelerator.wait_for_everyone()

    # if accelerator.is_main_process:
    wandb.init(dir=os.path.abspath(config['workdir']), project=f'mmht_{data_n}_{task}', config=config,
               name=config['hparams'], job_type='train', mode='offline')  # 离线存储
    utils.set_logger(log_level='info', fname=os.path.join(config['workdir'], data_n +'-'+ task +'-' + exp_num + '-output.log'))
    logging.info(config)
    # else:
    #     utils.set_logger(log_level='error')  # 禁用打印
    #     builtins.print = lambda *args: None
    # logging.info(f'Run on {accelerator.num_processes} devices')

    # dataset
    dataset = get_dataset(data_n, aug=False, samples=hp_dataset, config=config) # training 所以aug设置为False
    # assert os.path.exists(dataset.fid_stat)  # fid起到啥作用？label的作用

    # train_dataset = dataset.get_split(split='train', labeled=True)
    # train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
    #                                   num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=collate_fn_dict)
    # test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    # test_dataset_loader = DataLoader(test_dataset, batch_size=config['sample']['mini_batch_size'], shuffle=True,
    #                                  drop_last=False,
    #                                  num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=collate_fn_dict)
    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                      num_workers=1, pin_memory=True, persistent_workers=True,
                                      collate_fn=collate_fn_dict)
    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config['sample']['mini_batch_size'], shuffle=True,
                                     drop_last=False,
                                     num_workers=1, pin_memory=True, persistent_workers=True,
                                     collate_fn=collate_fn_dict)

    print("Input for diffusion", train_dataset[0])
    print("AAAAAAAA", (len(train_dataset)//batch_size +1)*batch_size, len(train_dataset)) # 如果dropout为False，就不要+1

    # model define
    train_state = utils.initialize_train_state(config, device, tokenizers, len(train_dataset)) # 多一个batch
    # nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = accelerator.prepare(
    #     train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader)
    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config['ckpt_root'] + data_n + '/' + task)  # load latest ckpt


    # @ torch.cuda.amp.autocast() # 半精度

    def decode(_batch, get_logits=None, thred=config['threshod']): # 可以进一步rounding
        nnet.eval()
        hard_samples = {}
        for index, feature_key in enumerate(tokenizers.keys()):
            logits = get_logits(_batch[:,index,:], feature_key)
            array_prob = F.sigmoid(logits).detach().cpu().numpy()
            max_indices = np.argmax(array_prob, axis=1) # B
            if feature_key == 'conditions':
                truncate_indices = np.argsort(array_prob, axis=1)[:, -20:]+2 # depends on dataset
                tokens = tokenizers[feature_key].batch_decode_2d(truncate_indices.tolist()) # indices+2
            else:
                array_prob[array_prob >= thred] = 1
                array_prob[array_prob < thred] = 0  # 优化同步
                indices = []
                for index, row in enumerate(array_prob):
                    indice = (np.where(row == 1)[0]+2).tolist()
                    if len(indice) <1: # 如果没有预测到，就取最大的作为预测
                        indice = [max_indices[index]+2] # 某一行最大的那个
                    indices.append(indice)
            # indices = [(np.where(row == 1)[0]+2).tolist() for row in array_prob] # 加上padding项
                tokens = tokenizers[feature_key].batch_decode_2d(indices) # indices+2
            hard_samples[feature_key] = tokens

        return _batch, hard_samples

    def get_data_generator():
        while True:
            for indices, data in tqdm(train_dataset_loader, desc='epoch'): # disable=not accelerator.is_main_process
                yield indices, data

    data_generator = get_data_generator()  # 转为生成器

    def get_test_generator():
        while True:
            for indices, data in test_dataset_loader:
                yield indices, data

    test_generator = get_test_generator()

    # diffusion prepare
    _betas = utils.stable_diffusion_beta_schedule()
    _schedule = utils.Schedule(_betas) # forward add noise schedule
    logging.info(f'use {_schedule}')

    def cfg_nnet(x, timesteps, context):
        # classifier-free config
        _cond = nnet_ema(x, timesteps, context=context)
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet_ema(x, timesteps, context=_empty_context)
        return _cond #+ config['sample']['scale'] * (_cond - _uncond) # classifier free 要用unconditional的表征进行放缩。

    def train_step(indices, _batch, epoch=None):
        _metrics = dict()
        optimizer.zero_grad()
        # _z, _z_mask, _z_his = encode(_batch[0], _batch[1], batch[2]) # B, 3, D; B, 3, D
        # _zc, loss_c = cluster(_z) # B, 1, D, near cluster

        _z, _context, neigh_loss, _, _ = nnet.encode_everything(indices, **_batch) # B，3，D； B,9,D； 对比损失

        loss_mse, loss_round, T_loss = utils.LSimple(_z, nnet, _schedule,
                             context=_context,  **_batch)  # currently only support the extracted feature version
        _metrics['loss_mse'] = loss_mse.detach().mean() #accelerator.gather(loss.detach()).mean()
        _metrics['loss_c'] = neigh_loss.detach().mean() #accelerator.gather(neigh_loss.detach()).mean()
        _metrics['loss_rounding'] = loss_round.detach().mean() # batch
        _metrics['T_loss'] = T_loss.detach().mean() # batch
        total_loss = loss_mse.mean() + neigh_loss.mean() + loss_round.mean() + T_loss.mean() if epoch > config['warm'] else loss_mse.mean() + loss_round.mean() + T_loss.mean()
        total_loss.backward()
        # accelerator.backward(loss.mean() + neigh_loss.mean())

        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return _z, dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def dpm_solver_sample(_n_samples, _sample_steps, **kwargs):
        _z_init = torch.randn(_n_samples, *config['z_shape'], device=device) # 从正态分布中采样噪声。 # B,3,128
        z_init2 = kwargs['z_init']
        mask = kwargs['mask'].unsqueeze(dim=-1).repeat(1,1, config['z_shape'][1]) # B,3,128
        _z_init = torch.where(mask, _z_init, z_init2) # 从正态分布中采样噪声。 # B,3,128
        # 先加噪声，重新定义schedule
        noise_schedule = NoiseScheduleVP(schedule='discrete',
                                         betas=torch.tensor(_betas, device=device).float())  # 选一个加噪声的方式

        def model_fn(x, t_continuous):
            t = t_continuous * _schedule.N
            return cfg_nnet(x, t ,kwargs['context'])

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1.)  # embedding， 感觉这里需要rounding
        return decode(_z, nnet.get_logits)


    def eval_step(n_samples, sample_steps):
        """这里需要做case study"""
        # 逐步采样。这里用上了DPM加速
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=dpm_solver, '
                     f'mini_batch_size={config["sample"]["mini_batch_size"]}')

        def sample_fn(_n_samples):
            indices, batch = next(test_generator)

            assert indices.size(0) == _n_samples
            indices, sample_input = convert_input((indices, batch))
            _z, _context_emb, _, labels, hard_labels = nnet.encode_everything_test(indices, **batch)
            # _context_emb = encode(_contexts)
            return labels, hard_labels, dpm_solver_sample(_n_samples, sample_steps, context=_context_emb, z_init=_z, mask=sample_input['mask'])

        with tempfile.TemporaryDirectory() as temp_path:  # 创建临时目录，进行文件管理
            path = config['sample']['path'] + data_n + '/' + task or temp_path
            work_dir = config['workdir'] + data_n + '/' + task
            # if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
            os.makedirs(work_dir, exist_ok=True)
            utils.sample2dir(path, n_samples, config['sample']['mini_batch_size'], sample_fn, tokenizers,
                             dataset.unpreprocess) # 看看要不要改成test，因为这里是test

            _quality = 0
            # if accelerator.is_main_process:  # 采样完成后计算fid和原图的差异。metrics,可以用于可视化
            # _fid = None
            _quality, jaccard_quality = utils.calculate_quality(path)
            logging.info(f'step={train_state.step} quality-{n_samples}={_quality}')
            logging.info(f'step={train_state.step} jaccard_quality-{n_samples}={jaccard_quality}')
            with open(os.path.join(config['workdir'] + data_n + '/' + task, 'eval.log'), 'a') as f:
                print(f'step={train_state.step} quality-{n_samples}={_quality} jaccard_quality-{n_samples}={jaccard_quality}', file=f)
            wandb.log({f'quality_mse-{n_samples}': _quality}, step=train_state.step)
            _quality = torch.tensor(_quality, device=device)
            # _fid = accelerator.reduce(_fid, reduction='sum')

        return _quality.item(), jaccard_quality # 这里存储的是embeddings，一旦不进行decode就是mediffusion。

    # logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config["mixed_precision"]}')

    logging.info(f'Start fitting, step={train_state.step}')

    def convert_input(batch_data, mask_static=True):
        # padding到相同的长度便于转为tensor
        indices, batch = batch_data
        # print("batch_data 0", batch['mask'])

        feature_keys = ['conditions', 'drugs', 'procedures']
        # batch_new = {}
        for feature_key in feature_keys:
            # tokenizers
            batch[feature_key] = tokenizers[feature_key].batch_encode_2d(batch[feature_key]) # B,M,  m——feature没有用。
            batch[feature_key + '_hist'] = tokenizers[feature_key].batch_encode_3d(batch[feature_key + '_hist']) # B,V,M
            origin = batch[feature_key + '_comp'].copy()
            batch[feature_key + '_comp'] = tokenizers[feature_key].batch_encode_2d(batch[feature_key + '_comp']) # B,M
            # tensor
            batch[feature_key] = torch.tensor(batch[feature_key], dtype=torch.long, device=device)
            batch[feature_key + '_hist'] = torch.tensor(batch[feature_key + '_hist'], dtype=torch.long, device=device)
            batch[feature_key + '_comp'] = torch.tensor(batch[feature_key + '_comp'], dtype=torch.long, device=device)
            # mask
            batch[feature_key + '_mask'] = batch[feature_key] != 0 # 为True的就是mask掉的, # B,M
            batch[feature_key + '_hist_mask'] = batch[feature_key + '_hist'] != 0 # 这个是为了padding, # B,V,M
            batch[feature_key + '_comp_mask'] = batch[feature_key + '_comp'] != 0 # 这个是为了padding # B,M

            # others
            batch[feature_key + '_comps'] = tokenizers[feature_key].batch_encode_2d(origin, padding=False,  truncation=False) # 不需要减去2,只需要能看清楚是什么就行，因为decode的时候会加上2
            batch[feature_key + '_comps_origin'] = origin # origin ID

            # note special
            batch['has_note'] = False
            ## 20251210注释掉"if config_health['DATASET'] == 'MIV-Note':"
            # if config_health['DATASET'] == 'MIV-Note': # LOS MIV修改下。
            #     batch['note'] = torch.tensor(batch['note'], device=device) # cur, B,D; 这里不需要对他进行重建，因为他可能变为pad字符串
            #     batch['note' + '_hist'] = pad_list(batch['note' + '_hist'], device=device)# hist, B,T,D
            #     batch['note' + '_hist_mask'] = batch['note' + '_hist'].sum(dim=-1) !=0 # B,T
            #
            #     # 作为context,需要合并
            #     batch['note' + '_hist'] = torch.cat([batch['note' + '_hist'], batch['note'].unsqueeze(dim=1)], dim=1)  # B,T+1,D
            #     batch['note' + '_mask'] = batch['note'].sum(dim=-1) != 0  # B,
            #     batch['note' + '_hist_mask'] = torch.cat(
            #         [batch['note' + '_hist_mask'], batch['note' + '_mask'].unsqueeze(dim=1)], dim=1)  # B,T+1
            #
            #     batch['has_note'] = True

        # mask设置为固定
        if mask_static:
            batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
        # mask不固定, evaldffs固定就好, diverse
        else:
            m = 3 # 4 if batch['has_note'] else 3
            batch['mask'] = utils.generate_mask(batch['conditions'].shape[0],m=m).to(device=device)


        aligned_data = [batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1) for feature_key in feature_keys] # B,V,1
        batch['mask_hist'] = torch.cat(aligned_data, dim=-1)  # B, V, 3，记录history mask状态

        return indices, batch


    ########## 加入训练集结果统计 #########
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=config['sample']['mini_batch_size'],  # 用mini_batch_size评估更省
        shuffle=False,  # 固定顺序更稳定可比
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_dict
    )

    def get_train_eval_generator():
        while True:
            for indices, data in train_eval_loader:
                yield indices, data

    train_eval_generator = get_train_eval_generator()

    def train_epoch_metrics(sample_steps, num_batches=2, mask_static=True):
        """
        在 train split 上抽 num_batches 个 batch，计算平均 MSE 和 (Jc,Jp,Jd)
        """
        nnet.eval()
        mse_list = []
        jc_list, jp_list, jd_list = [], [], []

        with torch.no_grad():
            for _ in range(num_batches):
                idx, b = next(train_eval_generator)
                idx, b = convert_input((idx, b), mask_static=mask_static)

                # encode + GT
                z0, contexts, _, labels, hard_labels = nnet.encode_everything_test(idx, **b)

                # sample + decode
                samples, hard_samples = dpm_solver_sample(
                    _n_samples=idx.size(0),
                    _sample_steps=sample_steps,
                    context=contexts,
                    z_init=z0,
                    mask=b['mask']
                )

                # MSE（你中途输出那段也是这样算的）
                mse_list.append(F.mse_loss(samples, labels.squeeze()).item())

                # Jaccard（三视图）
                gt_cond = tokenizers['conditions'].batch_decode_2d(hard_labels['conditions'])
                gt_proc = tokenizers['procedures'].batch_decode_2d(hard_labels['procedures'])
                gt_drug = tokenizers['drugs'].batch_decode_2d(hard_labels['drugs'])

                jc_list.append(utils.calculate_average_jaccard(hard_samples['conditions'], gt_cond))
                jp_list.append(utils.calculate_average_jaccard(hard_samples['procedures'], gt_proc))
                jd_list.append(utils.calculate_average_jaccard(hard_samples['drugs'], gt_drug))

        nnet.train()

        train_mse = float(np.mean(mse_list))
        train_jacc = (float(np.mean(jc_list)), float(np.mean(jp_list)), float(np.mean(jd_list)))
        return train_mse, train_jacc
    ######################################################
    step_quality = []
    jaccard_quality = []
    # while train_state.step < config['train']['n_steps']:
    max_n_steps = (len(train_dataset) // config['train']['batch_size']) + 1 if len(train_dataset) % config['train']['batch_size'] !=0 else len(train_dataset) // config['train']['batch_size'] # 不对，如果正好整除的话，就不需要+1
    for epoch in range(config['train']['n_epochs']):
        if epoch == 0:
            nnet.e_step()  # 初始化
        # print("AAAAAAAAA", nnet.total_visit_emb.shape, max_n_steps, len(train_dataset))
        total_visit_embs = []
        total_indices = []
        for i in range(max_n_steps):
            nnet.train()
            # batch = tree_map(lambda x: convert_input(x), next(data_generator))  # tree_map更快
            indices, batch = convert_input(next(data_generator), mask_static=False)

            _z, metrics = train_step(indices, batch, epoch)

            total_visit_embs.append(_z.view(_z.shape[0],-1)) # B, 3D
            total_indices.append(indices) # B,1

            nnet.eval()
            # 记录metrics, 输出train_state.step, 以及config.train.log_interval
            # if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            if train_state.step % config['train']['log_interval'] == 0:
                logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
                logging.info(config['workdir'])
                wandb.log(metrics, step=train_state.step) # 不同step

            # 中途输出
            # if train_state.step % config['train']['eval_interval'] == 0:  # 任何记录日志，保存模型都要使用
            #     torch.cuda.empty_cache()
                logging.info('Save a grid of {} samples for training...'.format(config['train_val_num']))
                sample_input = dataset.get_contexts(num=config['train_val_num'])
                indices = torch.arange(config['train_val_num']).to(device)
                indices, sample_input = convert_input((indices, sample_input))
                _z, contexts, _, labels, hard_labels = nnet.encode_everything_test(indices, **sample_input)

                # contexts = torch.tensor(dataset.contexts, device=device)[: 2 * 5] # 选取前10个context
                samples, hard_samples = dpm_solver_sample(_n_samples=config['train_val_num'], _sample_steps=50, context=contexts, z_init=_z, mask=sample_input['mask'])
                # print("generate", hard_samples['conditions'][0])
                # print("origin", tokenizers['conditions'].batch_decode_2d(hard_labels['conditions'])[0]) # 没啥问题，encode, decode用同一套
                # print("generate", hard_samples['procedures'][0])
                # print("origin", tokenizers['procedures'].batch_decode_2d(hard_labels['procedures'])[0])  # 没啥问题，encode, decode用同一套
                # print("generate", hard_samples['drugs'][0])
                # print("origin", tokenizers['drugs'].batch_decode_2d(hard_labels['drugs'])[0])  # 没啥问题，encode, decode用同一套
                # samples = make_grid(dataset.unpreprocess(samples), 5)
                # save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
                # wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
                jaccard_condition = utils.calculate_average_jaccard(hard_samples['conditions'], tokenizers['conditions'].batch_decode_2d(hard_labels['conditions']))
                jaccard_procedure = utils.calculate_average_jaccard(hard_samples['procedures'], tokenizers['procedures'].batch_decode_2d(hard_labels['procedures']))
                jaccard_drug = utils.calculate_average_jaccard(hard_samples['drugs'], tokenizers['drugs'].batch_decode_2d(hard_labels['drugs']))
                mse_loss = F.mse_loss(samples, labels.squeeze())
                logging.info({'mask num': sample_input['mask'].sum(dim=0), 'val samples mse': mse_loss, 'jaccard': (jaccard_condition,jaccard_procedure,jaccard_drug)})
                # wandb.log({'samples': mse_loss }, step=train_state.step)
                # torch.cuda.empty_cache()
            # accelerator.wait_for_everyone()

            # save ckpt， 这一块代码检查当前进程是否为主进程（accelerator.is_main_process），如果是则保存模型
        # if train_state.step % config['train']['save_interval'] == 0 or train_state.step == config['train']['n_steps']:
        # torch.cuda.empty_cache()

        ###################
        train_mse_ep, train_jacc_ep = train_epoch_metrics(
            sample_steps=config['sample']['sample_steps'],
            num_batches=config['train'].get('train_eval_batches', 2),
            mask_static=True  # 更稳定；想贴近训练mask可设False
        )

        logging.info(f'[TRAIN EVAL] epoch={epoch} train_mse={train_mse_ep} train_jacc={train_jacc_ep}')
        wandb.log({
            'train_epoch/mse': train_mse_ep,
            'train_epoch/jaccard_conditions': train_jacc_ep[0],
            'train_epoch/jaccard_procedures': train_jacc_ep[1],
            'train_epoch/jaccard_drugs': train_jacc_ep[2],
            'train_epoch/jaccard_mean': (train_jacc_ep[0] + train_jacc_ep[1] + train_jacc_ep[2]) / 3.0,
        }, step=train_state.step)
        ###################

        logging.info(f'Save and eval checkpoint {epoch}...')
        # if accelerator.local_process_index == 0:
        train_state.save(os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch}.ckpt'))
        # accelerator.wait_for_everyone()
        if epoch==0 or (epoch>=5 and epoch % config['train']['eval_epoch'] == 0):
            # quality, jaccard_tuple = eval_step(n_samples=len(test_dataset), sample_steps=50)  # calculate fid of the saved checkpoint
            quality, jaccard_tuple = eval_step(n_samples=len(test_dataset), sample_steps=config['sample']['sample_steps'])
        else:
            quality = 10000 # 因为靠mse取最小的mse
            jaccard_tuple = (1,1,1)
        step_quality.append((epoch, quality))
        jaccard_quality.append((epoch, jaccard_tuple))
        # torch.cuda.empty_cache()

        # e step
        total_visit_embs = torch.cat(total_visit_embs, dim=0)
        total_indices = torch.cat(total_indices, dim=0)
        total_visit_embs = total_visit_embs[total_indices.argsort()] # 重新排序， 这里需要check下对不对。
        print("CCCCCCCC", total_visit_embs.shape)
        nnet.update_train_visit_emb(total_visit_embs)
        visit_centroids, _ = nnet.e_step()
        nnet.update_centroid_emb(visit_centroids)



    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_quality: {step_quality}')
    logging.info(f'step_quality: {jaccard_quality}')
    epoch_best = sorted(step_quality, key=lambda x: x[1])[0][0] # 最小那个mse
    logging.info(f'step_best: {epoch_best}')
    train_state.load(os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch_best}.ckpt'))
    del metrics
    # accelerator.wait_for_everyone()
    eval_step(n_samples=len(test_dataset), sample_steps=config['sample']['sample_steps'])  # 选取最好的ckpt

    nnet_path = os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch_best}.ckpt/nnet.pth')
    return nnet_path, train_state.nnet, train_state.nnet.total_visit_emb.shape[0]


def run_diffusion1(config, config_health, hp_dataset, tokenizers, exp_num='0'):
    """
    :param config: diffusion config
    :param config_health: healthcare config
    :param hp_dataset: healthcare train dataset
    :param tokenizers: healthcare tokenizers
    :return:
    """
    print("run_diffusion")
    set_random_seed(config_health['SEED'])

    data_n = config_health['DATASET']
    task = config_health['TASK']

    # utils.calculate_quality(config['sample']['path'] + data_n)

    # 加速
    # accelerator = accelerate.Accelerator() # CUDA_VISIBLE_DEVICES=4  accelerate launch main.py
    # device = accelerator.device
    # accelerate.utils.set_seed(config['seed'], device_specific=True)
    device = torch.device('cuda:' + str(config['device']) if torch.cuda.is_available() else 'cpu')

    # logging.info(f'Process {accelerator.process_index} using device: {device}')
    logging.info(f'using device: {device}')

    # config['mixed_precision'] = accelerator.mixed_precision  # 混合精度
    # assert config['train']['batch_size'] % accelerator.num_processes == 0
    batch_size = config['train']['batch_size'] #// accelerator.num_processes

    # if accelerator.is_main_process:  # 主进程用作模型加载，记录日志等功能
    os.makedirs(config['ckpt_root'] + data_n + '/' + task, exist_ok=True)
    os.makedirs(config['sample_dir'] + data_n + '/' + task, exist_ok=True)

    # accelerator.wait_for_everyone()

    # if accelerator.is_main_process:
    wandb.init(dir=os.path.abspath(config['workdir']), project=f'mmht_{data_n}_{task}', config=config,
               name=config['hparams'], job_type='train', mode='offline')  # 离线存储
    utils.set_logger(log_level='info', fname=os.path.join(config['workdir'], data_n +'-'+ task +'-' + exp_num + '-output.log'))
    logging.info(config)
    # else:
    #     utils.set_logger(log_level='error')  # 禁用打印
    #     builtins.print = lambda *args: None
    # logging.info(f'Run on {accelerator.num_processes} devices')

    # dataset
    dataset = get_dataset(data_n, aug=False, samples=hp_dataset, config=config) # training 所以aug设置为False
    # assert os.path.exists(dataset.fid_stat)  # fid起到啥作用？label的作用

    # train_dataset = dataset.get_split(split='train', labeled=True)
    # train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
    #                                   num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=collate_fn_dict)
    # test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    # test_dataset_loader = DataLoader(test_dataset, batch_size=config['sample']['mini_batch_size'], shuffle=True,
    #                                  drop_last=False,
    #                                  num_workers=8, pin_memory=True, persistent_workers=True, collate_fn=collate_fn_dict)
    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                      num_workers=1, pin_memory=True, persistent_workers=True,
                                      collate_fn=collate_fn_dict)
    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config['sample']['mini_batch_size'], shuffle=True,
                                     drop_last=False,
                                     num_workers=1, pin_memory=True, persistent_workers=True,
                                     collate_fn=collate_fn_dict)

    print("Input for diffusion", train_dataset[0])
    print("AAAAAAAA", (len(train_dataset)//batch_size +1)*batch_size, len(train_dataset)) # 如果dropout为False，就不要+1

    # model define
    train_state = utils.initialize_train_state(config, device, tokenizers, len(train_dataset)) # 多一个batch
    # nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = accelerator.prepare(
    #     train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader)
    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config['ckpt_root'] + data_n + '/' + task)  # load latest ckpt


    # @ torch.cuda.amp.autocast() # 半精度

    def decode(_batch, get_logits=None, thred=config['threshod']): # 可以进一步rounding
        nnet.eval()
        hard_samples = {}
        for index, feature_key in enumerate(tokenizers.keys()):
            logits = get_logits(_batch[:,index,:], feature_key)
            array_prob = F.sigmoid(logits).detach().cpu().numpy()
            max_indices = np.argmax(array_prob, axis=1) # B
            if feature_key == 'conditions':
                truncate_indices = np.argsort(array_prob, axis=1)[:, -20:]+2 # depends on dataset
                tokens = tokenizers[feature_key].batch_decode_2d(truncate_indices.tolist()) # indices+2
            else:
                array_prob[array_prob >= thred] = 1
                array_prob[array_prob < thred] = 0  # 优化同步
                indices = []
                for index, row in enumerate(array_prob):
                    indice = (np.where(row == 1)[0]+2).tolist()
                    if len(indice) <1: # 如果没有预测到，就取最大的作为预测
                        indice = [max_indices[index]+2] # 某一行最大的那个
                    indices.append(indice)
            # indices = [(np.where(row == 1)[0]+2).tolist() for row in array_prob] # 加上padding项
                tokens = tokenizers[feature_key].batch_decode_2d(indices) # indices+2
            hard_samples[feature_key] = tokens

        return _batch, hard_samples

    def get_data_generator():
        while True:
            for indices, data in tqdm(train_dataset_loader, desc='epoch'): # disable=not accelerator.is_main_process
                yield indices, data

    data_generator = get_data_generator()  # 转为生成器

    def get_test_generator():
        while True:
            for indices, data in test_dataset_loader:
                yield indices, data

    test_generator = get_test_generator()

    # diffusion prepare
    _betas = utils.stable_diffusion_beta_schedule()
    _schedule = utils.Schedule(_betas) # forward add noise schedule
    logging.info(f'use {_schedule}')

    def cfg_nnet(x, timesteps, context):
        # classifier-free config
        _cond = nnet_ema(x, timesteps, context=context)
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet_ema(x, timesteps, context=_empty_context)
        return _cond #+ config['sample']['scale'] * (_cond - _uncond) # classifier free 要用unconditional的表征进行放缩。

    def train_step(indices, _batch, epoch=None):
        _metrics = dict()
        optimizer.zero_grad()
        # _z, _z_mask, _z_his = encode(_batch[0], _batch[1], batch[2]) # B, 3, D; B, 3, D
        # _zc, loss_c = cluster(_z) # B, 1, D, near cluster

        _z, _context, neigh_loss, _, _ = nnet.encode_everything(indices, **_batch) # B，3，D； B,9,D； 对比损失

        loss_mse, loss_round, T_loss = utils.LSimple(_z, nnet, _schedule,
                             context=_context,  **_batch)  # currently only support the extracted feature version
        _metrics['loss_mse'] = loss_mse.detach().mean() #accelerator.gather(loss.detach()).mean()
        _metrics['loss_c'] = neigh_loss.detach().mean() #accelerator.gather(neigh_loss.detach()).mean()
        _metrics['loss_rounding'] = loss_round.detach().mean() # batch
        _metrics['T_loss'] = T_loss.detach().mean() # batch
        total_loss = loss_mse.mean() + neigh_loss.mean() + loss_round.mean() + T_loss.mean() if epoch > config['warm'] else loss_mse.mean() + loss_round.mean() + T_loss.mean()
        total_loss.backward()
        # accelerator.backward(loss.mean() + neigh_loss.mean())

        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return _z, dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def dpm_solver_sample(_n_samples, _sample_steps, **kwargs):
        _z_init = torch.randn(_n_samples, *config['z_shape'], device=device) # 从正态分布中采样噪声。 # B,3,128
        z_init2 = kwargs['z_init']
        mask = kwargs['mask'].unsqueeze(dim=-1).repeat(1,1, config['z_shape'][1]) # B,3,128
        _z_init = torch.where(mask, _z_init, z_init2) # 从正态分布中采样噪声。 # B,3,128
        # 先加噪声，重新定义schedule
        noise_schedule = NoiseScheduleVP(schedule='discrete',
                                         betas=torch.tensor(_betas, device=device).float())  # 选一个加噪声的方式

        def model_fn(x, t_continuous):
            t = t_continuous * _schedule.N
            return cfg_nnet(x, t ,kwargs['context'])

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1.)  # embedding， 感觉这里需要rounding
        return decode(_z, nnet.get_logits)


    def eval_step(n_samples, sample_steps):
        """这里需要做case study"""
        # 逐步采样。这里用上了DPM加速
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=dpm_solver, '
                     f'mini_batch_size={config["sample"]["mini_batch_size"]}')

        def sample_fn(_n_samples):
            indices, batch = next(test_generator)

            assert indices.size(0) == _n_samples
            indices, sample_input = convert_input((indices, batch))
            _z, _context_emb, _, labels, hard_labels = nnet.encode_everything_test(indices, **batch)
            # _context_emb = encode(_contexts)
            return labels, hard_labels, dpm_solver_sample(_n_samples, sample_steps, context=_context_emb, z_init=_z, mask=sample_input['mask'])

        with tempfile.TemporaryDirectory() as temp_path:  # 创建临时目录，进行文件管理
            path = config['sample']['path'] + data_n + '/' + task or temp_path
            work_dir = config['workdir'] + data_n + '/' + task
            # if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
            os.makedirs(work_dir, exist_ok=True)
            utils.sample2dir(path, n_samples, config['sample']['mini_batch_size'], sample_fn, tokenizers,
                             dataset.unpreprocess) # 看看要不要改成test，因为这里是test

            _quality = 0
            # if accelerator.is_main_process:  # 采样完成后计算fid和原图的差异。metrics,可以用于可视化
            # _fid = None
            _quality, jaccard_quality = utils.calculate_quality(path)
            logging.info(f'step={train_state.step} quality-{n_samples}={_quality}')
            logging.info(f'step={train_state.step} jaccard_quality-{n_samples}={jaccard_quality}')
            with open(os.path.join(config['workdir'] + data_n + '/' + task, 'eval.log'), 'a') as f:
                print(f'step={train_state.step} quality-{n_samples}={_quality} jaccard_quality-{n_samples}={jaccard_quality}', file=f)
            wandb.log({f'quality_mse-{n_samples}': _quality}, step=train_state.step)
            _quality = torch.tensor(_quality, device=device)
            # _fid = accelerator.reduce(_fid, reduction='sum')

        return _quality.item(), jaccard_quality # 这里存储的是embeddings，一旦不进行decode就是mediffusion。

    # logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config["mixed_precision"]}')

    logging.info(f'Start fitting, step={train_state.step}')

    def convert_input(batch_data, mask_static=True):
        # padding到相同的长度便于转为tensor
        indices, batch = batch_data
        # print("batch_data 0", batch['mask'])

        feature_keys = ['conditions', 'drugs', 'procedures']
        # batch_new = {}
        for feature_key in feature_keys:
            # tokenizers
            batch[feature_key] = tokenizers[feature_key].batch_encode_2d(batch[feature_key]) # B,M,  m——feature没有用。
            batch[feature_key + '_hist'] = tokenizers[feature_key].batch_encode_3d(batch[feature_key + '_hist']) # B,V,M
            origin = batch[feature_key + '_comp'].copy()
            batch[feature_key + '_comp'] = tokenizers[feature_key].batch_encode_2d(batch[feature_key + '_comp']) # B,M
            # tensor
            batch[feature_key] = torch.tensor(batch[feature_key], dtype=torch.long, device=device)
            batch[feature_key + '_hist'] = torch.tensor(batch[feature_key + '_hist'], dtype=torch.long, device=device)
            batch[feature_key + '_comp'] = torch.tensor(batch[feature_key + '_comp'], dtype=torch.long, device=device)
            # mask
            batch[feature_key + '_mask'] = batch[feature_key] != 0 # 为True的就是mask掉的, # B,M
            batch[feature_key + '_hist_mask'] = batch[feature_key + '_hist'] != 0 # 这个是为了padding, # B,V,M
            batch[feature_key + '_comp_mask'] = batch[feature_key + '_comp'] != 0 # 这个是为了padding # B,M

            # others
            batch[feature_key + '_comps'] = tokenizers[feature_key].batch_encode_2d(origin, padding=False,  truncation=False) # 不需要减去2,只需要能看清楚是什么就行，因为decode的时候会加上2
            batch[feature_key + '_comps_origin'] = origin # origin ID

            # note special
            batch['has_note'] = False
            ## 20251210注释掉"if config_health['DATASET'] == 'MIV-Note':"
            # if config_health['DATASET'] == 'MIV-Note': # LOS MIV修改下。
            #     batch['note'] = torch.tensor(batch['note'], device=device) # cur, B,D; 这里不需要对他进行重建，因为他可能变为pad字符串
            #     batch['note' + '_hist'] = pad_list(batch['note' + '_hist'], device=device)# hist, B,T,D
            #     batch['note' + '_hist_mask'] = batch['note' + '_hist'].sum(dim=-1) !=0 # B,T
            #
            #     # 作为context,需要合并
            #     batch['note' + '_hist'] = torch.cat([batch['note' + '_hist'], batch['note'].unsqueeze(dim=1)], dim=1)  # B,T+1,D
            #     batch['note' + '_mask'] = batch['note'].sum(dim=-1) != 0  # B,
            #     batch['note' + '_hist_mask'] = torch.cat(
            #         [batch['note' + '_hist_mask'], batch['note' + '_mask'].unsqueeze(dim=1)], dim=1)  # B,T+1
            #
            #     batch['has_note'] = True

        # mask设置为固定
        if mask_static:
            batch['mask'] = torch.tensor(batch['mask'], dtype=torch.bool, device=device)
        # mask不固定, evaldffs固定就好, diverse
        else:
            m = 3 # 4 if batch['has_note'] else 3
            batch['mask'] = utils.generate_mask(batch['conditions'].shape[0],m=m).to(device=device)


        aligned_data = [batch[feature_key + '_hist_mask'].sum(dim=-1).unsqueeze(dim=-1) for feature_key in feature_keys] # B,V,1
        batch['mask_hist'] = torch.cat(aligned_data, dim=-1)  # B, V, 3，记录history mask状态

        return indices, batch


    ########## 加入训练集结果统计 #########
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=config['sample']['mini_batch_size'],  # 用mini_batch_size评估更省
        shuffle=False,  # 固定顺序更稳定可比
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn_dict
    )

    def get_train_eval_generator():
        while True:
            for indices, data in train_eval_loader:
                yield indices, data

    train_eval_generator = get_train_eval_generator()

    def train_epoch_metrics(sample_steps, num_batches=2, mask_static=True):
        """
        在 train split 上抽 num_batches 个 batch，计算平均 MSE 和 (Jc,Jp,Jd)
        """
        nnet.eval()
        mse_list = []
        jc_list, jp_list, jd_list = [], [], []

        with torch.no_grad():
            for _ in range(num_batches):
                idx, b = next(train_eval_generator)
                idx, b = convert_input((idx, b), mask_static=mask_static)

                # encode + GT
                z0, contexts, _, labels, hard_labels = nnet.encode_everything_test(idx, **b)

                # sample + decode
                samples, hard_samples = dpm_solver_sample(
                    _n_samples=idx.size(0),
                    _sample_steps=sample_steps,
                    context=contexts,
                    z_init=z0,
                    mask=b['mask']
                )

                # MSE（你中途输出那段也是这样算的）
                mse_list.append(F.mse_loss(samples, labels.squeeze()).item())

                # Jaccard（三视图）
                gt_cond = tokenizers['conditions'].batch_decode_2d(hard_labels['conditions'])
                gt_proc = tokenizers['procedures'].batch_decode_2d(hard_labels['procedures'])
                gt_drug = tokenizers['drugs'].batch_decode_2d(hard_labels['drugs'])

                jc_list.append(utils.calculate_average_jaccard(hard_samples['conditions'], gt_cond))
                jp_list.append(utils.calculate_average_jaccard(hard_samples['procedures'], gt_proc))
                jd_list.append(utils.calculate_average_jaccard(hard_samples['drugs'], gt_drug))

        nnet.train()

        train_mse = float(np.mean(mse_list))
        train_jacc = (float(np.mean(jc_list)), float(np.mean(jp_list)), float(np.mean(jd_list)))
        return train_mse, train_jacc

    # ############################ 1 #############################
    # def compute_secmi_metrics(member_scores, holdout_scores):
    #     """
    #     score 越小越 member，因此 attack_score = -score
    #     """
    #     member_scores = np.asarray(member_scores, dtype=np.float64)
    #     holdout_scores = np.asarray(holdout_scores, dtype=np.float64)
    #
    #     if len(member_scores) == 0 or len(holdout_scores) == 0:
    #         return {
    #             "AUC": float("nan"),
    #             "ASR": float("nan"),
    #             "TPR@1%FPR": float("nan"),
    #             "TPR@0.1%FPR": float("nan"),
    #             "member_mean": float("nan"),
    #             "holdout_mean": float("nan"),
    #             "member_std": float("nan"),
    #             "holdout_std": float("nan"),
    #         }
    #
    #     y_true = np.concatenate([
    #         np.ones_like(member_scores, dtype=np.int64),
    #         np.zeros_like(holdout_scores, dtype=np.int64)
    #     ])
    #     attack_score = -np.concatenate([member_scores, holdout_scores])
    #
    #     auc = roc_auc_score(y_true, attack_score)
    #     fpr, tpr, _ = roc_curve(y_true, attack_score)
    #
    #     bal_acc = 0.5 * (tpr + (1.0 - fpr))
    #     asr = float(np.max(bal_acc))
    #
    #     def tpr_at_fpr(target_fpr):
    #         mask = fpr <= target_fpr
    #         return float(np.max(tpr[mask])) if np.any(mask) else 0.0
    #
    #     return {
    #         "AUC": float(auc),
    #         "ASR": float(asr),
    #         "TPR@1%FPR": tpr_at_fpr(0.01),
    #         "TPR@0.1%FPR": tpr_at_fpr(0.001),
    #         "member_mean": float(np.mean(member_scores)),
    #         "holdout_mean": float(np.mean(holdout_scores)),
    #         "member_std": float(np.std(member_scores)),
    #         "holdout_std": float(np.std(holdout_scores)),
    #     }
    #
    # def build_alpha_bar_from_betas(betas, device):
    #     betas = torch.tensor(betas, dtype=torch.float32, device=device)
    #     alphas = 1.0 - betas
    #     alpha_bar = torch.cumprod(alphas, dim=0)
    #     return alpha_bar
    #
    # alpha_bar = build_alpha_bar_from_betas(_betas, device)
    #
    # @torch.no_grad()
    # def predict_eps(x_t, t_idx, context):
    #     """
    #     尽量兼容你的 nnet_ema 时间步格式：
    #     先试 long，再试 float。
    #     """
    #     B = x_t.size(0)
    #     t_long = torch.full((B,), int(t_idx), device=x_t.device, dtype=torch.long)
    #     t_float = t_long.float()
    #
    #     try:
    #         return nnet_ema(x_t, t_long, context=context)
    #     except Exception:
    #         return nnet_ema(x_t, t_float, context=context)
    #
    # def predict_x0_from_eps(x_t, eps, alpha_bar_t):
    #     return (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t.clamp_min(1e-12))
    #
    # @torch.no_grad()
    # def ddim_deterministic_step(x_t, t_idx, s_idx, context, alpha_bar):
    #     """
    #     用 deterministic DDIM 形式从 t -> s
    #     s > t 表示更往后（更噪）
    #     s < t 表示往前（去噪）
    #     """
    #     assert 0 <= t_idx < len(alpha_bar), f"t_idx={t_idx}, len(alpha_bar)={len(alpha_bar)}"
    #     assert 0 <= s_idx < len(alpha_bar), f"s_idx={s_idx}, len(alpha_bar)={len(alpha_bar)}"
    #
    #     ab_t = alpha_bar[t_idx].view(1, 1, 1).to(x_t.device)
    #     ab_s = alpha_bar[s_idx].view(1, 1, 1).to(x_t.device)
    #
    #     eps = predict_eps(x_t, t_idx, context)
    #     x0_hat = predict_x0_from_eps(x_t, eps, ab_t)
    #     x_s = torch.sqrt(ab_s) * x0_hat + torch.sqrt(1.0 - ab_s) * eps
    #     return x_s
    #
    # @torch.no_grad()
    # def secmi_t_error_single_t(z0, context, alpha_bar, t_sec=100, stride=10):
    #     """
    #     SecMI 风格的 local-cycle score:
    #     x0 -> x_tsec -> x_tsec+stride -> x_tsec
    #     score 越小，越像 member
    #     """
    #     T = len(alpha_bar) - 1
    #     t_sec = int(min(t_sec, T - stride - 1))
    #     t_sec = max(t_sec, 1)
    #
    #     x = z0
    #     cur = 0
    #     while cur < t_sec:
    #         nxt = min(cur + stride, t_sec)
    #         x = ddim_deterministic_step(x, cur, nxt, context, alpha_bar)
    #         cur = nxt
    #
    #     x_t = x
    #     x_t_plus = ddim_deterministic_step(x_t, t_sec, t_sec + stride, context, alpha_bar)
    #     x_back = ddim_deterministic_step(x_t_plus, t_sec + stride, t_sec, context, alpha_bar)
    #
    #     score = ((x_back - x_t) ** 2).flatten(1).mean(dim=1)
    #     return score
    #
    # @torch.no_grad()
    # def collect_secmi_bank(data_loader, max_batches=None):
    #     """
    #     先把 z0 和 context 全收集起来，后面扫不同 t_sec 时不用重复 encode。
    #     存 CPU，节省显存。
    #     """
    #     nnet.eval()
    #     nnet_ema.eval()
    #
    #     z_list = []
    #     c_list = []
    #
    #     for batch_idx, (indices, batch) in enumerate(data_loader):
    #         if max_batches is not None and batch_idx >= max_batches:
    #             break
    #
    #         indices, batch = convert_input((indices, batch), mask_static=True)
    #         z0, contexts, _, _, _ = nnet.encode_everything_test(indices, **batch)
    #
    #         z_list.append(z0.detach().cpu())
    #         c_list.append(contexts.detach().cpu())
    #
    #     if len(z_list) == 0:
    #         return None, None
    #
    #     z_bank = torch.cat(z_list, dim=0)
    #     c_bank = torch.cat(c_list, dim=0)
    #     return z_bank, c_bank
    #
    # @torch.no_grad()
    # def score_secmi_bank(z_bank_cpu, c_bank_cpu, alpha_bar, t_sec=100, stride=10, score_batch_size=256):
    #     """
    #     对已经收集好的 bank 计算某个 t_sec 的全部 score
    #     """
    #     if z_bank_cpu is None or c_bank_cpu is None:
    #         return np.array([])
    #
    #     scores = []
    #     total_n = z_bank_cpu.size(0)
    #
    #     for start in range(0, total_n, score_batch_size):
    #         end = min(start + score_batch_size, total_n)
    #         z0 = z_bank_cpu[start:end].to(device, non_blocking=True)
    #         context = c_bank_cpu[start:end].to(device, non_blocking=True)
    #
    #         batch_scores = secmi_t_error_single_t(
    #             z0=z0,
    #             context=context,
    #             alpha_bar=alpha_bar,
    #             t_sec=t_sec,
    #             stride=stride
    #         )
    #         scores.append(batch_scores.detach().cpu())
    #
    #     if len(scores) == 0:
    #         return np.array([])
    #
    #     return torch.cat(scores, dim=0).numpy()
    #
    # def save_secmi_scan_results(results, save_path):
    #     if results is None or len(results) == 0:
    #         return
    #
    #     fieldnames = [
    #         "t_sec", "stride", "num_member", "num_holdout",
    #         "AUC", "ASR", "TPR@1%FPR", "TPR@0.1%FPR",
    #         "member_mean", "holdout_mean", "member_std", "holdout_std"
    #     ]
    #
    #     with open(save_path, "w", newline="", encoding="utf-8") as f:
    #         writer = csv.DictWriter(f, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for row in results:
    #             writer.writerow(row)
    #
    # def run_secmi_privacy_eval(
    #         member_loader,
    #         holdout_loader,
    #         t_sec=100,
    #         stride=10,
    #         max_member_batches=None,
    #         max_holdout_batches=None,
    #         score_batch_size=256,
    # ):
    #     print("[SecMI] collecting member bank...")
    #     member_z_bank, member_c_bank = collect_secmi_bank(
    #         member_loader, max_batches=max_member_batches
    #     )
    #
    #     print("[SecMI] collecting holdout bank...")
    #     holdout_z_bank, holdout_c_bank = collect_secmi_bank(
    #         holdout_loader, max_batches=max_holdout_batches
    #     )
    #
    #     member_scores = score_secmi_bank(
    #         member_z_bank, member_c_bank,
    #         alpha_bar=alpha_bar,
    #         t_sec=t_sec,
    #         stride=stride,
    #         score_batch_size=score_batch_size
    #     )
    #
    #     holdout_scores = score_secmi_bank(
    #         holdout_z_bank, holdout_c_bank,
    #         alpha_bar=alpha_bar,
    #         t_sec=t_sec,
    #         stride=stride,
    #         score_batch_size=score_batch_size
    #     )
    #
    #     metrics = compute_secmi_metrics(member_scores, holdout_scores)
    #
    #     print("[SecMI Privacy Eval]")
    #     print(f"  t_sec      = {t_sec}")
    #     print(f"  stride     = {stride}")
    #     print(f"  #member    = {len(member_scores)}")
    #     print(f"  #holdout   = {len(holdout_scores)}")
    #     print(f"  AUC        = {metrics['AUC']:.4f}")
    #     print(f"  ASR        = {metrics['ASR']:.4f}")
    #     print(f"  TPR@1%     = {metrics['TPR@1%FPR']:.4f}")
    #     print(f"  TPR@0.1%   = {metrics['TPR@0.1%FPR']:.4f}")
    #     print(f"  member_mu  = {metrics['member_mean']:.6f}")
    #     print(f"  holdout_mu = {metrics['holdout_mean']:.6f}")
    #
    #     return metrics, member_scores, holdout_scores
    #
    # def run_secmi_tsec_scan(
    #         member_loader,
    #         holdout_loader,
    #         t_sec_list=(25, 50, 75, 100, 125, 150, 175, 200),
    #         stride=10,
    #         max_member_batches=None,
    #         max_holdout_batches=None,
    #         score_batch_size=256,
    # ):
    #     """
    #     一次性扫描多个 t_sec，输出每个 t_sec 的攻击结果
    #     """
    #     print("[SecMI Scan] collecting member bank once...")
    #     member_z_bank, member_c_bank = collect_secmi_bank(
    #         member_loader, max_batches=max_member_batches
    #     )
    #
    #     print("[SecMI Scan] collecting holdout bank once...")
    #     holdout_z_bank, holdout_c_bank = collect_secmi_bank(
    #         holdout_loader, max_batches=max_holdout_batches
    #     )
    #
    #     results = []
    #
    #     for t_sec in t_sec_list:
    #         member_scores = score_secmi_bank(
    #             member_z_bank, member_c_bank,
    #             alpha_bar=alpha_bar,
    #             t_sec=t_sec,
    #             stride=stride,
    #             score_batch_size=score_batch_size
    #         )
    #
    #         holdout_scores = score_secmi_bank(
    #             holdout_z_bank, holdout_c_bank,
    #             alpha_bar=alpha_bar,
    #             t_sec=t_sec,
    #             stride=stride,
    #             score_batch_size=score_batch_size
    #         )
    #
    #         metrics = compute_secmi_metrics(member_scores, holdout_scores)
    #
    #         row = {
    #             "t_sec": int(t_sec),
    #             "stride": int(stride),
    #             "num_member": int(len(member_scores)),
    #             "num_holdout": int(len(holdout_scores)),
    #             "AUC": float(metrics["AUC"]),
    #             "ASR": float(metrics["ASR"]),
    #             "TPR@1%FPR": float(metrics["TPR@1%FPR"]),
    #             "TPR@0.1%FPR": float(metrics["TPR@0.1%FPR"]),
    #             "member_mean": float(metrics["member_mean"]),
    #             "holdout_mean": float(metrics["holdout_mean"]),
    #             "member_std": float(metrics["member_std"]),
    #             "holdout_std": float(metrics["holdout_std"]),
    #         }
    #         results.append(row)
    #
    #         print(
    #             f"[SecMI Scan] t_sec={t_sec:>3d} | "
    #             f"AUC={row['AUC']:.4f} | "
    #             f"ASR={row['ASR']:.4f} | "
    #             f"TPR@1%={row['TPR@1%FPR']:.4f} | "
    #             f"TPR@0.1%={row['TPR@0.1%FPR']:.4f} | "
    #             f"member_mu={row['member_mean']:.6f} | "
    #             f"holdout_mu={row['holdout_mean']:.6f}"
    #         )
    #
    #     if len(results) > 0:
    #         best_auc_idx = int(np.argmax([r["AUC"] for r in results]))
    #         best_asr_idx = int(np.argmax([r["ASR"] for r in results]))
    #
    #         print("[SecMI Scan] best by AUC:")
    #         print(results[best_auc_idx])
    #
    #         print("[SecMI Scan] best by ASR:")
    #         print(results[best_asr_idx])
    #
    #     return results
    # # SecMI 使用
    # alpha_bar = build_alpha_bar_from_betas(_betas, device)
    # ########################### 1 #############################

    # ########## 2
    # def compute_mia_metrics(member_scores, holdout_scores):
    #     """
    #     member_scores / holdout_scores: numpy array
    #     这里 score 是 error，越小越像 member，所以最终 attack_score = -error
    #     """
    #     member_scores = np.asarray(member_scores, dtype=np.float64)
    #     holdout_scores = np.asarray(holdout_scores, dtype=np.float64)
    #
    #     if len(member_scores) == 0 or len(holdout_scores) == 0:
    #         return {
    #             "AUC": float("nan"),
    #             "ASR": float("nan"),
    #             "TPR@1%FPR": float("nan"),
    #             "TPR@0.1%FPR": float("nan"),
    #             "member_mean": float("nan"),
    #             "holdout_mean": float("nan"),
    #             "member_std": float("nan"),
    #             "holdout_std": float("nan"),
    #         }
    #
    #     y_true = np.concatenate([
    #         np.ones_like(member_scores, dtype=np.int64),
    #         np.zeros_like(holdout_scores, dtype=np.int64)
    #     ])
    #     attack_score = -np.concatenate([member_scores, holdout_scores])
    #
    #     auc = roc_auc_score(y_true, attack_score)
    #     fpr, tpr, _ = roc_curve(y_true, attack_score)
    #
    #     bal_acc = 0.5 * (tpr + (1.0 - fpr))
    #     asr = float(np.max(bal_acc))
    #
    #     def tpr_at_fpr(target_fpr):
    #         mask = fpr <= target_fpr
    #         return float(np.max(tpr[mask])) if np.any(mask) else 0.0
    #
    #     return {
    #         "AUC": float(auc),
    #         "ASR": float(asr),
    #         "TPR@1%FPR": tpr_at_fpr(0.01),
    #         "TPR@0.1%FPR": tpr_at_fpr(0.001),
    #         "member_mean": float(np.mean(member_scores)),
    #         "holdout_mean": float(np.mean(holdout_scores)),
    #         "member_std": float(np.std(member_scores)),
    #         "holdout_std": float(np.std(holdout_scores)),
    #     }
    #
    # def build_alpha_bar_from_betas(betas, device):
    #     betas = torch.tensor(betas, dtype=torch.float32, device=device)
    #     alphas = 1.0 - betas
    #     alpha_bar = torch.cumprod(alphas, dim=0)  # [T]
    #     return alpha_bar
    #
    # alpha_bar = build_alpha_bar_from_betas(_betas, device)
    #
    # @torch.no_grad()
    # def predict_eps(x_t, t_input, context):
    #     """
    #     尽量兼容你的 nnet_ema 时间步输入格式。
    #     如果你的 nnet_ema 只接受 float 时间步，就保留 float 分支即可。
    #     """
    #     if not torch.is_tensor(t_input):
    #         t_input = torch.full((x_t.size(0),), int(t_input), device=x_t.device, dtype=torch.long)
    #     else:
    #         t_input = t_input.to(x_t.device)
    #
    #     try:
    #         return nnet_ema(x_t, t_input.long(), context=context)
    #     except Exception:
    #         try:
    #             return nnet_ema(x_t, t_input.float(), context=context)
    #         except Exception:
    #             # 如果你的模型要求归一化到 [0, 1]，打开下面这一行并替换上一行
    #             # return nnet_ema(x_t, t_input.float() / float(_schedule.N), context=context)
    #             raise
    #
    # @torch.no_grad()
    # def q_sample_from_x0(x0, t_idx, alpha_bar, eps=None):
    #     """
    #     x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
    #     x0: [B, 3, D] or [B, C, D]
    #     t_idx: int or tensor[B]
    #     """
    #     B = x0.size(0)
    #
    #     if not torch.is_tensor(t_idx):
    #         t_idx = torch.full((B,), int(t_idx), device=x0.device, dtype=torch.long)
    #     else:
    #         t_idx = t_idx.to(x0.device).long()
    #
    #     if eps is None:
    #         eps = torch.randn_like(x0)
    #
    #     ab_t = alpha_bar.index_select(0, t_idx).view(B, 1, 1)
    #     x_t = torch.sqrt(ab_t) * x0 + torch.sqrt(1.0 - ab_t) * eps
    #     return x_t, eps
    #
    # @torch.no_grad()
    # def noise_pred_error_score_fixed_t(z0, context, alpha_bar, t_sec=100, n_noise_repeat=4):
    #     """
    #     固定某个 t_sec，重复采样多次噪声，取平均误差作为 patient-level score
    #     score 越小越像 member
    #     """
    #     B = z0.size(0)
    #     t_vec = torch.full((B,), int(t_sec), device=z0.device, dtype=torch.long)
    #
    #     score_list = []
    #     for _ in range(n_noise_repeat):
    #         x_t, eps = q_sample_from_x0(z0, t_vec, alpha_bar, eps=None)
    #         eps_hat = predict_eps(x_t, t_vec, context)
    #         err = ((eps_hat - eps) ** 2).flatten(1).mean(dim=1)   # [B]
    #         score_list.append(err)
    #
    #     scores = torch.stack(score_list, dim=1).mean(dim=1)  # [B]
    #     return scores
    #
    # @torch.no_grad()
    # def noise_pred_error_score_multi_t(
    #     z0,
    #     context,
    #     alpha_bar,
    #     t_candidates=(25, 50, 75, 100, 125, 150, 175, 200),
    #     n_noise_repeat=2
    # ):
    #     """
    #     对多个 t 做平均，通常比单个 t 更稳
    #     最终返回每个 patient 一个 score
    #     """
    #     all_t_scores = []
    #     for t_sec in t_candidates:
    #         scores_t = noise_pred_error_score_fixed_t(
    #             z0=z0,
    #             context=context,
    #             alpha_bar=alpha_bar,
    #             t_sec=t_sec,
    #             n_noise_repeat=n_noise_repeat
    #         )
    #         all_t_scores.append(scores_t)
    #
    #     scores = torch.stack(all_t_scores, dim=1).mean(dim=1)  # [B]
    #     return scores
    #
    # @torch.no_grad()
    # def collect_ehr_mia_bank(data_loader, max_batches=None):
    #     """
    #     先把 z0 和 context 收集起来，后面不同 t_sec 复用，避免重复 encode
    #     """
    #     nnet.eval()
    #     nnet_ema.eval()
    #
    #     z_list = []
    #     c_list = []
    #
    #     for batch_idx, (indices, batch) in enumerate(data_loader):
    #         if max_batches is not None and batch_idx >= max_batches:
    #             break
    #
    #         indices, batch = convert_input((indices, batch), mask_static=True)
    #         z0, contexts, _, _, _ = nnet.encode_everything_test(indices, **batch)
    #
    #         z_list.append(z0.detach().cpu())
    #         c_list.append(contexts.detach().cpu())
    #
    #     if len(z_list) == 0:
    #         return None, None
    #
    #     z_bank = torch.cat(z_list, dim=0)
    #     c_bank = torch.cat(c_list, dim=0)
    #     return z_bank, c_bank
    #
    # @torch.no_grad()
    # def score_ehr_mia_bank_fixed_t(
    #     z_bank_cpu,
    #     c_bank_cpu,
    #     alpha_bar,
    #     t_sec=100,
    #     n_noise_repeat=4,
    #     score_batch_size=256
    # ):
    #     """
    #     对缓存好的 bank，在固定 t_sec 上计算全部 score
    #     """
    #     if z_bank_cpu is None or c_bank_cpu is None:
    #         return np.array([])
    #
    #     scores = []
    #     total_n = z_bank_cpu.size(0)
    #
    #     for start in range(0, total_n, score_batch_size):
    #         end = min(start + score_batch_size, total_n)
    #
    #         z0 = z_bank_cpu[start:end].to(device, non_blocking=True)
    #         context = c_bank_cpu[start:end].to(device, non_blocking=True)
    #
    #         batch_scores = noise_pred_error_score_fixed_t(
    #             z0=z0,
    #             context=context,
    #             alpha_bar=alpha_bar,
    #             t_sec=t_sec,
    #             n_noise_repeat=n_noise_repeat
    #         )
    #         scores.append(batch_scores.detach().cpu())
    #
    #     if len(scores) == 0:
    #         return np.array([])
    #
    #     return torch.cat(scores, dim=0).numpy()
    #
    # @torch.no_grad()
    # def score_ehr_mia_bank_multi_t(
    #     z_bank_cpu,
    #     c_bank_cpu,
    #     alpha_bar,
    #     t_candidates=(25, 50, 75, 100, 125, 150, 175, 200),
    #     n_noise_repeat=2,
    #     score_batch_size=256
    # ):
    #     """
    #     对缓存好的 bank，在多个 t 上求平均 score
    #     """
    #     if z_bank_cpu is None or c_bank_cpu is None:
    #         return np.array([])
    #
    #     scores = []
    #     total_n = z_bank_cpu.size(0)
    #
    #     for start in range(0, total_n, score_batch_size):
    #         end = min(start + score_batch_size, total_n)
    #
    #         z0 = z_bank_cpu[start:end].to(device, non_blocking=True)
    #         context = c_bank_cpu[start:end].to(device, non_blocking=True)
    #
    #         batch_scores = noise_pred_error_score_multi_t(
    #             z0=z0,
    #             context=context,
    #             alpha_bar=alpha_bar,
    #             t_candidates=t_candidates,
    #             n_noise_repeat=n_noise_repeat
    #         )
    #         scores.append(batch_scores.detach().cpu())
    #
    #     if len(scores) == 0:
    #         return np.array([])
    #
    #     return torch.cat(scores, dim=0).numpy()
    #
    # def save_ehr_mia_scan_results(results, save_path):
    #     if results is None or len(results) == 0:
    #         return
    #
    #     fieldnames = [
    #         "t_sec", "num_member", "num_holdout",
    #         "AUC", "ASR", "TPR@1%FPR", "TPR@0.1%FPR",
    #         "member_mean", "holdout_mean", "member_std", "holdout_std"
    #     ]
    #
    #     with open(save_path, "w", newline="", encoding="utf-8") as f:
    #         writer = csv.DictWriter(f, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for row in results:
    #             writer.writerow(row)
    #
    # def run_ehr_noise_mia_eval_fixed_t(
    #     member_loader,
    #     holdout_loader,
    #     t_sec=100,
    #     n_noise_repeat=4,
    #     max_member_batches=None,
    #     max_holdout_batches=None,
    #     score_batch_size=256,
    # ):
    #     print("[EHR-MIA] collecting member bank...")
    #     member_z_bank, member_c_bank = collect_ehr_mia_bank(
    #         member_loader, max_batches=max_member_batches
    #     )
    #
    #     print("[EHR-MIA] collecting holdout bank...")
    #     holdout_z_bank, holdout_c_bank = collect_ehr_mia_bank(
    #         holdout_loader, max_batches=max_holdout_batches
    #     )
    #
    #     member_scores = score_ehr_mia_bank_fixed_t(
    #         member_z_bank,
    #         member_c_bank,
    #         alpha_bar=alpha_bar,
    #         t_sec=t_sec,
    #         n_noise_repeat=n_noise_repeat,
    #         score_batch_size=score_batch_size
    #     )
    #
    #     holdout_scores = score_ehr_mia_bank_fixed_t(
    #         holdout_z_bank,
    #         holdout_c_bank,
    #         alpha_bar=alpha_bar,
    #         t_sec=t_sec,
    #         n_noise_repeat=n_noise_repeat,
    #         score_batch_size=score_batch_size
    #     )
    #
    #     metrics = compute_mia_metrics(member_scores, holdout_scores)
    #
    #     print("[EHR Noise-MIA Fixed-t]")
    #     print(f"  t_sec      = {t_sec}")
    #     print(f"  repeat     = {n_noise_repeat}")
    #     print(f"  #member    = {len(member_scores)}")
    #     print(f"  #holdout   = {len(holdout_scores)}")
    #     print(f"  AUC        = {metrics['AUC']:.4f}")
    #     print(f"  ASR        = {metrics['ASR']:.4f}")
    #     print(f"  TPR@1%     = {metrics['TPR@1%FPR']:.4f}")
    #     print(f"  TPR@0.1%   = {metrics['TPR@0.1%FPR']:.4f}")
    #     print(f"  member_mu  = {metrics['member_mean']:.8e}")
    #     print(f"  holdout_mu = {metrics['holdout_mean']:.8e}")
    #
    #     return metrics, member_scores, holdout_scores
    #
    # def run_ehr_noise_mia_scan(
    #     member_loader,
    #     holdout_loader,
    #     t_sec_list=(25, 50, 75, 100, 125, 150, 175, 200),
    #     n_noise_repeat=4,
    #     max_member_batches=None,
    #     max_holdout_batches=None,
    #     score_batch_size=256,
    # ):
    #     """
    #     扫多个 t_sec，看哪个时间步最容易暴露 membership signal
    #     """
    #     print("[EHR-MIA Scan] collecting member bank once...")
    #     member_z_bank, member_c_bank = collect_ehr_mia_bank(
    #         member_loader, max_batches=max_member_batches
    #     )
    #
    #     print("[EHR-MIA Scan] collecting holdout bank once...")
    #     holdout_z_bank, holdout_c_bank = collect_ehr_mia_bank(
    #         holdout_loader, max_batches=max_holdout_batches
    #     )
    #
    #     results = []
    #
    #     for t_sec in t_sec_list:
    #         member_scores = score_ehr_mia_bank_fixed_t(
    #             member_z_bank,
    #             member_c_bank,
    #             alpha_bar=alpha_bar,
    #             t_sec=t_sec,
    #             n_noise_repeat=n_noise_repeat,
    #             score_batch_size=score_batch_size
    #         )
    #
    #         holdout_scores = score_ehr_mia_bank_fixed_t(
    #             holdout_z_bank,
    #             holdout_c_bank,
    #             alpha_bar=alpha_bar,
    #             t_sec=t_sec,
    #             n_noise_repeat=n_noise_repeat,
    #             score_batch_size=score_batch_size
    #         )
    #
    #         metrics = compute_mia_metrics(member_scores, holdout_scores)
    #
    #         row = {
    #             "t_sec": int(t_sec),
    #             "num_member": int(len(member_scores)),
    #             "num_holdout": int(len(holdout_scores)),
    #             "AUC": float(metrics["AUC"]),
    #             "ASR": float(metrics["ASR"]),
    #             "TPR@1%FPR": float(metrics["TPR@1%FPR"]),
    #             "TPR@0.1%FPR": float(metrics["TPR@0.1%FPR"]),
    #             "member_mean": float(metrics["member_mean"]),
    #             "holdout_mean": float(metrics["holdout_mean"]),
    #             "member_std": float(metrics["member_std"]),
    #             "holdout_std": float(metrics["holdout_std"]),
    #         }
    #         results.append(row)
    #
    #         print(
    #             f"[EHR-MIA Scan] t_sec={t_sec:>3d} | "
    #             f"AUC={row['AUC']:.4f} | "
    #             f"ASR={row['ASR']:.4f} | "
    #             f"TPR@1%={row['TPR@1%FPR']:.4f} | "
    #             f"TPR@0.1%={row['TPR@0.1%FPR']:.4f} | "
    #             f"member_mu={row['member_mean']:.8e} | "
    #             f"holdout_mu={row['holdout_mean']:.8e}"
    #         )
    #
    #     if len(results) > 0:
    #         best_auc_row = max(results, key=lambda x: x["AUC"])
    #         best_asr_row = max(results, key=lambda x: x["ASR"])
    #
    #         print("[EHR-MIA Scan] best by AUC:")
    #         print(best_auc_row)
    #
    #         print("[EHR-MIA Scan] best by ASR:")
    #         print(best_asr_row)
    #
    #     return results
    #
    # def run_ehr_noise_mia_eval_multi_t(
    #     member_loader,
    #     holdout_loader,
    #     t_candidates=(25, 50, 75, 100, 125, 150, 175, 200),
    #     n_noise_repeat=2,
    #     max_member_batches=None,
    #     max_holdout_batches=None,
    #     score_batch_size=256,
    # ):
    #     """
    #     对多个 t 的 noise-pred error 取平均，得到更稳的 patient-level score
    #     """
    #     print("[EHR-MIA Multi-t] collecting member bank...")
    #     member_z_bank, member_c_bank = collect_ehr_mia_bank(
    #         member_loader, max_batches=max_member_batches
    #     )
    #
    #     print("[EHR-MIA Multi-t] collecting holdout bank...")
    #     holdout_z_bank, holdout_c_bank = collect_ehr_mia_bank(
    #         holdout_loader, max_batches=max_holdout_batches
    #     )
    #
    #     member_scores = score_ehr_mia_bank_multi_t(
    #         member_z_bank,
    #         member_c_bank,
    #         alpha_bar=alpha_bar,
    #         t_candidates=t_candidates,
    #         n_noise_repeat=n_noise_repeat,
    #         score_batch_size=score_batch_size
    #     )
    #
    #     holdout_scores = score_ehr_mia_bank_multi_t(
    #         holdout_z_bank,
    #         holdout_c_bank,
    #         alpha_bar=alpha_bar,
    #         t_candidates=t_candidates,
    #         n_noise_repeat=n_noise_repeat,
    #         score_batch_size=score_batch_size
    #     )
    #
    #     metrics = compute_mia_metrics(member_scores, holdout_scores)
    #
    #     print("[EHR Noise-MIA Multi-t]")
    #     print(f"  t_candidates = {list(t_candidates)}")
    #     print(f"  repeat       = {n_noise_repeat}")
    #     print(f"  #member      = {len(member_scores)}")
    #     print(f"  #holdout     = {len(holdout_scores)}")
    #     print(f"  AUC          = {metrics['AUC']:.4f}")
    #     print(f"  ASR          = {metrics['ASR']:.4f}")
    #     print(f"  TPR@1%       = {metrics['TPR@1%FPR']:.4f}")
    #     print(f"  TPR@0.1%     = {metrics['TPR@0.1%FPR']:.4f}")
    #     print(f"  member_mu    = {metrics['member_mean']:.8e}")
    #     print(f"  holdout_mu   = {metrics['holdout_mean']:.8e}")
    #
    #     return metrics, member_scores, holdout_scores
    # ######################### 2 #############################

    ######################### 3 #############################
    def compute_mia_metrics(member_scores, holdout_scores):
        """
        member_scores / holdout_scores: numpy array
        这里 score 是 error，越小越像 member，所以最终 attack_score = -error
        """
        member_scores = np.asarray(member_scores, dtype=np.float64)
        holdout_scores = np.asarray(holdout_scores, dtype=np.float64)

        if len(member_scores) == 0 or len(holdout_scores) == 0:
            return {
                "AUC": float("nan"),
                "ASR": float("nan"),
                "TPR@1%FPR": float("nan"),
                "TPR@0.1%FPR": float("nan"),
                "member_mean": float("nan"),
                "holdout_mean": float("nan"),
                "member_std": float("nan"),
                "holdout_std": float("nan"),
            }

        y_true = np.concatenate([
            np.ones_like(member_scores, dtype=np.int64),
            np.zeros_like(holdout_scores, dtype=np.int64)
        ])
        attack_score = -np.concatenate([member_scores, holdout_scores])

        auc = roc_auc_score(y_true, attack_score)
        fpr, tpr, _ = roc_curve(y_true, attack_score)

        bal_acc = 0.5 * (tpr + (1.0 - fpr))
        asr = float(np.max(bal_acc))

        def tpr_at_fpr(target_fpr):
            mask = fpr <= target_fpr
            return float(np.max(tpr[mask])) if np.any(mask) else 0.0

        return {
            "AUC": float(auc),
            "ASR": float(asr),
            "TPR@1%FPR": tpr_at_fpr(0.01),
            "TPR@0.1%FPR": tpr_at_fpr(0.001),
            "member_mean": float(np.mean(member_scores)),
            "holdout_mean": float(np.mean(holdout_scores)),
            "member_std": float(np.std(member_scores)),
            "holdout_std": float(np.std(holdout_scores)),
        }

    def build_alpha_bar_from_betas(betas, device):
        betas = torch.tensor(betas, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)  # [T]
        return alpha_bar

    alpha_bar = build_alpha_bar_from_betas(_betas, device)

    @torch.no_grad()
    def predict_x0(x_t, t_input, context):
        """
        这版默认 nnet_ema 输出的是 x0_hat
        尽量兼容 long / float 时间步输入
        """
        if not torch.is_tensor(t_input):
            t_input = torch.full((x_t.size(0),), int(t_input), device=x_t.device, dtype=torch.long)
        else:
            t_input = t_input.to(x_t.device)

        try:
            return nnet_ema(x_t, t_input.long(), context=context)
        except Exception:
            try:
                return nnet_ema(x_t, t_input.float(), context=context)
            except Exception:
                # 如果你的模型要求归一化到 [0, 1]，把下面这一行打开
                # return nnet_ema(x_t, t_input.float() / float(_schedule.N), context=context)
                raise

    @torch.no_grad()
    def q_sample_from_x0(x0, t_idx, alpha_bar, eps=None):
        """
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        x0: [B, 3, D] or [B, C, D]
        t_idx: int or tensor[B]
        """
        B = x0.size(0)

        if not torch.is_tensor(t_idx):
            t_idx = torch.full((B,), int(t_idx), device=x0.device, dtype=torch.long)
        else:
            t_idx = t_idx.to(x0.device).long()

        if eps is None:
            eps = torch.randn_like(x0)

        ab_t = alpha_bar.index_select(0, t_idx).view(B, 1, 1)
        x_t = torch.sqrt(ab_t) * x0 + torch.sqrt(1.0 - ab_t) * eps
        return x_t, eps

    @torch.no_grad()
    def x0_recon_error_score_fixed_t(z0, context, alpha_bar, t_sec=100, n_noise_repeat=4):
        """
        固定某个 t_sec，重复采样多次噪声，取平均 x0 重建误差作为 patient-level score
        score 越小越像 member
        """
        B = z0.size(0)
        t_vec = torch.full((B,), int(t_sec), device=z0.device, dtype=torch.long)

        score_list = []
        for _ in range(n_noise_repeat):
            x_t, _ = q_sample_from_x0(z0, t_vec, alpha_bar, eps=None)
            x0_hat = predict_x0(x_t, t_vec, context)
            err = ((x0_hat - z0) ** 2).flatten(1).mean(dim=1)   # [B]
            score_list.append(err)

        scores = torch.stack(score_list, dim=1).mean(dim=1)  # [B]
        return scores

    @torch.no_grad()
    def x0_recon_error_score_multi_t(
        z0,
        context,
        alpha_bar,
        t_candidates=(25, 50, 75, 100, 125, 150, 175, 200),
        n_noise_repeat=2
    ):
        """
        对多个 t 做平均，通常比单个 t 更稳
        最终返回每个 patient 一个 score
        """
        all_t_scores = []
        for t_sec in t_candidates:
            scores_t = x0_recon_error_score_fixed_t(
                z0=z0,
                context=context,
                alpha_bar=alpha_bar,
                t_sec=t_sec,
                n_noise_repeat=n_noise_repeat
            )
            all_t_scores.append(scores_t)

        scores = torch.stack(all_t_scores, dim=1).mean(dim=1)  # [B]
        return scores

    @torch.no_grad()
    def collect_ehr_mia_bank(data_loader, max_batches=None):
        """
        先把 z0 和 context 收集起来，后面不同 t_sec 复用，避免重复 encode
        """
        nnet.eval()
        nnet_ema.eval()

        z_list = []
        c_list = []

        for batch_idx, (indices, batch) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            indices, batch = convert_input((indices, batch), mask_static=True)
            z0, contexts, _, _, _ = nnet.encode_everything_test(indices, **batch)

            z_list.append(z0.detach().cpu())
            c_list.append(contexts.detach().cpu())

        if len(z_list) == 0:
            return None, None

        z_bank = torch.cat(z_list, dim=0)
        c_bank = torch.cat(c_list, dim=0)
        return z_bank, c_bank

    @torch.no_grad()
    def score_ehr_mia_bank_fixed_t(
        z_bank_cpu,
        c_bank_cpu,
        alpha_bar,
        t_sec=100,
        n_noise_repeat=4,
        score_batch_size=256
    ):
        """
        对缓存好的 bank，在固定 t_sec 上计算全部 x0-recon score
        """
        if z_bank_cpu is None or c_bank_cpu is None:
            return np.array([])

        scores = []
        total_n = z_bank_cpu.size(0)

        for start in range(0, total_n, score_batch_size):
            end = min(start + score_batch_size, total_n)

            z0 = z_bank_cpu[start:end].to(device, non_blocking=True)
            context = c_bank_cpu[start:end].to(device, non_blocking=True)

            batch_scores = x0_recon_error_score_fixed_t(
                z0=z0,
                context=context,
                alpha_bar=alpha_bar,
                t_sec=t_sec,
                n_noise_repeat=n_noise_repeat
            )
            scores.append(batch_scores.detach().cpu())

        if len(scores) == 0:
            return np.array([])

        return torch.cat(scores, dim=0).numpy()

    @torch.no_grad()
    def score_ehr_mia_bank_multi_t(
        z_bank_cpu,
        c_bank_cpu,
        alpha_bar,
        t_candidates=(25, 50, 75, 100, 125, 150, 175, 200),
        n_noise_repeat=2,
        score_batch_size=256
    ):
        """
        对缓存好的 bank，在多个 t 上求平均 x0-recon score
        """
        if z_bank_cpu is None or c_bank_cpu is None:
            return np.array([])

        scores = []
        total_n = z_bank_cpu.size(0)

        for start in range(0, total_n, score_batch_size):
            end = min(start + score_batch_size, total_n)

            z0 = z_bank_cpu[start:end].to(device, non_blocking=True)
            context = c_bank_cpu[start:end].to(device, non_blocking=True)

            batch_scores = x0_recon_error_score_multi_t(
                z0=z0,
                context=context,
                alpha_bar=alpha_bar,
                t_candidates=t_candidates,
                n_noise_repeat=n_noise_repeat
            )
            scores.append(batch_scores.detach().cpu())

        if len(scores) == 0:
            return np.array([])

        return torch.cat(scores, dim=0).numpy()

    def save_ehr_mia_scan_results(results, save_path):
        if results is None or len(results) == 0:
            return

        fieldnames = [
            "t_sec", "num_member", "num_holdout",
            "AUC", "ASR", "TPR@1%FPR", "TPR@0.1%FPR",
            "member_mean", "holdout_mean", "member_std", "holdout_std"
        ]

        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    def run_ehr_x0_mia_eval_fixed_t(
        member_loader,
        holdout_loader,
        t_sec=100,
        n_noise_repeat=4,
        max_member_batches=None,
        max_holdout_batches=None,
        score_batch_size=256,
    ):
        print("[EHR-X0-MIA] collecting member bank...")
        member_z_bank, member_c_bank = collect_ehr_mia_bank(
            member_loader, max_batches=max_member_batches
        )

        print("[EHR-X0-MIA] collecting holdout bank...")
        holdout_z_bank, holdout_c_bank = collect_ehr_mia_bank(
            holdout_loader, max_batches=max_holdout_batches
        )

        member_scores = score_ehr_mia_bank_fixed_t(
            member_z_bank,
            member_c_bank,
            alpha_bar=alpha_bar,
            t_sec=t_sec,
            n_noise_repeat=n_noise_repeat,
            score_batch_size=score_batch_size
        )

        holdout_scores = score_ehr_mia_bank_fixed_t(
            holdout_z_bank,
            holdout_c_bank,
            alpha_bar=alpha_bar,
            t_sec=t_sec,
            n_noise_repeat=n_noise_repeat,
            score_batch_size=score_batch_size
        )

        metrics = compute_mia_metrics(member_scores, holdout_scores)

        print("[EHR X0-MIA Fixed-t]")
        print(f"  t_sec      = {t_sec}")
        print(f"  repeat     = {n_noise_repeat}")
        print(f"  #member    = {len(member_scores)}")
        print(f"  #holdout   = {len(holdout_scores)}")
        print(f"  AUC        = {metrics['AUC']:.4f}")
        print(f"  ASR        = {metrics['ASR']:.4f}")
        print(f"  TPR@1%     = {metrics['TPR@1%FPR']:.4f}")
        print(f"  TPR@0.1%   = {metrics['TPR@0.1%FPR']:.4f}")
        print(f"  member_mu  = {metrics['member_mean']:.8e}")
        print(f"  holdout_mu = {metrics['holdout_mean']:.8e}")

        return metrics, member_scores, holdout_scores

    def run_ehr_x0_mia_scan(
        member_loader,
        holdout_loader,
        t_sec_list=(25, 50, 75, 100, 125, 150, 175, 200),
        n_noise_repeat=4,
        max_member_batches=None,
        max_holdout_batches=None,
        score_batch_size=256,
    ):
        """
        扫多个 t_sec，看哪个时间步最容易暴露 membership signal
        """
        print("[EHR-X0-MIA Scan] collecting member bank once...")
        member_z_bank, member_c_bank = collect_ehr_mia_bank(
            member_loader, max_batches=max_member_batches
        )

        print("[EHR-X0-MIA Scan] collecting holdout bank once...")
        holdout_z_bank, holdout_c_bank = collect_ehr_mia_bank(
            holdout_loader, max_batches=max_holdout_batches
        )

        results = []

        for t_sec in t_sec_list:
            member_scores = score_ehr_mia_bank_fixed_t(
                member_z_bank,
                member_c_bank,
                alpha_bar=alpha_bar,
                t_sec=t_sec,
                n_noise_repeat=n_noise_repeat,
                score_batch_size=score_batch_size
            )

            holdout_scores = score_ehr_mia_bank_fixed_t(
                holdout_z_bank,
                holdout_c_bank,
                alpha_bar=alpha_bar,
                t_sec=t_sec,
                n_noise_repeat=n_noise_repeat,
                score_batch_size=score_batch_size
            )

            metrics = compute_mia_metrics(member_scores, holdout_scores)

            row = {
                "t_sec": int(t_sec),
                "num_member": int(len(member_scores)),
                "num_holdout": int(len(holdout_scores)),
                "AUC": float(metrics["AUC"]),
                "ASR": float(metrics["ASR"]),
                "TPR@1%FPR": float(metrics["TPR@1%FPR"]),
                "TPR@0.1%FPR": float(metrics["TPR@0.1%FPR"]),
                "member_mean": float(metrics["member_mean"]),
                "holdout_mean": float(metrics["holdout_mean"]),
                "member_std": float(metrics["member_std"]),
                "holdout_std": float(metrics["holdout_std"]),
            }
            results.append(row)

            print(
                f"[EHR-X0-MIA Scan] t_sec={t_sec:>3d} | "
                f"AUC={row['AUC']:.4f} | "
                f"ASR={row['ASR']:.4f} | "
                f"TPR@1%={row['TPR@1%FPR']:.4f} | "
                f"TPR@0.1%={row['TPR@0.1%FPR']:.4f} | "
                f"member_mu={row['member_mean']:.8e} | "
                f"holdout_mu={row['holdout_mean']:.8e}"
            )

        if len(results) > 0:
            best_auc_row = max(results, key=lambda x: x["AUC"])
            best_asr_row = max(results, key=lambda x: x["ASR"])

            print("[EHR-X0-MIA Scan] best by AUC:")
            print(best_auc_row)

            print("[EHR-X0-MIA Scan] best by ASR:")
            print(best_asr_row)

        return results

    def run_ehr_x0_mia_eval_multi_t(
        member_loader,
        holdout_loader,
        t_candidates=(25, 50, 75, 100, 125, 150, 175, 200),
        n_noise_repeat=2,
        max_member_batches=None,
        max_holdout_batches=None,
        score_batch_size=256,
    ):
        """
        对多个 t 的 x0 重建误差取平均，得到更稳的 patient-level score
        """
        print("[EHR-X0-MIA Multi-t] collecting member bank...")
        member_z_bank, member_c_bank = collect_ehr_mia_bank(
            member_loader, max_batches=max_member_batches
        )

        print("[EHR-X0-MIA Multi-t] collecting holdout bank...")
        holdout_z_bank, holdout_c_bank = collect_ehr_mia_bank(
            holdout_loader, max_batches=max_holdout_batches
        )

        member_scores = score_ehr_mia_bank_multi_t(
            member_z_bank,
            member_c_bank,
            alpha_bar=alpha_bar,
            t_candidates=t_candidates,
            n_noise_repeat=n_noise_repeat,
            score_batch_size=score_batch_size
        )

        holdout_scores = score_ehr_mia_bank_multi_t(
            holdout_z_bank,
            holdout_c_bank,
            alpha_bar=alpha_bar,
            t_candidates=t_candidates,
            n_noise_repeat=n_noise_repeat,
            score_batch_size=score_batch_size
        )

        metrics = compute_mia_metrics(member_scores, holdout_scores)

        print("[EHR X0-MIA Multi-t]")
        print(f"  t_candidates = {list(t_candidates)}")
        print(f"  repeat       = {n_noise_repeat}")
        print(f"  #member      = {len(member_scores)}")
        print(f"  #holdout     = {len(holdout_scores)}")
        print(f"  AUC          = {metrics['AUC']:.4f}")
        print(f"  ASR          = {metrics['ASR']:.4f}")
        print(f"  TPR@1%       = {metrics['TPR@1%FPR']:.4f}")
        print(f"  TPR@0.1%     = {metrics['TPR@0.1%FPR']:.4f}")
        print(f"  member_mu    = {metrics['member_mean']:.8e}")
        print(f"  holdout_mu   = {metrics['holdout_mean']:.8e}")

        return metrics, member_scores, holdout_scores
    ######################### 3 #############################

    step_quality = []
    jaccard_quality = []
    # while train_state.step < config['train']['n_steps']:
    max_n_steps = (len(train_dataset) // config['train']['batch_size']) + 1 if len(train_dataset) % config['train']['batch_size'] !=0 else len(train_dataset) // config['train']['batch_size'] # 不对，如果正好整除的话，就不需要+1
    for epoch in range(config['train']['n_epochs']):
        if epoch == 0:
            nnet.e_step()  # 初始化
        # print("AAAAAAAAA", nnet.total_visit_emb.shape, max_n_steps, len(train_dataset))
        total_visit_embs = []
        total_indices = []
        for i in range(max_n_steps):
            nnet.train()
            # batch = tree_map(lambda x: convert_input(x), next(data_generator))  # tree_map更快
            indices, batch = convert_input(next(data_generator), mask_static=False)

            _z, metrics = train_step(indices, batch, epoch)

            total_visit_embs.append(_z.view(_z.shape[0],-1)) # B, 3D
            total_indices.append(indices) # B,1

            nnet.eval()
            # 记录metrics, 输出train_state.step, 以及config.train.log_interval
            # if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            if train_state.step % config['train']['log_interval'] == 0:
                logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
                logging.info(config['workdir'])
                wandb.log(metrics, step=train_state.step) # 不同step

            # 中途输出
            # if train_state.step % config['train']['eval_interval'] == 0:  # 任何记录日志，保存模型都要使用
            #     torch.cuda.empty_cache()
                logging.info('Save a grid of {} samples for training...'.format(config['train_val_num']))
                sample_input = dataset.get_contexts(num=config['train_val_num'])
                indices = torch.arange(config['train_val_num']).to(device)
                indices, sample_input = convert_input((indices, sample_input))
                _z, contexts, _, labels, hard_labels = nnet.encode_everything_test(indices, **sample_input)

                # contexts = torch.tensor(dataset.contexts, device=device)[: 2 * 5] # 选取前10个context
                samples, hard_samples = dpm_solver_sample(_n_samples=config['train_val_num'], _sample_steps=50, context=contexts, z_init=_z, mask=sample_input['mask'])
                # print("generate", hard_samples['conditions'][0])
                # print("origin", tokenizers['conditions'].batch_decode_2d(hard_labels['conditions'])[0]) # 没啥问题，encode, decode用同一套
                # print("generate", hard_samples['procedures'][0])
                # print("origin", tokenizers['procedures'].batch_decode_2d(hard_labels['procedures'])[0])  # 没啥问题，encode, decode用同一套
                # print("generate", hard_samples['drugs'][0])
                # print("origin", tokenizers['drugs'].batch_decode_2d(hard_labels['drugs'])[0])  # 没啥问题，encode, decode用同一套
                # samples = make_grid(dataset.unpreprocess(samples), 5)
                # save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
                # wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
                jaccard_condition = utils.calculate_average_jaccard(hard_samples['conditions'], tokenizers['conditions'].batch_decode_2d(hard_labels['conditions']))
                jaccard_procedure = utils.calculate_average_jaccard(hard_samples['procedures'], tokenizers['procedures'].batch_decode_2d(hard_labels['procedures']))
                jaccard_drug = utils.calculate_average_jaccard(hard_samples['drugs'], tokenizers['drugs'].batch_decode_2d(hard_labels['drugs']))
                mse_loss = F.mse_loss(samples, labels.squeeze())
                logging.info({'mask num': sample_input['mask'].sum(dim=0), 'val samples mse': mse_loss, 'jaccard': (jaccard_condition,jaccard_procedure,jaccard_drug)})
                # wandb.log({'samples': mse_loss }, step=train_state.step)
                # torch.cuda.empty_cache()
            # accelerator.wait_for_everyone()

            # save ckpt， 这一块代码检查当前进程是否为主进程（accelerator.is_main_process），如果是则保存模型
        # if train_state.step % config['train']['save_interval'] == 0 or train_state.step == config['train']['n_steps']:
        # torch.cuda.empty_cache()

        ###################
        train_mse_ep, train_jacc_ep = train_epoch_metrics(
            sample_steps=config['sample']['sample_steps'],
            num_batches=config['train'].get('train_eval_batches', 2),
            mask_static=True  # 更稳定；想贴近训练mask可设False
        )

        logging.info(f'[TRAIN EVAL] epoch={epoch} train_mse={train_mse_ep} train_jacc={train_jacc_ep}')
        wandb.log({
            'train_epoch/mse': train_mse_ep,
            'train_epoch/jaccard_conditions': train_jacc_ep[0],
            'train_epoch/jaccard_procedures': train_jacc_ep[1],
            'train_epoch/jaccard_drugs': train_jacc_ep[2],
            'train_epoch/jaccard_mean': (train_jacc_ep[0] + train_jacc_ep[1] + train_jacc_ep[2]) / 3.0,
        }, step=train_state.step)
        ###################

        logging.info(f'Save and eval checkpoint {epoch}...')
        # if accelerator.local_process_index == 0:
        train_state.save(os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch}.ckpt'))
        # accelerator.wait_for_everyone()
        if epoch==0 or (epoch>=5 and epoch % config['train']['eval_epoch'] == 0):
            # quality, jaccard_tuple = eval_step(n_samples=len(test_dataset), sample_steps=50)  # calculate fid of the saved checkpoint
            quality, jaccard_tuple = eval_step(n_samples=len(test_dataset), sample_steps=config['sample']['sample_steps'])
        else:
            quality = 10000 # 因为靠mse取最小的mse
            jaccard_tuple = (1,1,1)
        step_quality.append((epoch, quality))
        jaccard_quality.append((epoch, jaccard_tuple))
        # torch.cuda.empty_cache()

        # e step
        total_visit_embs = torch.cat(total_visit_embs, dim=0)
        total_indices = torch.cat(total_indices, dim=0)
        total_visit_embs = total_visit_embs[total_indices.argsort()] # 重新排序， 这里需要check下对不对。
        print("CCCCCCCC", total_visit_embs.shape)
        nnet.update_train_visit_emb(total_visit_embs)
        visit_centroids, _ = nnet.e_step()
        nnet.update_centroid_emb(visit_centroids)



    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_quality: {step_quality}')
    logging.info(f'step_quality: {jaccard_quality}')
    epoch_best = sorted(step_quality, key=lambda x: x[1])[0][0] # 最小那个mse
    logging.info(f'step_best: {epoch_best}')
    train_state.load(os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch_best}.ckpt'))
    del metrics
    # accelerator.wait_for_everyone()
    eval_step(n_samples=len(test_dataset), sample_steps=config['sample']['sample_steps'])  # 选取最好的ckpt
    # ########################## 2 ##########################
    # # ===== EHR Noise-MIA: fixed t =====
    # ehr_mia_fixed_metrics, member_scores_fixed, holdout_scores_fixed = run_ehr_noise_mia_eval_fixed_t(
    #     member_loader=train_eval_loader,
    #     holdout_loader=test_dataset_loader,
    #     t_sec=100,
    #     n_noise_repeat=4,
    #     max_member_batches=None,
    #     max_holdout_batches=None,
    #     score_batch_size=256
    # )
    #
    # privacy_log_path = os.path.join(config['workdir'] + data_n + '/' + task, 'privacy_ehr_noise_mia.log')
    # with open(privacy_log_path, 'a', encoding='utf-8') as f:
    #     f.write("[EHR Noise-MIA fixed-t]\n")
    #     f.write(json.dumps(ehr_mia_fixed_metrics, ensure_ascii=False) + "\n")
    #
    # wandb.log({
    #     'ehr_mia_fixed/AUC': ehr_mia_fixed_metrics['AUC'],
    #     'ehr_mia_fixed/ASR': ehr_mia_fixed_metrics['ASR'],
    #     'ehr_mia_fixed/TPR@1%FPR': ehr_mia_fixed_metrics['TPR@1%FPR'],
    #     'ehr_mia_fixed/TPR@0.1%FPR': ehr_mia_fixed_metrics['TPR@0.1%FPR'],
    #     'ehr_mia_fixed/member_mean': ehr_mia_fixed_metrics['member_mean'],
    #     'ehr_mia_fixed/holdout_mean': ehr_mia_fixed_metrics['holdout_mean'],
    # }, step=train_state.step)
    #
    # # ===== EHR Noise-MIA: scan multiple t =====
    # ehr_mia_scan_results = run_ehr_noise_mia_scan(
    #     member_loader=train_eval_loader,
    #     holdout_loader=test_dataset_loader,
    #     t_sec_list=[25, 50, 75, 100, 125, 150, 175, 200],
    #     n_noise_repeat=4,
    #     max_member_batches=None,
    #     max_holdout_batches=None,
    #     score_batch_size=256
    # )
    #
    # scan_csv_path = os.path.join(config['workdir'] + data_n + '/' + task, 'privacy_ehr_noise_mia_scan.csv')
    # save_ehr_mia_scan_results(ehr_mia_scan_results, scan_csv_path)
    #
    # with open(privacy_log_path, 'a', encoding='utf-8') as f:
    #     f.write("[EHR Noise-MIA scan]\n")
    #     for row in ehr_mia_scan_results:
    #         f.write(json.dumps(row, ensure_ascii=False) + "\n")
    #
    # if len(ehr_mia_scan_results) > 0:
    #     best_auc_row = max(ehr_mia_scan_results, key=lambda x: x["AUC"])
    #     best_asr_row = max(ehr_mia_scan_results, key=lambda x: x["ASR"])
    #
    #     wandb.log({
    #         'ehr_mia_scan/best_AUC': best_auc_row['AUC'],
    #         'ehr_mia_scan/best_AUC_t_sec': best_auc_row['t_sec'],
    #         'ehr_mia_scan/best_ASR': best_asr_row['ASR'],
    #         'ehr_mia_scan/best_ASR_t_sec': best_asr_row['t_sec'],
    #     }, step=train_state.step)
    #
    #     print("[EHR Noise-MIA Scan] saved to:", scan_csv_path)
    #     print("[EHR Noise-MIA Scan] best_auc_row =", best_auc_row)
    #     print("[EHR Noise-MIA Scan] best_asr_row =", best_asr_row)
    #
    # # ===== EHR Noise-MIA: multi-t averaged patient score =====
    # ehr_mia_multi_metrics, member_scores_multi, holdout_scores_multi = run_ehr_noise_mia_eval_multi_t(
    #     member_loader=train_eval_loader,
    #     holdout_loader=test_dataset_loader,
    #     t_candidates=[25, 50, 75, 100, 125, 150, 175, 200],
    #     n_noise_repeat=2,
    #     max_member_batches=None,
    #     max_holdout_batches=None,
    #     score_batch_size=256
    # )
    #
    # with open(privacy_log_path, 'a', encoding='utf-8') as f:
    #     f.write("[EHR Noise-MIA multi-t]\n")
    #     f.write(json.dumps(ehr_mia_multi_metrics, ensure_ascii=False) + "\n")
    #
    # wandb.log({
    #     'ehr_mia_multi/AUC': ehr_mia_multi_metrics['AUC'],
    #     'ehr_mia_multi/ASR': ehr_mia_multi_metrics['ASR'],
    #     'ehr_mia_multi/TPR@1%FPR': ehr_mia_multi_metrics['TPR@1%FPR'],
    #     'ehr_mia_multi/TPR@0.1%FPR': ehr_mia_multi_metrics['TPR@0.1%FPR'],
    #     'ehr_mia_multi/member_mean': ehr_mia_multi_metrics['member_mean'],
    #     'ehr_mia_multi/holdout_mean': ehr_mia_multi_metrics['holdout_mean'],
    # }, step=train_state.step)
    # ########################## 2 ##########################
    # ########################## 1 ##########################
    # # ===== SecMI privacy evaluation =====
    # # ===== SecMI 单点评估（例如 t_sec=100）=====
    # secmi_metrics, member_scores, holdout_scores = run_secmi_privacy_eval(
    #     member_loader=train_eval_loader,
    #     holdout_loader=test_dataset_loader,
    #     t_sec=100,  ## t_sec 只是决定在哪个时间位置看这一步
    #     # stride=10,
    #     stride=1,
    #     max_member_batches=None,
    #     max_holdout_batches=None,
    #     score_batch_size=256
    # )
    # privacy_log_path = os.path.join(config['workdir'] + data_n + '/' + task, 'privacy_secmi.log')
    # with open(privacy_log_path, 'a', encoding='utf-8') as f:
    #     f.write("[SecMI single t]\n")
    #     f.write(json.dumps(secmi_metrics, ensure_ascii=False) + "\n")
    # wandb.log({
    #     'privacy/AUC': secmi_metrics['AUC'],
    #     'privacy/ASR': secmi_metrics['ASR'],
    #     'privacy/TPR@1%FPR': secmi_metrics['TPR@1%FPR'],
    #     'privacy/TPR@0.1%FPR': secmi_metrics['TPR@0.1%FPR'],
    #     'privacy/member_mean': secmi_metrics['member_mean'],
    #     'privacy/holdout_mean': secmi_metrics['holdout_mean'],
    # }, step=train_state.step)
    #
    # # ===== SecMI 多个 t_sec 扫描 =====
    # secmi_scan_results = run_secmi_tsec_scan(
    #     member_loader=train_eval_loader,
    #     holdout_loader=test_dataset_loader,
    #     t_sec_list=[25, 50, 75, 100, 125, 150, 175, 200],
    #     # stride=10,
    #     stride=1,
    #     max_member_batches=None,
    #     max_holdout_batches=None,
    #     score_batch_size=256,
    # )
    #
    # scan_csv_path = os.path.join(config['workdir'] + data_n + '/' + task, 'privacy_secmi_scan.csv')
    # save_secmi_scan_results(secmi_scan_results, scan_csv_path)
    #
    # with open(privacy_log_path, 'a', encoding='utf-8') as f:
    #     f.write("[SecMI scan]\n")
    #     for row in secmi_scan_results:
    #         f.write(json.dumps(row, ensure_ascii=False) + "\n")
    #
    # if len(secmi_scan_results) > 0:
    #     best_auc_row = max(secmi_scan_results, key=lambda x: x["AUC"])
    #     best_asr_row = max(secmi_scan_results, key=lambda x: x["ASR"])
    #
    #     wandb.log({
    #         'privacy_scan/best_AUC': best_auc_row['AUC'],
    #         'privacy_scan/best_AUC_t_sec': best_auc_row['t_sec'],
    #         'privacy_scan/best_ASR': best_asr_row['ASR'],
    #         'privacy_scan/best_ASR_t_sec': best_asr_row['t_sec'],
    #     }, step=train_state.step)
    #
    #     print("[SecMI Scan] saved to:", scan_csv_path)
    #     print("[SecMI Scan] best_auc_row =", best_auc_row)
    #     print("[SecMI Scan] best_asr_row =", best_asr_row)
    # ########################## 1 ##########################

    ########################## 3 ##########################
    # ===== EHR X0-MIA: fixed t =====
    ehr_x0_mia_fixed_metrics, member_scores_fixed, holdout_scores_fixed = run_ehr_x0_mia_eval_fixed_t(
        member_loader=train_eval_loader,
        holdout_loader=test_dataset_loader,
        t_sec=100,
        n_noise_repeat=4,
        max_member_batches=None,
        max_holdout_batches=None,
        score_batch_size=256
    )

    privacy_log_path = os.path.join(config['workdir'] + data_n + '/' + task, 'privacy_ehr_x0_mia.log')
    with open(privacy_log_path, 'a', encoding='utf-8') as f:
        f.write("[EHR X0-MIA fixed-t]\n")
        f.write(json.dumps(ehr_x0_mia_fixed_metrics, ensure_ascii=False) + "\n")

    wandb.log({
        'ehr_x0_mia_fixed/AUC': ehr_x0_mia_fixed_metrics['AUC'],
        'ehr_x0_mia_fixed/ASR': ehr_x0_mia_fixed_metrics['ASR'],
        'ehr_x0_mia_fixed/TPR@1%FPR': ehr_x0_mia_fixed_metrics['TPR@1%FPR'],
        'ehr_x0_mia_fixed/TPR@0.1%FPR': ehr_x0_mia_fixed_metrics['TPR@0.1%FPR'],
        'ehr_x0_mia_fixed/member_mean': ehr_x0_mia_fixed_metrics['member_mean'],
        'ehr_x0_mia_fixed/holdout_mean': ehr_x0_mia_fixed_metrics['holdout_mean'],
    }, step=train_state.step)

    # ===== EHR X0-MIA: scan multiple t =====
    ehr_x0_mia_scan_results = run_ehr_x0_mia_scan(
        member_loader=train_eval_loader,
        holdout_loader=test_dataset_loader,
        t_sec_list=[25, 50, 75, 100, 125, 150, 175, 200],
        n_noise_repeat=4,
        max_member_batches=None,
        max_holdout_batches=None,
        score_batch_size=256
    )

    scan_csv_path = os.path.join(config['workdir'] + data_n + '/' + task, 'privacy_ehr_x0_mia_scan.csv')
    save_ehr_mia_scan_results(ehr_x0_mia_scan_results, scan_csv_path)

    with open(privacy_log_path, 'a', encoding='utf-8') as f:
        f.write("[EHR X0-MIA scan]\n")
        for row in ehr_x0_mia_scan_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if len(ehr_x0_mia_scan_results) > 0:
        best_auc_row = max(ehr_x0_mia_scan_results, key=lambda x: x["AUC"])
        best_asr_row = max(ehr_x0_mia_scan_results, key=lambda x: x["ASR"])

        wandb.log({
            'ehr_x0_mia_scan/best_AUC': best_auc_row['AUC'],
            'ehr_x0_mia_scan/best_AUC_t_sec': best_auc_row['t_sec'],
            'ehr_x0_mia_scan/best_ASR': best_asr_row['ASR'],
            'ehr_x0_mia_scan/best_ASR_t_sec': best_asr_row['t_sec'],
        }, step=train_state.step)

        print("[EHR X0-MIA Scan] saved to:", scan_csv_path)
        print("[EHR X0-MIA Scan] best_auc_row =", best_auc_row)
        print("[EHR X0-MIA Scan] best_asr_row =", best_asr_row)

    # ===== EHR X0-MIA: multi-t averaged patient score =====
    ehr_x0_mia_multi_metrics, member_scores_multi, holdout_scores_multi = run_ehr_x0_mia_eval_multi_t(
        member_loader=train_eval_loader,
        holdout_loader=test_dataset_loader,
        t_candidates=[25, 50, 75, 100, 125, 150, 175, 200],
        n_noise_repeat=2,
        max_member_batches=None,
        max_holdout_batches=None,
        score_batch_size=256
    )

    with open(privacy_log_path, 'a', encoding='utf-8') as f:
        f.write("[EHR X0-MIA multi-t]\n")
        f.write(json.dumps(ehr_x0_mia_multi_metrics, ensure_ascii=False) + "\n")

    wandb.log({
        'ehr_x0_mia_multi/AUC': ehr_x0_mia_multi_metrics['AUC'],
        'ehr_x0_mia_multi/ASR': ehr_x0_mia_multi_metrics['ASR'],
        'ehr_x0_mia_multi/TPR@1%FPR': ehr_x0_mia_multi_metrics['TPR@1%FPR'],
        'ehr_x0_mia_multi/TPR@0.1%FPR': ehr_x0_mia_multi_metrics['TPR@0.1%FPR'],
        'ehr_x0_mia_multi/member_mean': ehr_x0_mia_multi_metrics['member_mean'],
        'ehr_x0_mia_multi/holdout_mean': ehr_x0_mia_multi_metrics['holdout_mean'],
    }, step=train_state.step)
    ########################## 3 ##########################


    nnet_path = os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch_best}.ckpt/nnet.pth')
    return nnet_path, train_state.nnet, train_state.nnet.total_visit_emb.shape[0]