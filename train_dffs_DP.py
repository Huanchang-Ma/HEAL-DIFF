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
import torch.nn as nn
import shutil
import glob
import pandas as pd

class DPDenoiserWrapper(nn.Module):
    def __init__(self, denoiser: nn.Module):
        super().__init__()
        self.denoiser = denoiser
        self._timesteps = None
        self._context = None

    def set_conditioning(self, timesteps, context):
        self._timesteps = timesteps
        self._context = context

    def forward(self, x):
        if self._timesteps is None or self._context is None:
            raise RuntimeError("Conditioning is not set before DP forward.")
        return self.denoiser(x, self._timesteps, context=self._context)

def append_row_to_csv(csv_path, row_dict):
    row_df = pd.DataFrame([row_dict])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        new_df = pd.concat([old_df, row_df], ignore_index=True)
    else:
        new_df = row_df

    new_df.to_csv(csv_path, index=False)

def run_diffusion_DP(config, config_health, hp_dataset, tokenizers, exp_num='0'):
    """
    :param config: diffusion config
    :param config_health: healthcare config
    :param hp_dataset: healthcare train dataset
    :param tokenizers: healthcare tokenizers
    :return:
    """
    """
        DP-SGD version of run_diffusion
        """
    import os
    import tempfile
    import logging
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import einops
    import wandb

    from contextlib import nullcontext
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    # -----------------------------
    # helpers
    # -----------------------------
    def _unwrap_model(m):
        return m._module if hasattr(m, "_module") else m

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
                dp_attn = dp_attn.to(
                    device=child.out_proj.weight.device,
                    dtype=child.out_proj.weight.dtype,
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

    def _find_trainable_modules_with_buffers(model):
        bad = []
        for mod_name, mod in model.named_modules():
            direct_buffers = list(mod.named_buffers(recurse=False))
            if not direct_buffers:
                continue
            has_trainable = any(p.requires_grad for p in mod.parameters())
            if has_trainable:
                bad.append({
                    "module": mod_name if mod_name else "<root>",
                    "type": mod.__class__.__name__,
                    "buffers": [name for name, _ in direct_buffers],
                })
        return bad

    def _remove_root_buffers_for_dp(model):
        removed = {}
        for name in ["total_visit_emb", "visit_centroids"]:
            if name in model._buffers:
                buf = model._buffers.pop(name)
                setattr(model, name, buf)
                removed[name] = tuple(buf.shape) if hasattr(buf, "shape") else str(type(buf))
        return removed

    def _list_all_buffers(model):
        bufs = []
        for name, buf in model.named_buffers(recurse=True):
            shape = tuple(buf.shape) if hasattr(buf, "shape") else str(type(buf))
            bufs.append((name, shape))
        return bufs

    def _remove_buffers_by_fullname(model, buffer_full_names):
        removed = {}
        for full_name in buffer_full_names:
            if "." in full_name:
                mod_path, buf_name = full_name.rsplit(".", 1)
                mod = model.get_submodule(mod_path)
            else:
                mod = model
                buf_name = full_name

            if buf_name in mod._buffers:
                buf = mod._buffers.pop(buf_name)
                setattr(mod, buf_name, buf)
                removed[full_name] = tuple(buf.shape) if hasattr(buf, "shape") else str(type(buf))
        return removed

    def _find_functorch_unfriendly_modules(model):
        import inspect
        bad = []
        for name, m in model.named_modules():
            try:
                sig = inspect.signature(m.forward)
            except (TypeError, ValueError):
                continue

            extra_args = [k for k in ("timesteps", "context") if k in sig.parameters]
            direct_trainable = [pn for pn, p in m.named_parameters(recurse=False) if p.requires_grad]
            if extra_args and direct_trainable:
                bad.append({
                    "module": name if name else "<root>",
                    "type": m.__class__.__name__,
                    "extra_args": extra_args,
                    "direct_trainable": direct_trainable,
                })
        return bad

    def _find_missing_grad_sample(model):
        target = _unwrap_model(model)
        bad = []
        for name, p in target.named_parameters():
            if p.requires_grad and getattr(p, "grad_sample", None) is None:
                bad.append(name)
        return bad

    def _freeze_dp_unused_params(module):

        freeze_prefixes = [
            "denoiser.visit_encoder.",
            "denoiser.context_encoder.",
            "denoiser.lm_heads.",
            "denoiser.note_linear.",
        ]

        frozen = []
        for name, p in module.named_parameters():
            if any(name.startswith(pref) for pref in freeze_prefixes):
                p.requires_grad = False
                frozen.append(name)
        return frozen


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
    # os.makedirs(config['ckpt_root'] + data_n + '/' + task, exist_ok=True)
    # os.makedirs(config['sample_dir'] + data_n + '/' + task, exist_ok=True)

    # exp_tag = f"eps{str(config['train']['dp_target_epsilon']).replace('.', 'p')}"
    # run_name = f"{task}_{exp_tag}_{exp_num}"
    # ckpt_dir = os.path.join(config['ckpt_root'], data_n, task, run_name)
    eps_tag = f"eps{str(config['train']['dp_target_epsilon']).replace('.', 'p')}"
    ckpt_dir = os.path.join(config['ckpt_root'], data_n, task, eps_tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    # accelerator.wait_for_everyone()

    # if accelerator.is_main_process:
    wandb.init(dir=os.path.abspath(config['workdir']), project=f'mmht_{data_n}_{task}', config=config,
               name=config['hparams'], job_type='train', mode='offline')  # 离线存储
    utils.set_logger(log_level='info', fname=os.path.join(config['workdir'], data_n +'-'+ task +'-' + exp_num + '-output.log'))
    logging.info(config)

    ################################
    step_history = {
        'step': [],
        'loss_mse': [],
        'loss_c': [],
        'loss_rounding': [],
        'T_loss': [],
        'total_loss': [],
        'lr': [],
    }
    epoch_history = {
        'epoch': [],
        'quality_mse': [],
        'quality_jaccard_conditions': [],
        'quality_jaccard_procedures': [],
        'quality_jaccard_drugs': [],
        'train_mse': [],
        'train_jaccard_conditions': [],
        'train_jaccard_procedures': [],
        'train_jaccard_drugs': [],
        'train_jaccard_mean': [],
    }
    ################################

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
    # train_state.resume(config['ckpt_root'] + data_n + '/' + task)  # load latest ckpt
    if config['train'].get('resume', False):
        train_state.resume(ckpt_dir)

    ###################### DP-SGD #######################
    from contextlib import nullcontext

    privacy_engine = None
    use_dp = bool(config['train'].get('dp_sgd', False))

    if use_dp:
        from opacus import PrivacyEngine
        from opacus.utils.batch_memory_manager import BatchMemoryManager
        from opacus.validators import ModuleValidator

        privacy_engine = PrivacyEngine(
            accountant=config['train'].get('dp_accountant', 'prv'),
            secure_mode=bool(config['train'].get('dp_secure_mode', False)),
        )

        bad = _find_trainable_modules_with_buffers(nnet)
        print("[DP] trainable modules with buffers:", bad)

        replaced = _replace_multihead_attention_for_dp(nnet)
        print(f"[DP] replaced multihead attention count = {replaced}")

        bn_replaced = _replace_batchnorm_for_dp(nnet)
        print(f"[DP] replaced batchnorm count = {bn_replaced}")

        errors = ModuleValidator.validate(nnet, strict=False)
        print("[DP] validator errors after replacement:", errors)
        if errors:
            raise RuntimeError(errors)

        removed = _remove_root_buffers_for_dp(nnet)
        print("[DP] removed root buffers for DP:", removed)

        print("[DP] all buffers before make_private:", _list_all_buffers(nnet))
        all_bufs = [name for name, _ in nnet.named_buffers(recurse=True)]
        print("[DP] all buffers before removal:", all_bufs)
        removed = _remove_buffers_by_fullname(nnet, all_bufs)
        print("[DP] removed buffers:", removed)
        print("[DP] all buffers after removal:", list(nnet.named_buffers(recurse=True)))

        # 根模块直接参数里，pos_embed 会触发 functorch fallback，先冻结
        if hasattr(nnet, "pos_embed") and isinstance(nnet.pos_embed, torch.nn.Parameter):
            nnet.pos_embed.requires_grad = False
            print("[DP] froze root parameter: pos_embed")

        bad2 = _find_functorch_unfriendly_modules(nnet)
        print("[DP] functorch-unfriendly trainable modules after freezing:", bad2)

        # 所有替换结束后，重新放回 device
        # nnet = nnet.to(device)

        # # 重建 optimizer / scheduler
        # optimizer = torch.optim.AdamW(
        #     [p for p in nnet.parameters() if p.requires_grad],
        #     lr=config['optimizer']['lr'],
        #     weight_decay=config['optimizer'].get('weight_decay', 0.0),
        # )

        base_nnet = train_state.nnet
        if use_dp and (not hasattr(base_nnet, "visit_2cluster") or base_nnet.visit_2cluster is None):
            base_nnet.visit_2cluster = torch.zeros(
                len(train_dataset),
                dtype=torch.long,
                device=device,
            )
            print("[DP] initialized dummy visit_2cluster")

        print("[DP] make_private grad_sample_mode = hooks")

        dp_nnet = DPDenoiserWrapper(base_nnet).to(device)

        frozen = _freeze_dp_unused_params(dp_nnet)
        print("[DP] frozen params:", frozen[:20], "... total =", len(frozen))

        optimizer = torch.optim.AdamW(
            [p for p in dp_nnet.parameters() if p.requires_grad],
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer'].get('weight_decay', 0.0),
        )

        dp_nnet, optimizer, train_dataset_loader = privacy_engine.make_private_with_epsilon(
            module=dp_nnet,
            optimizer=optimizer,
            data_loader=train_dataset_loader,
            target_epsilon=float(config['train']['dp_target_epsilon']),
            target_delta=float(config['train']['dp_delta']),
            epochs=int(config['train']['n_epochs']),
            max_grad_norm=float(config['train']['dp_max_grad_norm']),
            poisson_sampling=bool(config['train'].get('dp_poisson_sampling', True)),
            grad_sample_mode='hooks',
            clipping=str(config['train'].get('dp_clipping', 'flat')),
        )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=(
                train_state.lr_scheduler.lr_lambdas[0]
                if hasattr(train_state.lr_scheduler, "lr_lambdas")
                else (lambda step: 1.0)
            ),
        )

        train_state.nnet = base_nnet
        train_state.dp_nnet = dp_nnet
        train_state.optimizer = optimizer
        train_state.lr_scheduler = lr_scheduler


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

    # def cfg_nnet(x, timesteps, context):
    #     # classifier-free config
    #     _cond = nnet_ema(x, timesteps, context=context)
    #     _empty_context = torch.tensor(dataset.empty_context, device=device)
    #     _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
    #     _uncond = nnet_ema(x, timesteps, context=_empty_context)
    #     return _cond #+ config['sample']['scale'] * (_cond - _uncond) # classifier free 要用unconditional的表征进行放缩。
    def cfg_nnet(x, timesteps, context):
        _cond = train_state.nnet(x, timesteps, context=context)
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = train_state.nnet(x, timesteps, context=_empty_context)
        return _cond

    def train_step(indices, _batch, epoch=None):
        _metrics = dict()
        optimizer.zero_grad(set_to_none=True)

        base_nnet = train_state.nnet
        dp_model = train_state.dp_nnet

        dp_model.train()

        # 原始模型只负责抽取 z/context，不参与 DP backward
        with torch.no_grad():
            _z, _context, neigh_loss, _, _ = base_nnet.encode_everything(indices, **_batch)

        _z = _z.detach()
        _context = _context.detach()

        # 只有 denoiser 这条路径走 DP
        loss_mse, loss_round, T_loss = utils.LSimple(
            _z, dp_model, _schedule, context=_context, **_batch
        )

        _metrics['loss_mse'] = loss_mse.detach().mean()
        _metrics['loss_c'] = neigh_loss.detach().mean()
        _metrics['loss_rounding'] = loss_round.detach().mean()
        _metrics['T_loss'] = T_loss.detach().mean()

        # DP baseline 先只反传这一项
        total_loss = loss_mse.mean()
        # total_loss = loss_mse.mean() + neigh_loss.mean() + loss_round.mean() + T_loss.mean() if epoch > config['warm'] else loss_mse.mean() + loss_round.mean() + T_loss.mean()

        total_loss.backward()

        if use_dp and train_state.step == 0:
            target = dp_model._module if hasattr(dp_model, "_module") else dp_model
            missing_grad_sample = []
            for name, p in target.named_parameters():
                if p.requires_grad and getattr(p, "grad_sample", None) is None:
                    missing_grad_sample.append(name)
            if missing_grad_sample:
                raise RuntimeError(f"Missing grad_sample for parameters: {missing_grad_sample}")

        optimizer.step()
        lr_scheduler.step()
        train_state.step += 1

        return _z, dict(lr=optimizer.param_groups[0]['lr'], **_metrics)

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
        # return decode(_z, nnet.get_logits)
        return decode(_z, train_state.nnet.get_logits)


    def eval_step(n_samples, sample_steps):
        """这里需要做case study"""
        # 逐步采样。这里用上了DPM加速
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=dpm_solver, '
                     f'mini_batch_size={config["sample"]["mini_batch_size"]}')

        def sample_fn(_n_samples):
            indices, batch = next(test_generator)

            assert indices.size(0) == _n_samples
            indices, sample_input = convert_input((indices, batch))
            # _z, _context_emb, _, labels, hard_labels = nnet.encode_everything_test(indices, **batch)
            _z, _context_emb, _, labels, hard_labels = train_state.nnet.encode_everything_test(indices, **batch)
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

    def save_history_csv():
        work_dir = os.path.join(config['workdir'], data_n, task)
        os.makedirs(work_dir, exist_ok=True)

        step_df = pd.DataFrame(step_history)
        step_df.to_csv(
            os.path.join(work_dir, f'{data_n}-{task}-{exp_num}-step_history.csv'),
            index=False
        )

        epoch_df = pd.DataFrame(epoch_history)
        epoch_df.to_csv(
            os.path.join(work_dir, f'{data_n}-{task}-{exp_num}-epoch_history.csv'),
            index=False
        )

    def plot_history():
        import matplotlib.pyplot as plt
        work_dir = os.path.join(config['workdir'], data_n, task)
        os.makedirs(work_dir, exist_ok=True)

        def clean_xy(x, y):
            xy = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
            if len(xy) == 0:
                return [], []
            xx, yy = zip(*xy)
            return list(xx), list(yy)

        # ===== 图1：step losses =====
        plt.figure(figsize=(10, 6))
        for key in ['loss_mse', 'loss_c', 'loss_rounding', 'T_loss', 'total_loss']:
            x, y = clean_xy(step_history['step'], step_history[key])
            if len(x) > 0:
                plt.plot(x, y, label=key)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, f'{data_n}-{task}-{exp_num}-step_losses.png'), dpi=150)
        plt.close()

        # ===== 图2：lr =====
        plt.figure(figsize=(10, 6))
        x, y = clean_xy(step_history['step'], step_history['lr'])
        if len(x) > 0:
            plt.plot(x, y, label='lr')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, f'{data_n}-{task}-{exp_num}-lr.png'), dpi=150)
        plt.close()

        # ===== 图3：epoch mse =====
        plt.figure(figsize=(10, 6))
        for key in ['quality_mse', 'train_mse']:
            x, y = clean_xy(epoch_history['epoch'], epoch_history[key])
            if len(x) > 0:
                plt.plot(x, y, label=key)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, f'{data_n}-{task}-{exp_num}-epoch_mse.png'), dpi=150)
        plt.close()

        # ===== 图4：epoch jaccard =====
        plt.figure(figsize=(10, 6))
        for key in [
            'train_jaccard_conditions',
            'train_jaccard_procedures',
            'train_jaccard_drugs',
            'train_jaccard_mean',
            'quality_jaccard_conditions',
            'quality_jaccard_procedures',
            'quality_jaccard_drugs',
        ]:
            x, y = clean_xy(epoch_history['epoch'], epoch_history[key])
            if len(x) > 0:
                plt.plot(x, y, label=key)
        plt.xlabel('Epoch')
        plt.ylabel('Jaccard')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, f'{data_n}-{task}-{exp_num}-epoch_jaccard.png'), dpi=150)
        plt.close()

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
    best_quality_so_far = float("inf")
    best_epoch = None
    best_ckpt_dir = None
    # while train_state.step < config['train']['n_steps']:
    # max_n_steps = (len(train_dataset) // config['train']['batch_size']) + 1 if len(train_dataset) % config['train']['batch_size'] !=0 else len(train_dataset) // config['train']['batch_size'] # 不对，如果正好整除的话，就不需要+1
    max_n_steps = len(train_dataset_loader)
    for epoch in range(config['train']['n_epochs']):
        if (not use_dp) and epoch == 0:
            train_state.nnet.e_step()
        # if epoch == 0:
        #     # nnet.e_step()
        #     _unwrap_model(nnet).e_step()

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

                # ===== save step metrics =====
                step_history['step'].append(int(train_state.step))
                step_history['loss_mse'].append(float(metrics['loss_mse']))
                step_history['loss_c'].append(float(metrics['loss_c']))
                step_history['loss_rounding'].append(float(metrics['loss_rounding']))
                step_history['T_loss'].append(float(metrics['T_loss']))
                # total_loss 在当前 DP 分支 metrics
                if 'total_loss' in metrics:
                    step_history['total_loss'].append(float(metrics['total_loss']))
                else:
                    total_loss_val = (
                            float(metrics['loss_mse']) + float(metrics['loss_rounding']) + float(metrics['T_loss'])
                    )
                    if epoch > config['warm']:
                        total_loss_val += float(metrics['loss_c'])
                    step_history['total_loss'].append(total_loss_val)
                step_history['lr'].append(float(metrics['lr']))
                # =============================

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
                    if not use_dp:
                        logging.info('Save a grid of {} samples for training...'.format(config['train_val_num']))
                        sample_input = dataset.get_contexts(num=config['train_val_num'])
                        indices = torch.arange(config['train_val_num']).to(device)
                        indices, sample_input = convert_input((indices, sample_input))
                        _z, contexts, _, labels, hard_labels = nnet.encode_everything_test(indices, **sample_input)
                        # contexts = torch.tensor(dataset.contexts, device=device)[: 2 * 5] # 选取前10个context
                        samples, hard_samples = dpm_solver_sample(_n_samples=config['train_val_num'], _sample_steps=50,
                                                                  context=contexts, z_init=_z,
                                                                  mask=sample_input['mask'])
                        # print("generate", hard_samples['conditions'][0])
                        # print("origin", tokenizers['conditions'].batch_decode_2d(hard_labels['conditions'])[0]) # 没啥问题，encode, decode用同一套
                        # print("generate", hard_samples['procedures'][0])
                        # print("origin", tokenizers['procedures'].batch_decode_2d(hard_labels['procedures'])[0])  # 没啥问题，encode, decode用同一套
                        # print("generate", hard_samples['drugs'][0])
                        # print("origin", tokenizers['drugs'].batch_decode_2d(hard_labels['drugs'])[0])  # 没啥问题，encode, decode用同一套
                        # samples = make_grid(dataset.unpreprocess(samples), 5)
                        # save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
                        # wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
                        jaccard_condition = utils.calculate_average_jaccard(hard_samples['conditions'],
                                                                            tokenizers['conditions'].batch_decode_2d(
                                                                                hard_labels['conditions']))
                        jaccard_procedure = utils.calculate_average_jaccard(hard_samples['procedures'],
                                                                            tokenizers['procedures'].batch_decode_2d(
                                                                                hard_labels['procedures']))
                        jaccard_drug = utils.calculate_average_jaccard(hard_samples['drugs'],
                                                                       tokenizers['drugs'].batch_decode_2d(
                                                                           hard_labels['drugs']))
                        mse_loss = F.mse_loss(samples, labels.squeeze())
                        logging.info({'mask num': sample_input['mask'].sum(dim=0), 'val samples mse': mse_loss,
                                      'jaccard': (jaccard_condition, jaccard_procedure, jaccard_drug)})
                        # wandb.log({'samples': mse_loss }, step=train_state.step)
                        # torch.cuda.empty_cache()
                    # accelerator.wait_for_everyone()

                    # save ckpt， 这一块代码检查当前进程是否为主进程（accelerator.is_main_process），如果是则保存模型
                    # if train_state.step % config['train']['save_interval'] == 0 or train_state.step == config['train']['n_steps']:
                    # torch.cuda.empty_cache()

        ###################
        if not use_dp:
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
        epoch_history['epoch'].append(int(epoch))
        epoch_history['train_mse'].append(float(train_mse_ep))
        epoch_history['train_jaccard_conditions'].append(float(train_jacc_ep[0]))
        epoch_history['train_jaccard_procedures'].append(float(train_jacc_ep[1]))
        epoch_history['train_jaccard_drugs'].append(float(train_jacc_ep[2]))
        epoch_history['train_jaccard_mean'].append(
            float((train_jacc_ep[0] + train_jacc_ep[1] + train_jacc_ep[2]) / 3.0)
        )

        # logging.info(f'Save and eval checkpoint {epoch}...')
        # # if accelerator.local_process_index == 0:
        # # train_state.save(os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch}.ckpt'))
        # train_state.save(os.path.join(ckpt_dir, f'{epoch}.ckpt'))
        # # accelerator.wait_for_everyone()
        # if epoch==0 or (epoch>=5 and epoch % config['train']['eval_epoch'] == 0):
        #     # quality, jaccard_tuple = eval_step(n_samples=len(test_dataset), sample_steps=50)  # calculate fid of the saved checkpoint
        #     quality, jaccard_tuple = eval_step(n_samples=len(test_dataset), sample_steps=config['sample']['sample_steps'])
        # else:
        #     quality = 10000 # 因为靠mse取最小的mse
        #     jaccard_tuple = (1,1,1)
        # step_quality.append((epoch, quality))
        # jaccard_quality.append((epoch, jaccard_tuple))
        # torch.cuda.empty_cache()
        logging.info(f'Eval checkpoint candidate {epoch}.')
        if epoch >= 0 or (epoch >= 5 and epoch % config['train']['eval_epoch'] == 0):
            quality, jaccard_tuple = eval_step(
                n_samples=len(test_dataset),
                sample_steps=config['sample']['sample_steps']
            )
            epoch_history['quality_mse'].append(float(quality))
            epoch_history['quality_jaccard_conditions'].append(float(jaccard_tuple[0]))
            epoch_history['quality_jaccard_procedures'].append(float(jaccard_tuple[1]))
            epoch_history['quality_jaccard_drugs'].append(float(jaccard_tuple[2]))
        else:
            quality = 10000.0
            jaccard_tuple = (1.0, 1.0, 1.0)
        step_quality.append((epoch, quality))
        jaccard_quality.append((epoch, jaccard_tuple))
        # 只有更优时才保存 ckpt，并删除旧的非 best
        if quality < best_quality_so_far:
            best_quality_so_far = float(quality)
            best_epoch = int(epoch)
            new_best_ckpt_dir = os.path.join(ckpt_dir, f'{epoch}.ckpt')
            if os.path.exists(new_best_ckpt_dir):
                shutil.rmtree(new_best_ckpt_dir)
            logging.info(f'[BEST] save new best checkpoint at epoch={epoch}, quality={quality}')
            train_state.save(new_best_ckpt_dir)
            for path in glob.glob(os.path.join(ckpt_dir, '*.ckpt')):
                if os.path.abspath(path) != os.path.abspath(new_best_ckpt_dir):
                    shutil.rmtree(path)
            best_ckpt_dir = new_best_ckpt_dir

        if not use_dp:
            total_visit_embs = torch.cat(total_visit_embs, dim=0)
            total_indices = torch.cat(total_indices, dim=0)
            total_visit_embs = total_visit_embs[total_indices.argsort()]
            print("CCCCCCCC", total_visit_embs.shape)

            train_state.nnet.update_train_visit_emb(total_visit_embs)
            visit_centroids, _ = train_state.nnet.e_step()
            train_state.nnet.update_centroid_emb(visit_centroids)

        if use_dp and privacy_engine is not None:
            eps = privacy_engine.get_epsilon(float(config['train']['dp_delta']))
            logging.info(
                f"[DP] epoch={epoch} epsilon={eps:.6f} delta={config['train']['dp_delta']}"
            )
            wandb.log({"dp/epsilon": eps}, step=train_state.step)

    save_history_csv()
    try:
        plot_history()
    except Exception as e:
        logging.info(f"[plot_history failed] {e}")

    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_quality: {step_quality}')
    logging.info(f'step_quality: {jaccard_quality}')
    # epoch_best = sorted(step_quality, key=lambda x: x[1])[0][0] # 最小那个mse
    # logging.info(f'step_best: {epoch_best}')
    # # train_state.load(os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch_best}.ckpt'))
    # # best_ckpt_dir = os.path.join(ckpt_dir, f'{epoch_best}.ckpt')
    # # train_state.load(best_ckpt_dir)
    # # del metrics
    # # # accelerator.wait_for_everyone()
    # # eval_step(n_samples=len(test_dataset), sample_steps=config['sample']['sample_steps'])  # 选取最好的ckpt
    # #
    # # # nnet_path = os.path.join(config['ckpt_root'] + data_n + '/' + task, f'{epoch_best}.ckpt/nnet.pth')
    # # nnet_path = os.path.join(best_ckpt_dir, 'nnet.pth')
    # # return nnet_path, train_state.nnet, train_state.nnet.total_visit_emb.shape[0]
    # best_ckpt_dir = os.path.join(ckpt_dir, f'{epoch_best}.ckpt')
    # # 删除不是 best 编号的 ckpt
    # for path in glob.glob(os.path.join(ckpt_dir, '*.ckpt')):
    #     if os.path.abspath(path) != os.path.abspath(best_ckpt_dir):
    #         shutil.rmtree(path)
    # # 加载 best ckpt
    # train_state.load(best_ckpt_dir)
    # del metrics
    # eval_step(
    #     n_samples=len(test_dataset),
    #     sample_steps=config['sample']['sample_steps']
    # )
    #
    # nnet_path = os.path.join(best_ckpt_dir, 'nnet.pth')
    # return nnet_path, train_state.nnet, train_state.nnet.total_visit_emb.shape[0]
    if best_epoch is None or best_ckpt_dir is None:
        raise RuntimeError("No best checkpoint was saved during training.")
    logging.info(f'step_best: {best_epoch}')
    best_quality = dict(step_quality)[best_epoch]
    best_jacc = dict(jaccard_quality)[best_epoch]
    if use_dp and privacy_engine is not None:
        final_epsilon = float(privacy_engine.get_epsilon(float(config['train']['dp_delta'])))
    else:
        final_epsilon = float("nan")
    train_state.load(best_ckpt_dir)
    del metrics
    eval_step(
        n_samples=len(test_dataset),
        sample_steps=config['sample']['sample_steps']
    )
    nnet_path = os.path.join(best_ckpt_dir, 'nnet.pth')
    generation_csv = os.path.join(
        config['workdir'],
        data_n,
        task,
        "generation_dp_summary.csv"
    )
    append_row_to_csv(
        generation_csv,
        {
            "exp_num": str(exp_num),
            "target_epsilon": float(config['train']['dp_target_epsilon']),
            "actual_epsilon": final_epsilon,
            "best_epoch": int(best_epoch),
            "best_quality_mse": float(best_quality),
            "best_jaccard_conditions": float(best_jacc[0]),
            "best_jaccard_procedures": float(best_jacc[1]),
            "best_jaccard_drugs": float(best_jacc[2]),
            "best_jaccard_mean": float(np.mean(best_jacc)),
            "ckpt_dir": ckpt_dir,
            "best_ckpt_dir": best_ckpt_dir,
            "nnet_path": nnet_path,
        }
    )

    return nnet_path, train_state.nnet, train_state.nnet.total_visit_emb.shape[0]

