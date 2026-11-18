# # !/usr/bin/env python
# # -*-coding:utf-8 -*-
#
# """
# # File       : diff_config.py
# # Time       ：8/11/2024 5:12 pm
# # Author     ：Chuang Zhao
# # version    ：python
# # Description：
# """
#
# class DIFFCONFIG(): # 不要有drugs
#     """DRL config"""
#     seed = 528
#     device = 6
#     warm = 10
#     benchmark = False
#
#     ckpt_root = '/home/exp1/mhc/MMHealth-main/log/difflog/ckpt/'
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt1/'
#     sample_dir = '/home/exp1/mhc/MMHealth-main/log/difflog/sample/'
#     workdir = '/home/exp1/mhc/MMHealth-main/log/difflog/work_dir/' # wandb
#     output_path = '/home/exp1/mhc/MMHealth-main/log/difflog/work_dir/output.log'
#     z_shape = (3, 128)
#     train_val_num = 50
#     dim=128
#     threshod = 0.4 # jaccard
#
#
#     dataset = {
#         'MIII':{
#             'name': 'MIII',
#             'feature_keys':  ['conditions', 'procedures', 'drugs'],
#             'cfg' : True,
#             'p_uncond' : 0.1,
#         },
#         'MIV-Note': {
#             'name': 'MIV-Note',
#             'feature_keys': ['conditions', 'procedures', 'drugs'],
#             'cfg': True,
#             'p_uncond': 0.1,
#         },
#         'eICU': {
#             'name': 'eICU',
#             'feature_keys': ['conditions', 'procedures', 'drugs'],
#             'cfg': True,
#             'p_uncond': 0.1,
#         }
#     }
#     sample = {
#         'mini_batch_size': 16, # 训练的时候搞大点似乎也行； 但是eCIU不行，数量太少了，一次不够；
#         'scale': 1,
#         'path': '/home/exp1/mhc/MMHealth-main/log/difflog/eval_sample/',
#         'sample_steps': 20, # dpm solver用更少的步
#         # 'sample_steps': 50,  # dpm solver用更少的步
#         'n_samples': 10000,
#     }
#     hparams = 'lr' # monitor hyper
#
#     nnet = {
#         'name':'dit',
#         'visit_config':  {
#             'dim': 128,
#             'num_heads': 2,
#             'num_ids':2,
#             'mode': 'sum',
#             'logits_mode':'linear'
#         },
#         'con_config':{
#             'dim': 128,
#             'mode': 'concat',
#             'num_heads': 2,
#             'hist_mode': 'individual', # joint
#         },
#         'cluster_config':{
#             'dim': 128,
#             'k': 10, # 5， 10， 20，100
#             'proto_reg': 1e-4,
#             'ssl_temp': 0.1,
#             'mode': 'k-means',
#         },
#         # 'img_size': 32,
#         'in_chans' : 4,
#         # 'patch_size' :  2,
#         'embed_dim' :  128,
#         'depth' :  2, # 12
#         'num_heads' :  4,
#         'mlp_ratio' :  4,
#         'qkv_bias' :  False,
#         'mlp_time_embed' :  False,
#         'contx_dim' : 128,
#         # 'num_contx_token': 9  # MIII eICU
#         'num_contx_token' :  9 # 可能要改, 10 for note
#             }
#
#
#     optimizer = {
#         'name': 'adamw',
#         'lr' : 5e-4, #0.0002,eicu 用5e-3
#         # 'lr': 1e-3,
#         'weight_decay' : 0,#0.03,
#         # 'weight_decay': 0.03,  # 0.03,
#         'betas' : (0.9, 0.9),
#     }
#
#     lr_scheduler = {
#         'name': 'customized',
#         'warmup_steps':5000
#     }
#
#     train = {
#         'n_epochs': 50, # step/ epcoh
#         'batch_size': 16,
#         'log_interval': 10000, # 就直接存
#         'eval_epoch': 5, # 3 eoch eval一次, 比max小1个
#         # 'save_interval': 1000,
#         # 'n_steps': 5000 # 这里的值 wrong
#     }
#
# config = vars(DIFFCONFIG)
# config = {k:v for k,v in config.items() if not k.startswith('__')}




# ## DiffmvDP04
# # !/usr/bin/env python
# # -*-coding:utf-8 -*-
# """
# # File       : diff_config.py
# # Time       ：8/11/2024 5:12 pm
# # Author     ：Chuang Zhao
# # version    ：python
# # Description：
# """
#
#
# class DIFFCONFIG():  # 不要有drugs
#     """DRL config"""
#     seed = 528
#     device = 6
#     warm = 10
#     benchmark = False
#
#     # ckpt_root = '/home/exp1/mhc/MMHealth-main/log/difflog/ckpt2/'  ## 2
#     # ckpt_root = '/home/exp1/mhc/MMHealth-main/log/difflog/ckpt4/'  ## 4
#     # ckpt_root = '/home/exp1/mhc/MMHealth-main/log/difflog/ckpt7/'  ## 7
#     # ckpt_root = '/home/exp1/mhc/MMHealth-main/log/difflog/ckpt9/'  ## 9
#     # ckpt_root = '/home/exp1/mhc/MMHealth-main/log/difflog/ckpt/'   ## 6
#     # ckpt_root = '/home/exp1/mhc/MMHealth-main/log/difflog/ckpt1/'  ## 当前训练好的
#     # sample_dir = '/home/exp1/mhc/MMHealth-main/log/difflog/sample/'
#     # workdir = '/home/exp1/mhc/MMHealth-main/log/difflog/work_dir/'  # wandb
#     # output_path = '/home/exp1/mhc/MMHealth-main/log/difflog/work_dir/output.log'
#
#     ## 4090
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2/'  ## 2
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt4/'  ## 4
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt7/'  ## 7
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt9/'  ## 9
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt6/'  ## 6
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt1/'  ## 当前训练好的
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2603/'
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2603DiffmvDP/'
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2603-2/'  ## main_unify_DiffmvDP
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2603-4/'
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2603-6/'
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2603-7/'
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2603-9/'
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2603/' ## loss
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2604/'  ## loss
#     ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260407/'  ## loss
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260407_DP/'  ## loss
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260407_nonDP/'  ## loss
#
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260331-2/' ## main_unify_DP
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260331-4/'
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260331-6/'
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260331-7/'
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260331-9/'
#     # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260331/' ## loss
#     sample_dir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/sample/'
#     workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir/'  # wandb
#     output_path = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir/output.log'
#     z_shape = (3, 128)
#     train_val_num = 50
#     dim = 128
#     threshod = 0.4  # jaccard
#     # threshod = 0.2  # jaccard
#
#     dataset = {
#         'MIII': {
#             'name': 'MIII',
#             'feature_keys': ['conditions', 'procedures', 'drugs'],
#             'cfg': True,
#             'p_uncond': 0.1,
#         },
#         'MIV-Note': {
#             'name': 'MIV-Note',
#             'feature_keys': ['conditions', 'procedures', 'drugs'],
#             'cfg': True,
#             'p_uncond': 0.1,
#         },
#         'eICU': {
#             'name': 'eICU',
#             'feature_keys': ['conditions', 'procedures', 'drugs'],
#             'cfg': True,
#             'p_uncond': 0.1,
#         }
#     }
#     sample = {
#         # 'mini_batch_size': 16,  # 训练的时候搞大点似乎也行； 但是eCIU不行，数量太少了，一次不够；
#         'mini_batch_size': 32,
#         # 'scale': 1,
#         'scale': 0.5, ## 论文设置
#         'path': '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/eval_sample/',
#         # 'path': '/home/exp1/mhc/MMHealth-main/log/difflog/eval_sample/',
#         # 'sample_steps': 20,  # dpm solver用更少的步 (原先步数)
#         'sample_steps': 5,  # dpm solver用更少的步
#         # 'sample_steps': 15,  # dpm solver用更少的步
#         # 'sample_steps': 50,  # dpm solver用更少的步
#         'n_samples': 10000,
#
#         ############
#         'best_of_k': 4,
#         'keep_ratio': 0.35,
#         'use_latent_score': True,
#         'balance_by_label': False,
#
#         'target_proc_len': 8.0,
#         'target_drug_len': 8.0,
#
#         'decode_thresholds': {
#             'conditions': 0.40,
#             'procedures': 0.40,
#             'drugs': 0.40,
#         },
#         # 'decode_thresholds': {
#         #     'conditions': 0.30,
#         #     'procedures': 0.30,
#         #     'drugs': 0.30,
#         # },
#
#         'decode_topk': {
#             'conditions': None,
#             'procedures': None,
#             'drugs': None,
#         }
#     }
#     hparams = 'lr'  # monitor hyper
#
#     nnet = {
#         'name': 'dit',
#         'visit_config': {
#             'dim': 128,
#             'num_heads': 2,
#             'num_ids': 2,
#             'mode': 'sum',
#             'logits_mode': 'linear'
#         },
#         'con_config': {
#             'dim': 128,
#             'mode': 'concat',
#             'num_heads': 2,
#             'hist_mode': 'individual',  # joint
#         },
#         'cluster_config': {
#             'dim': 128,
#             'k': 10,  # 5， 10， 20，100
#             # 'proto_reg': 1e-4,
#             'proto_reg': 1e-3, # 论文设置
#             'ssl_temp': 0.1,
#             'mode': 'k-means',
#         },
#         # 'img_size': 32,
#         'in_chans': 4,
#         # 'patch_size' :  2,
#         'embed_dim': 128,
#         'depth': 2,  # 12
#         'num_heads': 4,
#         'mlp_ratio': 4,
#         'qkv_bias': False,
#         'mlp_time_embed': False,
#         'contx_dim': 128,
#         # 'num_contx_token': 9  # MIII eICU
#         'num_contx_token': 9  # 可能要改, 10 for note
#     }
#
#     optimizer = {
#         'name': 'adamw',
#         # 'lr': 5e-4,  # 0.0002,eicu 用5e-3
#         'lr': 1e-3,
#         # 'lr': 0.1,
#         # 'lr': 0.01,
#         # 'lr': 0.05,
#         # 'lr': 5e-5,
#         'weight_decay': 5e-4,  # 0.03,
#         # 'weight_decay': 0.03,
#         'betas': (0.9, 0.9),
#     }
#
#     lr_scheduler = {
#         'name': 'customized',
#         'warmup_steps': 5000
#     }
#
#     train = {
#         # 'n_epochs': 200,  # step/ epcoh
#         'batch_size': 32,
#         # 'batch_size': 64,
#         # 'batch_size': 128,
#         'n_epochs': 50,  # step/ epcoh
#         # 'n_epochs': 200,  # step/ epcoh
#         # 'batch_size': 16,
#         # 'log_interval': 10000,  # 就直接存 原来
#         # 'log_interval': 200,
#         'log_interval': 20,
#         'eval_epoch': 1,  # 3 eoch eval一次, 比max小1个   ## if epoch==0 or (epoch>=5 and epoch % config['train']['eval_epoch'] == 0)
#         # 'save_interval': 1000,
#         # 'n_steps': 5000 # 这里的值 wrong
#
#         # ===== DP-SGD =====
#         'dp_sgd': True,
#         # 'dp_sgd': False,
#         # 'dp_target_epsilon': 8.0,
#         # 'dp_target_epsilon': 64,
#         'dp_target_epsilon': 32,
#         # 'dp_target_epsilon': 0.1,  # [0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 4.0, 8.0, 16.0]
#         'dp_delta': 1e-5,  # 允许有 δ 的极小概率不严格成立
#         # 'dp_delta': 0.1,  # 1e-3
#         # 'dp_max_grad_norm': 2.0,
#         'dp_max_grad_norm': 0.5, ## 较好结果
#         # 'dp_max_grad_norm': 0.2,
#         # 'dp_max_grad_norm': 1.0,
#         'dp_secure_mode': False,
#         'dp_grad_sample_mode': 'functorch',
#         # 'dp_grad_sample_mode': 'hooks',
#         'dp_poisson_sampling': True,
#         # 'dp_physical_batch_size': 16,
#         'dp_accountant': 'prv',
#         'dp_clipping': 'flat',
#         'dp_physical_batch_size': 16,
#         # 'use_freeze_inactive': False,  ## 是否冻结
#         'use_full_centroid_refresh': False,
#         'use_pos_adapter': True,
#
#     }
#
#
#
#
# config = vars(DIFFCONFIG)
# config = {k: v for k, v in config.items() if not k.startswith('__')}


## DiffmvDP04
# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# File       : diff_config.py
# Time       ：8/11/2024 5:12 pm
# Author     ：Chuang Zhao
# version    ：python
# Description：
"""


class DIFFCONFIG():  # 不要有drugs
    """DRL config"""
    seed = 528
    device = 6
    warm = 10
    benchmark = False

    # ckpt_root = '/home/exp1/mhc/MMHealth-main/log/difflog/ckpt2/'  ## 2
    # ckpt_root = '/home/exp1/mhc/MMHealth-main/log/difflog/ckpt4/'  ## 4
    # ckpt_root = '/home/exp1/mhc/MMHealth-main/log/difflog/ckpt7/'  ## 7
    # ckpt_root = '/home/exp1/mhc/MMHealth-main/log/difflog/ckpt9/'  ## 9
    # ckpt_root = '/home/exp1/mhc/MMHealth-main/log/difflog/ckpt/'   ## 6
    # ckpt_root = '/home/exp1/mhc/MMHealth-main/log/difflog/ckpt1/'  ## 当前训练好的
    # sample_dir = '/home/exp1/mhc/MMHealth-main/log/difflog/sample/'
    # workdir = '/home/exp1/mhc/MMHealth-main/log/difflog/work_dir/'  # wandb
    # output_path = '/home/exp1/mhc/MMHealth-main/log/difflog/work_dir/output.log'

    ## 4090
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2/'  ## 2
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt4/'  ## 4
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt7/'  ## 7
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt9/'  ## 9
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt6/'  ## 6
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt1/'  ## 当前训练好的
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2603/'
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2603DiffmvDP/'
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2603-2/'  ## main_unify_DiffmvDP
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2603-4/'
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2603-6/'
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2603-7/'
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2603-9/'
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2603/' ## loss
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt2604/'  ## loss
    ##ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260407/'  ## loss
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_MIII_PHE/'  ## MIII原版PHE    
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_MIII_PHE_pmi/'  ## MIII PMI PHE
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_MIII_PHE_pmi0.085/'  ## MIII PMI PHE2
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_MIII_PHE_pmi0.02/'  ## MIII PMI PHE2
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_eICU_PHE/'  ## eICU原版PHE
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_eICU_PHE_pmi/'  ## eICU PMI PHE    
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_MIV_PHE11/'  ## MIV原版PHE    
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_MIV_PHE_pmi/'  ## MIV PMI PHE
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_MIV_PHE_subset/'  ## MIV PHE suset
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_MIV_PHE_subset_pmi/'
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_MIV_LOS_subset_pmi/'
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_MIV_LOS_subset/'
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_MIII_LOS/'  ## MIII原版LOS  
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_MIII_LOS_pmi/'  ## MIII PMI LOS
    ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_eICU_LOS111/'  ## eICU原版LOS
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_eICU_LOS_pmi0.1/'  ## eICU PMI LOS    
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_MIV_LOS11/'  ## MIV原版LOS
    #ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt_3split_MIV_LOS_pmi/'  ## MIV PMI LOS
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260407_DP/'  ## loss
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260407_nonDP/'  ## loss

    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260331-2/' ## main_unify_DP
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260331-4/'
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260331-6/'
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260331-7/'
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260331-9/'
    # ckpt_root = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/ckpt260331/' ## loss
    sample_dir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/sample/'
    #sample_dir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/sample_3split_MIV_PHE/'
    #sample_dir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/sample_3split_MIV_PHE_pmi/'
    #sample_dir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/sample_3split_MIII_LOS/'
    #sample_dir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/sample_3split_MIII_LOS_pmi/'
    #sample_dir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/sample_3split_eICU_LOS_pmi/'
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir/'  # wandb
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_MIII_PHE/'
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_MIII_PHE_pmi/'
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_MIII_PHE_pmi0.02/'
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_MIII_PHE_pmi0.085/'    
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_eICU_PHE/'
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_eICU_PHE_pmi/'
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_MIV_PHE11/'
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_MIV_PHE_pmi/'
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_MIV_PHE_subset/'
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_MIV_PHE_subset_pmi/'
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_MIV_LOS_subset_pmi/'
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_MIV_LOS_subset/'
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_MIII_LOS/'
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_MIII_LOS_pmi/'
    workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_eICU_LOS111/'
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_eICU_LOS_pmi0.1/'
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_MIV_LOS11/'
    #workdir = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir_3split_MIV_LOS_pmi/'
    output_path = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir/output.log'
    z_shape = (3, 128)
    train_val_num = 50
    dim = 128
    threshod = 0.4  # jaccard
    # threshod = 0.2  # jaccard

    dataset = {
        'MIII': {
            'name': 'MIII',
            'feature_keys': ['conditions', 'procedures', 'drugs'],
            'cfg': True,
            'p_uncond': 0.1,
        },
        'MIV-Note': {
            'name': 'MIV-Note',
            'feature_keys': ['conditions', 'procedures', 'drugs'],
            'cfg': True,
            'p_uncond': 0.1,
        },
        'eICU': {
            'name': 'eICU',
            'feature_keys': ['conditions', 'procedures', 'drugs'],
            'cfg': True,
            'p_uncond': 0.1,
        }
    }
    sample = {
        # 'mini_batch_size': 16,  # 训练的时候搞大点似乎也行； 但是eCIU不行，数量太少了，一次不够；
        'mini_batch_size': 128,
        # 'scale': 1,
        'scale': 0.5, ## 论文设置
        'path': '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/eval_sample/',
        # 'path': '/home/exp1/mhc/MMHealth-main/log/difflog/eval_sample/',
        'sample_steps': 20,  # dpm solver用更少的步 (原先步数)
        #'sample_steps': 5,  # dpm solver用更少的步
        # 'sample_steps': ‘’15,  # dpm solver用更少的步
        # 'sample_steps': 50,  # dpm solver用更少的步
        'n_samples': 10000,

        ############
        #'best_of_k': 4, ## original
        # 'best_of_k': 1, ## best
        'best_of_k': 5,
        # 'best_of_k': 10,
        'keep_ratio': 0.35, ## original 保留生成质量较高的一部分样本
        # 'keep_ratio': 0.5,
        # 'keep_ratio': 0.1,
        'use_latent_score': True,
        'balance_by_label': False,

        'target_proc_len': 8.0,
        'target_drug_len': 8.0,

        'decode_thresholds': {
            'conditions': 0.40,
            'procedures': 0.40,
            'drugs': 0.40,
        },
        #'decode_thresholds': {
         #   'conditions': 0.35,
          #  'procedures': 0.40,
           # 'drugs': 0.25,
        #},
        # 'decode_thresholds': {
        #     'conditions': 0.30,
        #     'procedures': 0.30,
        #     'drugs': 0.30,
        # },

        'decode_topk': {
            'conditions': None,
            'procedures': None,
            'drugs': None,
        }
    }
    hparams = 'lr'  # monitor hyper

    nnet = {
        'name': 'dit',
        'visit_config': {
            'dim': 128,
            'num_heads': 2,
            'num_ids': 2,
            'mode': 'sum',
            'logits_mode': 'linear'
        },
        'con_config': {
            'dim': 128,
            'mode': 'concat',
            'num_heads': 2,
            'hist_mode': 'individual',  # joint
        },
        'cluster_config': {
            'dim': 128,
            'k': 10,  # 5， 10， 20，100
            # 'proto_reg': 1e-4,
            'proto_reg': 1e-3, # 论文设置
            'ssl_temp': 0.1,
            'mode': 'k-means',
        },
        # 'img_size': 32,
        'in_chans': 4,
        # 'patch_size' :  2,
        'embed_dim': 128,
        'depth': 2,  # 12
        'num_heads': 4,
        'mlp_ratio': 4,
        'qkv_bias': False,
        'mlp_time_embed': False,
        'contx_dim': 128,
        # 'num_contx_token': 9  # MIII eICU
        'num_contx_token': 9  # 可能要改, 10 for note
    }

    optimizer = {
        'name': 'adamw',
        # 'lr': 5e-4,  # 0.0002,eicu 用5e-3
        'lr': 1e-3,
        # 'lr': 0.1,
        # 'lr': 0.01,
        # 'lr': 0.05,
        # 'lr': 5e-5,
        'weight_decay': 5e-4,  # 0.03,
        # 'weight_decay': 0.03,
        'betas': (0.9, 0.9),
    }

    lr_scheduler = {
        'name': 'customized',
        'warmup_steps': 5000
    }

    train = {
        #'n_epochs': 200,  # step/ epcoh
        #'batch_size': 32,
        'batch_size': 64,
        # 'batch_size': 128,
        'n_epochs': 50,  # step/ epcoh original
        #'n_epochs': 200,  # step/ epcoh
        # 'batch_size': 16,
        # 'log_interval': 10000,  # 就直接存 原来
        # 'log_interval': 200,
        'log_interval': 20,
        'eval_epoch': 1,  # 3 eoch eval一次, 比max小1个   ## if epoch==0 or (epoch>=5 and epoch % config['train']['eval_epoch'] == 0)
        # 'save_interval': 1000,
        # 'n_steps': 5000 # 这里的值 wrong

        # ===== DP-SGD =====
        'dp_sgd': True,
        #'dp_sgd': False,
        'dp_mode': 'noise',
        #'dp_target_epsilon': 0.5,
        'dp_target_epsilon': 8.0,
        # 'dp_target_epsilon': 32,  ## 当前较好结果
        # 'dp_target_epsilon': 0.1,  # [0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 4.0, 8.0, 16.0]
        'dp_delta': 1e-5,  # 允许有 δ 的极小概率不严格成立
        # 'dp_delta': 0.1,  # 1e-3
        # 'dp_max_grad_norm': 2.0,
        'dp_max_grad_norm': 0.5, ## 较好结果
        # 'dp_max_grad_norm': 4.0,
        # 'dp_max_grad_norm': 1.0,
        'dp_secure_mode': False,
        'dp_grad_sample_mode': 'functorch',
        # 'dp_grad_sample_mode': 'hooks',
        'dp_poisson_sampling': True,
        'dp_physical_batch_size': 16,
        'dp_accountant': 'prv',
        'dp_clipping': 'flat',  ## original
        # 'dp_clipping': 'per_layer',
        #'dp_physical_batch_size': 32,
        # 'use_freeze_inactive': False,  ## 是否冻结
        'use_full_centroid_refresh': False, # original
        # 'use_full_centroid_refresh': True,
        'use_pos_adapter': True,
        'static_mask_epochs': 0,
        'force_static_mask': False,
        # 'fixed_mask_view': 'conditions',
        # 'fixed_mask_view': 'procedures',
        # 'fixed_mask_view': 'drugs',

        ## Ablation Study
        ## ########## non-DP Clip ##########
        'clip_only_max_grad_norm':0.5,
        ## ########## DP Clip-Only ##########
        # 'dp_noise_multiplier': 0.0
        ## ########## DP Clip + 少量噪声 ##########
        'dp_noise_multiplier': 0.015,
        #'dp_noise_multiplier': 0.085,
        #'dp_noise_multiplier': 0.02,

        ## PMI
        'use_medical_prior': True,
        # 'use_medical_prior': False,
        #'medical_prior_path': "/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/difflog/work_dir/MIII/PHE/medical_prior.pt",
        ##自动生成workdir/{dataset}/{task}/medical_prior.pt
        'prior_alpha': 0.3,
        #'prior_alpha': 0.2,
        'prior_smooth':1.0,
        'prior_topm': 30,
        'model_prior_gate_topm': 50,


    }


config = vars(DIFFCONFIG)
config = {k: v for k, v in config.items() if not k.startswith('__')}


