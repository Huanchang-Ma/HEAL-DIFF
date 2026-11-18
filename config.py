# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : config.py
# Time       ：4/11/2024 3:27 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：nohup python main_unify.py > tes_ours_miv_phe.log 2>&1 &
"""

MIII_PARAMS = {
    'FEATURE' : ['conditions', 'procedures', 'drugs'],
    'SEED': 528,
    'USE_CUDA': True,
    # 'GPU': '3',
    'GPU': '0',
    'EPOCH': 50, # 20 for rebuttal
    'WARM_EPOCH': 50,
    'DIM': 128,
    'HIDDEN': 128,
    'LR': 1e-3,  # PHE/LOS
    'BATCH': 32,
    'DROPOUT': 0.1,
    'WD': 0.,#1e-3,
    # 'WD': 1e-3,#1e-3,
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10,
    'MAXCODESEQ': 512,
    'AUX' : 0.00001,

}

eICU_PARAMS = {
    # 'FEATURE' : ['conditions', 'drugs', 'physicalExam', 'treatment', 'admissionDx'],
    'FEATURE': ['conditions', 'procedures', 'drugs'],
    'SEED': 528,
    'USE_CUDA': True,
    'GPU': '0',
    'EPOCH': 50, # 20 for rebuttal
    'WARM_EPOCH': 50,
    'DIM': 128,
    'HIDDEN': 128,
    # 'LR': 1e-3, #PHE
    'LR': 1e-4,  # LOS
    'BATCH': 32,
    'DROPOUT': 0.1,
    'WD': 0.,#1e-3,
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10,
    'MAXCODESEQ': 512,
    'AUX' : 0.00001,

}

MIV_Note_PARAMS = {
    # 'FEATURE' : ['conditions', 'procedures', 'drugs', 'note'],
    'FEATURE': ['conditions', 'procedures', 'drugs'],
    'SEED': 528,
    'USE_CUDA': True,
    'GPU': '0',
    'EPOCH': 50, # 20 for rebuttal
    'WARM_EPOCH': 50,
    'DIM': 128,
    'HIDDEN': 128,
    #'LR': 1e-3,
    'LR': 1e-4,
    # 'BATCH': 32,
    'BATCH': 8,
    'DROPOUT': 0.1,
    'WD': 0.,#1e-3,
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10,
    'MAXCODESEQ': 512,
    'AUX' : 0.00001,

}


class PHECONFIG(): # 不要有drugs
    """DRL config"""
    # data_info parameter
    DEV = False
    MODEL = "ours"
    PLM = "Clinical-BERT"
    #TASK = 'PHE'
    TASK = 'LOS'
    #DATASET = 'MIII'
    DATASET = 'eICU'
    #DATASET = 'MIV-Note'
    LABEL = 'labels' 

    ATCLEVEL = 3
    RATIO = 0.6 # train-test split
    THRES = 0.4 # pred threshold
    # train parameter
    ##########
    USE_DP = True
    DP = {
        "target_epsilon": 8.0,
        "target_delta": 1e-5,
        "max_grad_norm": 1.0,
        "accountant": "rdp",
        "poisson_sampling": True,
        "grad_sample_mode": "hooks",
        "clipping": "flat",
    }
    ##########

    DATASET_PARAMS = {
        'MIII': MIII_PARAMS,
        'eICU': eICU_PARAMS,
        'MIV-Note': MIV_Note_PARAMS
        # 'EHR-SHOT': EHR_PARAMS,
        # 'OMOP': OMOP_PARAMS,
    }

    @classmethod
    def get_params(cls):
        return cls.DATASET_PARAMS.get(cls.DATASET, {})

    # log
    # LOGDIR = '/home/czhaobo/MMHealth/log/ckpt/'
    LOGDIR = '/home/mhc/Diffusion_Model/Health_Care/MMHealth-main/log/ckpt/'
    # LOGDIR = '/home/exp1/mhc/MMHealth-main/log/ckpt/'




# config = vars(PHECONFIG)
# config = {k:v for k,v in config.items() if not k.startswith('__')}

config = {**vars(PHECONFIG), **PHECONFIG.get_params()}
config = {k: v for k, v in config.items() if not k.startswith('__')}
