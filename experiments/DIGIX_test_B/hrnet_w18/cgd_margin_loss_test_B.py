import os
import sys
import logging

import torch

sys.path.append(".")
from src.tests import cgd_extract_feature_retrieval as main
from src.cli import parse_dict_args
from src.run_context import RunContext


parameters = {
    # Technical details
    'gpus': '0,1,2,3,4,5,6,7',
    'omp_threads': 4,

    'workers': 8,
    'checkpoint_epochs': 20,
    'print-freq': 50,

    # Data
    'dataset': 'DIGIX_test_B',
    'resize_size': 800,
    'input_size': 736,

    # Data sampling
    'batch_size': 24,
    'eval_batch_size': 24,

    # Architecture
    'arch': 'hrnet_w18',

    'gd_config': 'SMG',
    'feature_dim': 6144,

    'easy_margin': False,
    'metric': 'cos_margin',
    'margin': 0.4,
    'feat_norm': 30,

    'pretrained': 'pretrained/hrnet_w18.rm_fc.fp16.ckpt',

    'tencrops': True,
}


if __name__ == "__main__":
    context = RunContext(__file__, parameters)
    main.main(context)
    
