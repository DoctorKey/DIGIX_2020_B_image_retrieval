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
    'omp_threads': 8,


    'workers': 16,
    'checkpoint_epochs': 20,
    'print-freq': 50,

    # Data
    'dataset': 'DIGIX',
    'resize_size': 1024,
    'input_size': 896,

    # Data sampling
    'batch_size': 16,
    'eval_batch_size': 16,

    # Architecture
    'arch': 'dla102x',

    'gd_config': 'SMG',
    'feature_dim': 6144,

    'easy_margin': False,
    'metric': 'cos_margin',
    'margin': 0.4,
    'feat_norm': 30,

    'pretrained': 'pretrained/dla102x_896_5088.rm_fc.fp16.ckpt',

    'tencrops': True,
}


if __name__ == "__main__":
    context = RunContext(__file__, parameters)
    main.main(context)
    
