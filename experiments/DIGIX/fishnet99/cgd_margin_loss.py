import os
import sys
import logging

import torch

sys.path.append(".")
from src.train_retrieval import cgd_margin_loss as main
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
    'resize_size': 800,
    'input_size': 736,
    
    # Data sampling
    'batch_size': 64,
    'eval_batch_size': 64,
    
    # Architecture
    'arch': 'fishnet99',
    
    'gd_config': 'SMG',
    'feature_dim': 6144,
    
    'easy_margin': False,
    'metric': 'cos_margin',
    'margin': 0.4,
    'feat_norm': 30,
    
    
    # Optimization
    'lr_rampup': 0,
    'lr': 0.0001,
    'epochs': 40,
    'lr_rampdown_epochs': 40,
    
    'pretrained': 'https://localhost/fishnet99_ckpt.tar'
}

if __name__ == "__main__":
    context = RunContext(__file__, parameters)
    main.main(context)

    

