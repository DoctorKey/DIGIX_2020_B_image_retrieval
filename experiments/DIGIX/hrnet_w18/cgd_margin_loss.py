import os
import sys
import logging

sys.path.append(".")
from src.train_retrieval import cgd_margin_loss as main
from src.cli import parse_dict_args
from src.run_context import RunContext

parameters = {
    'gpus': '0,1,2,3,4,5,6,7',
    'omp_threads': 4,
    
    'workers': 8,
    'checkpoint_epochs': 20,
    'print-freq': 50,
    
    # Data
    'dataset': 'DIGIX',
    'resize_size': 800,
    'input_size': 736,
    
    # Data sampling
    'batch_size': 48,
    'eval_batch_size': 48,
    
    # Architecture
    'arch': 'hrnet_w18',
    
    'gd_config': 'SMG',
    'feature_dim': 6144,

    'easy_margin': False,
    'metric': 'cos_margin',
    'margin': 0.4,
    'feat_norm': 30,
    
    
    # Optimization
    'lr_rampup': 0,
    'lr': 5e-4,
    'epochs': 40,
    'lr_rampdown_epochs': 40,
    
    'pretrained': "https://localhost/hrnetv2_w18_imagenet_pretrained.pth"
}

if __name__ == "__main__":
    context = RunContext(__file__, parameters)
    main.main(context)


