from datetime import datetime
import time
import logging
import os
from .cli import parse_dict_args

class RunContext:
    """Creates directories and files for the run"""

    def __init__(self, runner_file, parameters_dict, run_idx=None, log=True, tensorboard=False):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        dataset = runner_file.split('/')[1]
        network = runner_file.split('/')[2]
        runner_name = os.path.basename(runner_file).split(".")[0]
        self.result_dir = "{root}/{dataset}/{network}/{runner_name}/{date:%Y-%m-%d_%H:%M:%S}".format(
            root='results',
            dataset=dataset,
            network=network,
            runner_name=runner_name,
            date=datetime.now(),
        )
        self.transient_dir = self.result_dir + "/transient" 
        if log:  
            os.makedirs(self.result_dir)
            os.makedirs(self.transient_dir)

        self._init_log(log)
        self.args = parse_dict_args(**parameters_dict)
        self._init_env()
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.vis_log = SummaryWriter(self.result_dir + "/TB_log")
        else:
            self.vis_log = None

    def _init_env(self):
        os.environ['OMP_NUM_THREADS'] = str(self.args.omp_threads)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpus

    def _init_log(self, log):
        LOG = logging.getLogger('main')
        FileHandler = logging.FileHandler(os.path.join(self.result_dir, 'log.txt'))
        LOG.addHandler(FileHandler)

