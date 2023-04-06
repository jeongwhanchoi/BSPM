import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = os.path.dirname(os.path.abspath(__file__ + "/../"))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['gowalla','yelp2018','amazon-book']
all_models  = ['mf', 'lgn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False

config['solver_idl'] = args.solver_idl
config['solver_blr'] = args.solver_blr
config['solver_shr'] = args.solver_shr
config['K_idl'] = args.K_idl
config['K_b'] = args.K_b
config['K_s'] = args.K_s
config['T_idl'] = args.T_idl
config['T_b'] = args.T_b
config['T_s'] = args.T_s

config['factor_dim'] = args.factor_dim
config['idl_beta'] = args.idl_beta

config['final_sharpening'] = args.final_sharpening
config['sharpening_off'] = args.sharpening_off
config['t_point_combination'] = args.t_point_combination

GPU = torch.cuda.is_available()
GPU_NUM = args.gpu
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
config['device'] = device
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

config['dataset'] = args.dataset
dataset = args.dataset
simple_model = args.simple_model
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")




TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██████╗ ███████╗██████╗ ███╗   ███╗
██╔══██╗██╔════╝██╔══██╗████╗ ████║
██████╔╝███████╗██████╔╝██╔████╔██║
██╔══██╗╚════██║██╔═══╝ ██║╚██╔╝██║
██████╔╝███████║██║     ██║ ╚═╝ ██║
╚═════╝ ╚══════╝╚═╝     ╚═╝     ╚═╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=BSPM
print(logo)
