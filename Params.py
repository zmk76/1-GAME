import argparse

from mig_gt.configs.default_config import add_arguments_by_config_class
from mig_gt.configs.mm_mgdcf_default_config import MMMGDCFConfig

config_class = MMMGDCFConfig
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='baby', help='name of datasets')
parser.add_argument('--method', type=str)
parser.add_argument('--result_dir', type=str)
parser.add_argument('--seed', default=2026, type=int)
parser.add_argument('--gpu', type=str)
parser.add_argument('--temp', type=float, default=0.5)
parser.add_argument('--ssl_reg', default=1e-2, type=float, help='weight for contrative learning')
parser.add_argument('--mf_lr', default=5e-3, type=float, help='meanflows learning rate')   #default=1e-2 -> default=5e-3



#修改点--新增
parser.add_argument('--flow_ratio', default=0.5, type=float, help='meanflows flow_ratio (r=t sampling ratio)')# ← 加这一行
parser.add_argument('--clip_eps', type=float, default=0.2, help='PPO clip epsilon')
parser.add_argument('--walk_steps', default=7, type=int, help='random walk steps (S) for negative sampling')

parser.add_argument('--meanflows_weight', type=float, default=1.0,help='weight for meanflows loss (ELBO + cl)')
parser.add_argument('--unsmooth_weight', type=float, default=1.0,help='weight for unsmooth loss (λ3 in Eq.14)')
############




parser = add_arguments_by_config_class(parser, config_class)
p_args = parser.parse_args()
