from __future__ import print_function
import argparse
import time


def str2bool(v):
    return v is True or v.lower() in ('true', '1')


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


parser.add_argument('--name', type=str, default='Debias', help='The main model name')
parser.add_argument('--time', type=str, default='', help='Current time')
# Learning process
parser.add_argument('--epoch_max', type=int, default=10, help='The learning epoches')
parser.add_argument('--iter_save', type=int, default=1, help='The save turn')
parser.add_argument('--pt_load_path', type=str, default='')

# Pretrain network
parser.add_argument('--train_mode', type=str, default='dev', choices=['pretrain', 'dev', 'test'],
                    help='Weighted learning')
parser.add_argument('--use_weight', type=str2bool, default=False, help='Weighted learning')
parser.add_argument('--is_debias', type=str2bool, default=True, help='Using feature balance as representation')

# dev network
parser.add_argument('--model_dir', type=str, default='', help='The model dir')
parser.add_argument('--user_feature_path', type=str, default=None, help="User fature director")
parser.add_argument('--item_feature_path', type=str, default=None, help="Item fature director")
parser.add_argument('--feature_data', type=bool, default=False, help="If the data contains feature")
parser.add_argument('--lambda_impression', type=float, default=0.8, help='The trade-off parameter of lambda of impression loss')
parser.add_argument('--lambda_click', type=float, default=0.8, help='The trade-off parameter of lambda of click loss')
parser.add_argument('--experiment_id', type=int, default=0, help="The id of current experiment")
# Used to be an option, but now is solved
parser.add_argument('--impression_or_click', type=str, default='impression', help="Using impression or click label as adversarial model")
parser.add_argument('--user_dim', type=int, default=300, help="User fearure dimension")
parser.add_argument('--item_dim', type=int, default=211, help="Item fearure dimension")
parser.add_argument('--user_ori_dim', type=int, default=1, help="User oringinal data dimension (if only provide ID number, then set 1)")
parser.add_argument('--item_ori_dim', type=int, default=1, help="Item oringinal data dimension (if only provide ID number, then set 1)")
parser.add_argument('--user_size', type=int, default=15400, help="User id size")
parser.add_argument('--item_size', type=int, default=1000, help="Item id size")
parser.add_argument('--prediction_size', type=int, default=2, help="The output dimension")
parser.add_argument('--user_item_size', type=int, nargs='+', default=[15400, 1000], help="User and Item id size")
parser.add_argument('--user_emb_dim', type=int, default=32, help="User feature dimension")
parser.add_argument('--item_emb_dim', type=int, default=32, help="Item feature dimension")
parser.add_argument('--ctr_layer_dims', type=int, nargs='+', default=[32, 16, 8],
                    help="Hidden layer dimension of ctr prediction model")
# parser.add_argument('--debias_mode',  type=str, default='IPM_Embedding', choices=['Noweight', 'IPM', 'DensityRatio'], help="The mode of weight")
parser.add_argument('--debias_mode', type=str, default='Adversarial',
                    choices=[ 'IPM_Embedding', 'Adversarial'], help="The mode of algorithm")
parser.add_argument('--downstream', type=str, default='MLP',
                    choices=['MLP', 'gmfBPR', 'bprBPR', 'mlpBPR', 'NeuBPR', 'LightGCN', 'DCN'], help="The mode of downstream model")
parser.add_argument('--weight_decay', type=float, default=0.001, help="weight_decay in BPR model")
parser.add_argument('--dropout', type=float, default=0)
# IPM function
parser.add_argument('--IMP_mode', type=str, default='gaussian', choices=['gaussian', 'functional'],
                    help="The mode of IPM function")
parser.add_argument('--kernel_mul', type=float, default=2.0, help='The mean of gaussian kernel')
parser.add_argument('--kernel_num', type=int, default=5, help='The number of kernel number')
parser.add_argument('--fix_sigma', type=float, default=None, help='The variance of gaussian kernel')

parser.add_argument('--ipm_layer_dims', type=int, nargs='+', default=[64, 32, 16],
                    help='Hidden layer dimension of IPM prediction model')
parser.add_argument('--ipm_embedding', type=int, default=256, help='Hidden layer dimension of IPM prediction model')
parser.add_argument('--embedding_classweight', type=int, nargs='+', default=[1, 4],
                    help='the class weight of objective function of discriminitor')
parser.add_argument('--ctr_classweight', type=int, nargs='+', default=[1, 10],
                    help='the class weight of objective function of prediction model')
# confounder inference
parser.add_argument('--confounder', type=str2bool, default=False, help='Train with confounder or not')
parser.add_argument('--confounder_dims', type=int, default=3, help='The dimension of confounder')
parser.add_argument('--confounder_encode_layer_dims', type=int, nargs='+', default=[64, 32, 16],
                    help='Hidden layer dimension of confounder inference model')
parser.add_argument('--confounder_decode_layer_dims', type=int, nargs='+', default=[64, 32, 16],
                    help='Hidden layer dimension of  generate model from confounder')
parser.add_argument('--lambda_confounder', type=float, default=0.05,
                    help='The weight of loss of confounder inference part')

# DCN configuration
parser.add_argument('--fc_dims', type=int, nargs='+', default=[64, 64],
                    help='Hidden layer dimension of Fc net in DCN layer')
parser.add_argument('--out_dims', type=int, default=32 + 299,
                    help='Output dimension of DCN Layer : number of feature dimension plus fc dimension in DCN Layer ')
parser.add_argument('--cross_depth', type=int, default=2, help='The depth of cross layer')

# density ratio function
parser.add_argument('--dr_layer_dim', type=list, default=[64, 32, 16],
                    help='Hidden layer dimension of IPM prediction model')
parser.add_argument('--dr_step', type=int, default=60, help='Load n-th density ratio model')
# data
parser.add_argument('--dataset', type=str, default='yahoo')

parser.add_argument('--batch_size', type=int, default=8192)
parser.add_argument('--num_worker', type=int, default=8,
                    help='number of threads to use for loading and preprocessing data')
parser.add_argument('--uni_percent', type=float, default=1.0, help='The percentage of uniform dataset')
parser.add_argument('--sparsity', type=float, default=1.0, help='The percentage of data sample')
parser.add_argument('--clip_value', type=float, nargs='+', default=[0.2, 0.8],
                    help='weighted clip')


def get_config():
    config, unparsed = parser.parse_known_args()
    current_time = time.localtime(time.time())
    config.time = '{}_{}_{}_{}'.format(current_time.tm_mon, current_time.tm_mday, current_time.tm_hour,
                                       current_time.tm_min)
    model_name = [
        ('name={:s}', config.name),
        ('dataset={:s}', config.dataset),
        ('use_weight={}', config.use_weight),
        ('debias_mode={}', config.debias_mode),
        ('confounder={}', config.confounder),
        ('downstream={}', config.downstream),
        ('is_debias={}', config.is_debias),
    ]
    config.model_dir = '_'.join([t.format(v) for (t, v) in model_name])
    print('Loaded ./config.py')
    return config, unparsed


if __name__ == '__main__':
    # for debug of config
    config, unparsed = get_config()

'''
# usage
from causal_controller.config import get_config as get_cc_config
cc_config,_=get_cc_config()
'''
