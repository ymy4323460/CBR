import os
import shutil
import torch
import numpy as np
from torch.nn import functional as F
import gc
import hashlib
import json
import os
import sys
import time
# from catboost import Pool
from scipy.spatial.distance import pdist, squareform
import pandas as pd

def ndcg_func(model, datadir, top_k_list = [5, 10]):
    """Evaluate nDCG@K of the trained model on test dataset.
    """
    data = pd.read_csv(datadir).values

    x_te, y_te = data[:, :2], data[:, -2]
    all_user_idx = np.unique(x_te[:, 0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:,0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model.predict(x_u)

        for top_k in top_k_list:
            pred_top_k = np.argsort(-pred_u)[:top_k]
            count = y_u[pred_top_k].sum()

            log2_iplus1 = np.log2(1+np.arange(1,top_k+1))

            dcg_k = y_u[pred_top_k] / log2_iplus1

            best_dcg_k = y_u[np.argsort(-y_u)][:top_k] / log2_iplus1

            if np.sum(best_dcg_k) == 0:
                ndcg_k = 1
            else:
                ndcg_k = np.sum(dcg_k) / np.sum(best_dcg_k)

            result_map["ndcg_{}".format(top_k)].append(ndcg_k)

    return result_map

def save_model_by_name(model_dir, model, global_step, history=None):
	save_dir = os.path.join('checkpoints', model_dir, model.name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	file_path = os.path.join(save_dir, 'model.pt'.format(global_step))
	state = model.state_dict()
	torch.save(state, file_path)
	if history is not None:
		np.save(os.path.join(save_dir, 'test_metrics_history'), history)
	print('Saved to {}'.format(file_path))

def load_model_by_name(model_dir, model, global_step):
	"""
	Load a model based on its name model.name and the checkpoint iteration step

	Args:
		model: Model: (): A model
		global_step: int: (): Checkpoint iteration
	"""
	file_path = os.path.join('checkpoints', model_dir, model.name,
							 'model.pt')
	state = torch.load(file_path)
	model.load_state_dict(state)
	print("Loaded from {}".format(file_path))

ce = torch.nn.CrossEntropyLoss(reduction='none')

def cross_entropy_loss(x, logits):
	"""
	Computes the log probability of a Bernoulli given its logits

	Args:
		x: tensor: (batch, dim): Observation
		logits: tensor: (batch, dim): Bernoulli logits

	Return:
		log_prob: tensor: (batch,): log probability of each sample
	"""
	log_prob = ce(input=logits, target=x).sum(-1)
	return log_prob
def load_imputation(model_dir, model, global_step):
	"""
	Load a model based on its name model.name and the checkpoint iteration step

	Args:
		model: Model: (): A model
		global_step: int: (): Checkpoint iteration
	"""
	file_path = os.path.join('checkpoints', model_dir, model.name,
							 'model.pt')
	state = torch.load(file_path, map_location='cpu')
	model.load_state_dict(state)
	print("Loaded from {}".format(file_path))






def change_str_to_list(col_str):
    nums = list()
    cols = col_str.split(",")
    for col in cols:
        if col == "":
            nums = []
        elif "-" in col:
            index = col.split("-")
            start = int(index[0].strip())
            end = int(index[1].strip())
            for i in range(start, end + 1):
                nums.append(i)
        else:
            nums.append(int(col.strip()))
    # print(nums)
    return nums


def preprocess_label(config, data_df):
    if config["filter_label"][0] == "True":
        minimal_label = config["filter_label"][1]
        print('-' * 20)
        print("Label_filter: delete samples with label less than {}".format(minimal_label))
        print('DataShape before label filter', data_df.shape)
        try:
            data_df = data_df[data_df.label >= minimal_label]
        except ValueError:
            print(ValueError, data_df.label)
            exit(-2)
        print('DataShape after label filter', data_df.shape)

    if config["relabel"][0] == "True":
        print('-' * 20)
        print('before relabel, range:[{},{}]'.format(np.min(data_df['label'].values), np.max(data_df['label'].values)))
        for info in config["relabel"][1]:
            original_labels = change_str_to_list(info["original_labels"])
            # print(original_labels)
            data_df['label'] = data_df['label'].map(lambda y: str(info["new_label"]) if y in original_labels else y)
        data_df['label'] = data_df['label'].astype(int)
        gc.collect()
        print('after relabel, range:[{},{}]'.format(np.min(data_df['label'].values), np.max(data_df['label'].values)))

    if config["normalize_label"] == "True":
        print('-' * 20)
        y_min = np.min(data_df['label'].values)
        y_max = np.max(data_df['label'].values)
        print("Normalize_label: divide label by y_max {}, where y_min is {}".format(y_max, y_min))
        data_df['label'] = data_df['label'].map(lambda y: y / y_max)
        y_min = np.min(data_df['label'].values)
        y_max = np.max(data_df['label'].values)
        print('after normalize_label, range:[{},{}]'.format(y_min, y_max))
    gc.collect()
    return data_df


def intersect(list1, list2):
    ret = []
    for i in list2:
        for j in list1:
            if i in j:
                ret.append(j)
    return ret


def space_monitor(config_dict, df=None):
    t_now = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))
    path = config_dict['output_dir']
    print('-' * 20, '\n', t_now, path)
    os.system("ls %s" % path)
    if df is not None:
        df.info(verbose=True)
    gc.collect()
    disk_stat = os.statvfs(path)
    uasge_percent = (disk_stat.f_blocks - disk_stat.f_bfree) * 100 / (disk_stat.f_blocks - disk_stat.f_bfree + disk_stat.f_bavail) + 1
    print('usage percent {:.2f}%'.format(uasge_percent))
    return uasge_percent


def is_number(s):
    if s in [None, np.NaN, 'null', '']:
        return False
    else:
        return True


def fill_nan(config, data_df):
    fill_value_dict = dict()

    used_feature_names = [line.strip() for line in open('config/all_numerical_header_info')][4:]
    for fill_info in config["fill_nan"][1]:
        feature_type = fill_info.get("type")
        indices = change_str_to_list(fill_info.get("index"))
        fill_mode = fill_info.get("fill_mode")
        default_nan = int(fill_info.get("default_nan"))

        fill_features = intersect(used_feature_names, [feature_type + ',' + str(index) for index in indices])
        fill_value_dict[feature_type] = dict([[str(index), 0] for index in indices])
        for f in fill_features:
            original_values = np.array(list(filter(lambda x: is_number(x), data_df[f].values)))
            y = original_values[original_values != default_nan]
            if len(y) == 0:
                print('All \'{}\' is {}'.format(f, default_nan))
                fill_value = default_nan
            elif fill_mode == 'median':
                fill_value = np.median(y)
            elif fill_mode == 'max':
                fill_value = np.max(y)
            elif fill_mode == 'mean':
                fill_value = np.mean(y)
            else:
                fill_value = np.min(y)
            data_df[f].replace(default_nan, float(fill_value), inplace=True)  # replace original_value with fill_value
            print('fill::\'{}\'::original:{},fill_value:{}'.format(f, default_nan, float(fill_value)))
            fill_value_dict[f.split(',')[1]][str(f.split(',')[2])] = float(fill_value)   # return fill_value_dict dictionary
    print('Finish filling nan!')
    gc.collect()
    return data_df, fill_value_dict


# def prepare_pool(data_df):
#     X = data_df.drop(['label', 'query', 'id', 'docId'], axis=1)  # <class 'pandas.core.frame.DataFrame'>
#     Y = data_df['label']  # <class 'pandas.core.frame.DataFrame'>
#     group_id = data_df['id'].values.tolist()  # <class 'list'>
#     cat_features_indices = np.where((X.dtypes == 'category') | (X.dtypes == 'object'))[0]
#     del data_df
#     data_pool = Pool(data=X, label=Y, group_id=group_id, cat_features=cat_features_indices)
#     gc.collect()
#     os.system('free -g')
#     return data_pool


def generate_model_config(config, fill_value_dict=None):
    model_config = dict()
    all_features = [line.strip().split(',') for line in open(os.path.join(config["output_dir"], 'used_header_info'))][4:]
    all_features_df = pd.DataFrame(all_features, columns=['feature_name', 'feature_type', 'feature_id', 'data_type'])
    for k, v in all_features_df.groupby('feature_type'):
        model_config[k + "Max"] = int(np.max(v.feature_id.astype(float).tolist())) + 1
        model_config[k] = ""
    model_config["pageCategoryMax"] = 63
    model_config["siteCategoryMax"] = 20
    model_config["catboost_file_path"] = "../config/web/" + config["output_model_file"]
    model_config["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    model_config["md5"] = hashlib.md5(open(os.path.join(config["output_dir"], config["output_model_file"]), "rb").read()).hexdigest()

    used_features = [line.strip().split(',') for line in open('config/all_numerical_header_info')][4:]
    used_features_df = pd.DataFrame(used_features, columns=['feature_name', 'feature_type', 'feature_id', 'data_type'])
    for k, v in used_features_df.groupby('feature_type'):
        features_id_list = v.feature_id.tolist()
        if k in ["numericalFeatures", "matchFeatures", "clickFeatures", "videoNumericalFeatures", "newsNumericalFeatures"]:
            model_config[k] = dict([[index, 0] for index in features_id_list])
        else:
            model_config[k] = ','.join(index for index in features_id_list)

    if config['fill_nan'][0] == 'True' and fill_value_dict is not None:
        for k in fill_value_dict:
            model_config[k].update(fill_value_dict[k])
    print(model_config)
    with open(os.path.join(config["output_dir"], config["output_model_config"]), "w") as f:
        json.dump(model_config, f)



def distcorr(X, Y):
    """ Compute the distance correlation function

    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor
