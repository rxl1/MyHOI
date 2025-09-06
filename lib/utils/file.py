from easydict import EasyDict as edict
import yaml
import numpy as np
import tqdm
import json
import pickle



""" 读取文件的相关函数 """
# 读取yaml配置文件
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)

# 读取json文件
def read_json(json_file_path):
    with open(json_file_path, "r") as f:
        json_data = json.load(f)
    return json_data

# 读取pkl文件
def read_pkl(pkl_file_path):
    with open(pkl_file_path, "rb") as f:
        pkl_data = pickle.load(f)
    return pkl_data


