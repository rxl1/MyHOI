
import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

import tqdm
import numpy as np
import hydra
from easydict import EasyDict as edict
from omegaconf import OmegaConf
from copy import deepcopy 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lib.networks.myCLIP import load_and_freeze_clip, encoded_text
from lib.datasets.datasets import get_dataloader