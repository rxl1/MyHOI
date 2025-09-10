import os
import os.path as osp   # 智能拼接路径组件
import sys
sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

import torch
from torch import Tensor
import torch.nn as nn

from lib.networks.pointnet import PointNetLatent