import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import os.path as osp   # 智能拼接路径组件
import sys
sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

"""参考链接:
    https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    https://www.zhihu.com/question/267963612/answer/2895706227
"""

""" 基于 PointNet 的点云特征提取网络,主要用于处理三维点云数据（如物体表面点云）并提取其全局或局部特征 """

class STNkd(nn.Module):
    """ 用于学习点云的空间变换矩阵（如旋转、缩放）,增强模型对输入点云姿态变化的鲁棒性,支持 k 维特征的变换 """
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    """ 基础点云特征提取网络,通过卷积层和最大池化层提取点云的全局 / 局部特征,可选择是否使用特征变换（feature_transform） """
    def __init__(
        self, 
        global_feat=True, 
        feature_transform=False, 
        in_dim=3, 
        **kwargs, 
    ):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(in_dim)
        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        x = x.transpose(2, 1) # B, N, D
        n_pts = x.size()[2] # B, D, N
        trans = self.stn(x) # B, D, D
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1).transpose(2, 1) # B, D, N -> B, N, D (B: batch_size, N: point num, D: feature dim)

class PointNetLatent(nn.Module):
    """ 在 PointNetfeat 基础上增加全连接层,将提取的特征映射到低维 latent 空间,用于生成点云的紧凑特征表示 """
    def __init__(
        self, 
        k=64*2,   # 输出的潜在特征维度，默认128
        feature_transform=False, # 是否使用特征变换（与PointNetfeat一致）
        in_dim=4,      # 输入点云的维度（如4维点云：3D坐标+1个额外特征）
    ):
        super(PointNetLatent, self).__init__()
        self.feature_transform = feature_transform
        # 复用PointNetfeat提取全局特征（全局特征维度为1024）
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, in_dim=in_dim)
        self.fc1 = nn.Linear(1024, 512) # 将 1024 维全局特征逐步降维到 k 维（默认 128 维），实现特征压缩
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)  # 引入 Dropout 层，防止过拟合
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feat(x)  # 调用PointNetfeat获取1024维全局特征
        x = F.relu(self.bn1(self.fc1(x)))  # 1024→512，激活+批归一化
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))  # 512→256，激活+Dropout+批归一化
        x = self.fc3(x)  # 256→k，输出最终的潜在特征
        return x