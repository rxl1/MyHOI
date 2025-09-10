import numpy as np

import torch
import torch.nn.functional as F

from lib.utils.frame import sample_with_window_size
from lib.utils.rot import rot6d_to_rotmat
from lib.utils.proc_hand_obj_output import (
    get_pytorch3d_meshes, 
    get_hand_joints_w_tip, 
    get_transformed_obj_pc, 
    get_NN, get_interior
)

def get_filtered_loss_valid_mask(loss, valid_mask, loss_weight=None):
    """  
        desc: 过滤损失值,根据有效掩码对损失进行加权平均,并计算批次平均损失
        input:
            loss: 输入损失值,形状为 (batch_size, time_steps, ...)（批量大小、时间步数量、其他维度）
            valid_mask: 有效掩码,形状与 loss 相同,用于过滤无效区域
            loss_weight: 样本权重,用于加权损失计算
        output:
            filtered_loss: 过滤后的损失值
    """
    # 将高维损失（如含空间或特征维度）压缩为与 valid_mask 对齐的二维形状（batch_size, time_steps），便于后续掩码过滤
    if len(loss.shape)==3:    # 假设形状为 (batch_size, time_steps, features)
        loss_mean = loss.mean([2])    # 对特征维度（第2维）求平均，结果为 (batch_size, time_steps)
    elif len(loss.shape)==4:  # 假设形状为 (batch_size, time_steps, H, W)
        loss_mean = loss.mean([2, 3])  # 对空间维度（第2、3维）求平均，结果为 (batch_size, time_steps)
    
    # 根据 valid_mask 保留有效区域的损失，无效区域损失置为 0
    filtered_loss = torch.where(valid_mask, loss_mean, torch.zeros_like(loss_mean)) # 形状仍为 (batch_size, time_steps)，但无效区域的损失已被清零

    # 计算批次内有效损失的平均值,对时间 / 样本维度（第 1 维）求和，得到每个样本的总有效损失
    filtered_loss_summed = filtered_loss.sum(1)  # 形状：(batch_size,) 
    # 统计每个样本的有效区域数量（避免除以 0）
    valid_mask_summed = valid_mask.sum(1)   # 最终形状：(batch_size,) 代表对应样本在所有时间步中有效的帧数（或有效位置数量）
    # 若某样本无有效区域（和为 0），则用 1 代替分母，避免除法错误
    valid_mask_summed = torch.where(valid_mask_summed!=0, valid_mask_summed, torch.tensor(1).to(valid_mask_summed.device))
    # batch_mean 计算每个样本的平均有效损失
    filtered_loss_bm = filtered_loss_summed / valid_mask_summed

    if loss_weight is not None:  # 若提供 loss_weight, 根据损失权重weight1对每个样本的平均损失进行加权
        filtered_loss_bm = filtered_loss_bm * loss_weight
    
    # 对批次内所有样本的损失求平均，得到最终损失
    filtered_loss = filtered_loss_bm.mean()
    return filtered_loss

def l2_loss_unit(pred, targ, mask=None, weight=None):
    """ 计算预测值(pred)与目标值(targ)之间的 L2 损失(均方误差损失),
    并支持通过掩码(mask)过滤无效样本和设置权重(weight) """
    l2_loss = F.mse_loss(pred, targ, reduction='none') # 计算基础 L2 损失, reduction='none' 表示不自动进行降维(如求和或平均),而是保留原始的损失张量形状
    if mask is not None:
        filtered_loss = get_filtered_loss_valid_mask(l2_loss, mask, weight) # 计算掩码标记为 “有效” 的区域的损失(过滤掉无效区域)
    else:
        filtered_loss = l2_loss.mean()  # 直接对所有元素的损失取平均值作为最终结果
    return filtered_loss

def get_l2_loss(
        pred_lhand=None, pred_rhand=None, pred_obj=None, 
        targ_lhand=None, targ_rhand=None, targ_obj=None, 
        mask_lhand=None, mask_rhand=None, mask_obj=None, 
        weight=None, 
    ):
    """ 计算左手、右手和物体相关预测结果与目标值之间的总 L2 损失(均方误差损失),是对 l2_loss_unit 函数的封装和扩展 """
    total_loss = 0
    if targ_lhand is not None:
        lhand_loss = l2_loss_unit(pred_lhand, targ_lhand, mask_lhand, weight)
        total_loss += lhand_loss
    if targ_rhand is not None:
        rhand_loss = l2_loss_unit(pred_rhand, targ_rhand, mask_rhand, weight)
        total_loss += rhand_loss
    if targ_obj is not None:
        obj_loss = l2_loss_unit(pred_obj, targ_obj, mask_obj, weight)
        total_loss += obj_loss
    return total_loss

def get_filtered_loss_valid_map(loss, valid_map, weight=None):
    """  
        desc: 基于有效区域掩码(valid_map)筛选出需要计算的损失部分,并基于有效区域数量进行平均.
              针对形状为 (batch_size, nframes, 1024, 21) 的高维损失张量（如距离映射损失），
              仅计算 valid_map 标记的有效区域的平均损失，忽略无效区域的影响，同时支持加权和数值稳定性处理
        input:
            loss: 原始损失张量,形状为 (batch_size, time_steps, 1024, 21)(批量大小、时间步数量、1024个点、21个关节)
            valid_map: 有效映射,形状与 loss 相同,用于过滤无效区域
            weight: 样本权重,用于加权损失计算   
        output:
            filtered_loss: 过滤后的损失值
    """
    filtered_loss = torch.where(valid_map, loss, torch.zeros_like(loss)) # filtered_loss形状(batch_size, nframes, 1024, 21)
    filtered_loss_summed = filtered_loss.sum([1, 2, 3]) # batch, nframes, 1024, 21 最终形状为 (batch_size,)
    valid_map_summed = valid_map.sum([1, 2, 3])  # 对维度1、2、3求和 最终形状为 (batch_size,)
    valid_map_summed = torch.where(valid_map_summed!=0, valid_map_summed, torch.tensor(1).to(valid_map_summed.device))
    # batch_mean
    filtered_loss_bm = filtered_loss_summed / valid_map_summed 
    if weight is not None:
        filtered_loss_bm = filtered_loss_bm*weight
    filtered_loss = filtered_loss_bm.mean()
    return filtered_loss

def get_distance_map_loss(
        pred_ldist, pred_rdist, 
        targ_ldist, targ_rdist, 
        weight=None, 
    ):
    """  
        desc: 计算预测距离图与目标距离图之间的均方误差损失,用于评估模型预测的距离图与真实距离图之间的差异
        input:
            pred_ldist / pred_rdist: 预测的左手、右手距离图(通常为二维或三维张量,包含空间位置上的距离信息)
            targ_ldist / targ_rdist: 对应的真实距离图标签,与预测值形状一致
            weight: 样本权重,用于加权损失计算
        output:
            distance_map_loss: 左手和右手的距离图损失相加,得到最终结果
    """
    
    ldist_loss = F.mse_loss(pred_ldist, targ_ldist, reduction="none")  # reduction="none" 表示保留逐元素的损失值,不直接求和或平均,以便后续过滤无效区域
    rdist_loss = F.mse_loss(pred_rdist, targ_rdist, reduction="none")
    valid_map_ldist = targ_ldist > 0   # 左手距离图的有效掩码
    valid_map_rdist = targ_rdist > 0
    # 调用 get_filtered_loss_valid_map 函数,结合有效掩码对原始损失进行过滤和平均
    filtered_ldist_loss = get_filtered_loss_valid_map(ldist_loss, valid_map_ldist, weight)
    filtered_rdist_loss = get_filtered_loss_valid_map(rdist_loss, valid_map_rdist, weight)
    return filtered_ldist_loss + filtered_rdist_loss  # 左手和右手的距离图损失相加

def relative_rotation_matrix(R1, R2):
    """ 
    desc: 计算`手`相对于`物体`的旋转, 以物体为坐标系，手部相对于物体的旋转关系
    input:
        R1: 第一个旋转矩阵,形状为 (batch_size, 3, 3)   手
        R2: 第二个旋转矩阵,形状为 (batch_size, 3, 3)   物体
    output:
        relative_matrix: 相对旋转矩阵,形状为 (batch_size, 3, 3)
    """
    R1_inv = torch.inverse(R1)
    relative_matrix = torch.matmul(R2, R1_inv)
    return relative_matrix

# ro: relative orientation
def get_ro(hand, obj, valid_mask):
    """  
        desc: 计算手部与物体的相对旋转矩阵,用于评估手部相对于物体的姿态关系
              通过提取手部和物体的姿态信息（旋转部分），将其转换为旋转矩阵后，计算手部相对于物体的相对旋转矩阵，量化手部与物体在空间中的旋转姿态关系
        input:
            hand: 手部姿态数据,包含位置和旋转信息,形状通常为 (batch_size, time_steps, 21, 6)（批量大小、时间步数量、关节数量、旋转表示维度）
            obj: 物体姿态数据,包含位置和旋转信息,形状与 hand 相同
            valid_mask: 手部关节坐标的有效掩码,形状与 hand 相同,用于过滤无效区域
        output:
            ro_hand_obj: 手部与物体的相对旋转矩阵,形状为 (batch_size, time_steps, 3, 3)（批量大小、时间步数量、旋转矩阵维度）
    """
    # 姿态数据中通常前 3 个特征为位置（平移），后 6 个特征为旋转的 6D 表示（一种紧凑的旋转编码方式
    hand_orient = hand[valid_mask][..., 3:9]  # 手部旋转的6D表示
    obj_orient = obj[valid_mask][..., 3:9]    # 物体旋转的6D表示

    hand_orient_rotmat = rot6d_to_rotmat(hand_orient)
    obj_orient_rotmat = rot6d_to_rotmat(obj_orient)  # 将6D旋转表示转换为3x3旋转矩阵, 转换后形状为 (num_valid_frames, 3, 3)

    #  计算相对旋转矩阵
    ro_hand_obj = relative_rotation_matrix(hand_orient_rotmat, obj_orient_rotmat)  # 形状为 (num_valid_frames, 3, 3)，即手部相对于物体的相对旋转矩阵
    return ro_hand_obj

def get_relative_orientation_loss(
        pred_lhand, pred_rhand, pred_obj, 
        targ_lhand, targ_rhand, targ_obj, 
        mask_lhand, mask_rhand, 
        weight=None
    ):
    """  
        desc: 计算手部与物体的相对姿态损失,通过对比模型预测的 “左手 - 物体”“右手 - 物体” 相对旋转矩阵与真实相对旋转矩阵的偏差,
              确保手部相对于物体的姿态(如抓取时手掌朝向与物体表面的角度关系)符合物理交互逻辑
        input:
            pred_lhand / pred_rhand / pred_obj: 预测的左手、右手(包含指尖)、物体的姿态数据(含位置和旋转信息)
            targ_lhand / targ_rhand / targ_obj: 目标接触点坐标,与预测值形状相同,形状通常为 (batch_size, time_steps, features)（批量大小、时间步数量、特征维度）
            mask_lhand / mask_rhand: 手部关节坐标的有效掩码
            weight: 样本权重,用于加权损失计算
        output:
            relative_orientation_loss: 相对方向损失
    """
    # 调用 get_ro 函数分别计算预测值和真实值中 “左手 - 物体”“右手 - 物体” 的相对旋转矩阵
    pred_ro_lhand = get_ro(pred_lhand, pred_obj, mask_lhand)
    pred_ro_rhand = get_ro(pred_rhand, pred_obj, mask_rhand)
    targ_ro_lhand = get_ro(targ_lhand, targ_obj, mask_lhand)
    targ_ro_rhand = get_ro(targ_rhand, targ_obj, mask_rhand)
    if weight is not None:  # 带权重的损失计算(当 weight 存在时)
        nframes = targ_obj.shape[1]  # 提取time_steps,即 帧数
        weight = weight.unsqueeze(1).expand(-1, nframes) # 将权重从 (batch_size,) 扩展为 (batch_size, nframes)，与时间序列对齐
        weight_lhand = weight[mask_lhand]  # 仅保留有效帧的权重
        weight_rhand = weight[mask_rhand]  # 仅保留有效帧的权重
        ro_lhand_loss = F.mse_loss(pred_ro_lhand, targ_ro_lhand, reduction="none")  # 计算逐元素 MSE 损失
        ro_rhand_loss = F.mse_loss(pred_ro_rhand, targ_ro_rhand, reduction="none")
        ro_lhand_loss = ro_lhand_loss.mean([1, 2])*weight_lhand
        ro_rhand_loss = ro_rhand_loss.mean([1, 2])*weight_rhand
        ro_lhand_loss = ro_lhand_loss.mean()
        ro_rhand_loss = ro_rhand_loss.mean()
    else: # 无权重的损失计算（当 weight 不存在时）
        ro_lhand_loss = F.mse_loss(pred_ro_lhand, targ_ro_lhand)
        ro_rhand_loss = F.mse_loss(pred_ro_rhand, targ_ro_rhand)

    return ro_lhand_loss + ro_rhand_loss
