import torch
import numpy as np

from lib.utils.rot import(
    axis_angle_to_rot6d,
    rotmat_to_rot6d
)
from lib.utils.proc_hand_obj_output import(
    get_hand_joints_w_tip
)

# 点云最远点采样算法
def farthest_point_sample(xyz, npoint, random=False):
    """
    最远点采样(FPS算法) 减少点数量的同时保留点云的几何分布特征
    Input:
        xyz: pointcloud data, [B, N, 3]
             B:批次大小(一次处理的点云数量)
             N:每个点云包含的原始点数
             3:每个点的三维坐标(x, y, z)
        npoint: 需要采样的点数量
        random:是否随机选择初始点(默认False,即固定选第一个点作为初始点)
    Return:
        centroids: 采样点的索引,形状为 [B, npoint],即每个批次的点云中采样出的npoint个点的索引
    """
    device = xyz.device
    B, N, C = xyz.shape     # 解析点云形状：B=批次，N=总点数，C=坐标维度（3）
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) # 存储采样点索引，初始化为0
    distance = torch.ones(B, N).to(device) * 1e10   # 记录每个点到最近已选点的距离（初始设为极大值）
    if random:
        # 随机选择每个批次的初始点（索引范围0~N-1）
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    else:
        # 固定选择索引为0的点作为初始点
        farthest = 0
    # 生成批次索引（0~B-1），用于按批次访问点云
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        # 1. 记录当前选中的最远点索引
        centroids[:, i] = farthest

        # 2. 获取当前最远点的三维坐标
        # 按批次和索引取出点：xyz[batch_indices, farthest, :] → 形状[B, 3]
        # 重塑为[B, 1, 3]，方便后续计算与所有点的距离
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)

        # 3. 计算所有点到当前最远点的距离平方（省略开方，提高效率）
        # (xyz - centroid) → 每个点与当前中心点的坐标差 [B, N, 3]
        # 平方后求和（沿最后一维）→ 距离平方 [B, N]
        dist = torch.sum((xyz - centroid) ** 2, -1)

        # 4. 更新每个点到“最近已选点”的距离
        # 若当前点到新选点的距离更近，则更新distance
        mask = dist < distance  # 标记距离更小的点
        distance[mask] = dist[mask] # 更新最小距离

        # 5. 选择下一个最远点（距离最大的点的索引）
        farthest = torch.max(distance, -1)[1]   # 沿最后一维取最大值的索引
    return centroids

def proc_torch_cuda(d):
    if not isinstance(d, torch.Tensor):
        d = torch.FloatTensor(d)
    if d.device != "cuda":
        d = d.cuda()
    return d

def proc_torch_frame(l):
    if isinstance(l, list) or isinstance(l, np.ndarray):
        l = [torch.FloatTensor(_l).unsqueeze(0) for _l in l]
        l = torch.cat(l)
        l = l.cuda()
    return l

def proc_numpy(d):
    if isinstance(d, torch.Tensor):
        if d.requires_grad:
            d = d.detach()
        if d.is_cuda:
            d = d.cpu()
        d = d.numpy()
    return d

def transform_hand_to_xdata(trans, pose):
    trans_torch, pose_torch = proc_torch_frame(trans), proc_torch_frame(pose)
    nframes = pose_torch.shape[0]
    rot6d_torch = axis_angle_to_rot6d(pose_torch.reshape(-1, 3)).reshape(nframes, 16*6)
    xdata = torch.cat([trans_torch, rot6d_torch], dim=1)
    xdata = proc_numpy(xdata)
    return xdata

def transform_xdata_to_joints(xdata, hand_layer):
    xdata = proc_torch_cuda(xdata).unsqueeze(0)
    hand_joints = get_hand_joints_w_tip(xdata, hand_layer)
    hand_joints = proc_numpy(hand_joints.squeeze(0))
    return hand_joints

def transform_obj_to_xdata(obj_matrix):
    orl = proc_torch_frame(obj_matrix) # object rotation list
    obj_rotmat = orl[:, :3, :3]
    obj_trans = orl[:, :3, 3]
    nframes = obj_rotmat.shape[0]
    rot6d_torch = rotmat_to_rot6d(obj_rotmat).reshape(nframes, 6)
    xdata = torch.cat([obj_trans, rot6d_torch], dim=1)
    xdata = proc_numpy(xdata)
    return xdata

# 计算手部关节与物体顶点之间的距离
def get_hand_object_dist(hand_joints, sampled_obj_verts):
    hand_joints = proc_torch_cuda(hand_joints)
    sampled_obj_verts = proc_torch_cuda(sampled_obj_verts)

    # sampled_obj_verts = obj_verts[:, point_set]
    dist = torch.cdist(sampled_obj_verts, hand_joints)
    return dist

def get_which_hands_inter(lcf_idx, rcf_idx):
    # Contact frame idx, Contact object verts idx, Contact hand joints idx
    is_lhand = 0
    is_rhand = 0
    if len(lcf_idx) > 0:
        is_lhand = 1
    if len(rcf_idx) > 0:
        is_rhand = 1
    return is_lhand, is_rhand

# 
def get_contact_idx(dist, contact_threshold):
    # Contact frame idx, Contact object verts idx, Contact hand joints idx
    # 返回值含义：接触的帧索引（cf_idx）、接触的物体顶点索引（cov_idx）、接触的手关节索引（chj_idx）
    # 使用 torch.where 找出距离矩阵中小于接触阈值的位置
    # dist 是一个多维张量（通常维度为 [帧数量, 物体顶点数量, 手关节数量]）
    # 当 dist 中某个元素 < contact_threshold 时，记录其在各维度的索引
    cf_idx, cov_idx, chj_idx = torch.where(dist < contact_threshold)
    
    # 返回三个一维张量，分别对应满足条件的帧、物体顶点、手关节的索引
    return cf_idx, cov_idx, chj_idx

def get_contact_info(
     lhand_pose_list, lhand_beta_list, lhand_trans_list, 
        rhand_pose_list, rhand_beta_list, rhand_trans_list, 
        object_rotmat_list, lhand_layer, rhand_layer,
        sampled_obj_verts_org, mul_rv=True,
    ):
    contact_threshold = 0.02

    sampled_obj_verts_org = proc_torch_cuda(sampled_obj_verts_org)
    orl = proc_torch_frame(object_rotmat_list)
    obj_rotmat = orl[:, :3, :3]
    obj_trans = orl[:, :3, 3]

    if mul_rv:
        sampled_obj_verts = torch.einsum("tij,kj->tki", obj_rotmat, sampled_obj_verts_org) \
                            + obj_trans.unsqueeze(1)
    else:
        sampled_obj_verts = torch.einsum("tij,ki->tkj", obj_rotmat, sampled_obj_verts_org) \
                            + obj_trans.unsqueeze(1)
                            
    if len(lhand_pose_list) > 0:
        lpl, lbl, ltl = proc_torch_frame(lhand_pose_list), proc_torch_frame(lhand_beta_list), proc_torch_frame(lhand_trans_list)
        out_l = lhand_layer(lbl, lpl[..., :3], lpl[..., 3:])
        lhand_joints = out_l.joints_w_tip+ltl.unsqueeze(1)
        ldist = get_hand_object_dist(
            lhand_joints,
            sampled_obj_verts, 
        )
        lcf_idx, lcov_idx, lchj_idx = get_contact_idx(ldist, contact_threshold)
        ldist_value = ldist[lcf_idx, lcov_idx, lchj_idx]

    else:
        lcf_idx, lcov_idx, lchj_idx = np.array([]), np.array([]), np.array([])
        ldist_value = np.array([])

    if len(rhand_pose_list) > 0:
        rpl, rbl, rtl = proc_torch_frame(rhand_pose_list), proc_torch_frame(rhand_beta_list), proc_torch_frame(rhand_trans_list)
        out_r = rhand_layer(rbl, rpl[..., :3], rpl[..., 3:])
        rhand_joints = out_r.joints_w_tip+rtl.unsqueeze(1)
        rdist = get_hand_object_dist(
            rhand_joints,
            sampled_obj_verts, 
        )
        rcf_idx, rcov_idx, rchj_idx = get_contact_idx(rdist, contact_threshold)
        rdist_value = rdist[rcf_idx, rcov_idx, rchj_idx]
        
    else:
        rcf_idx, rcov_idx, rchj_idx = np.array([]), np.array([]), np.array([])
        rdist_value = np.array([])

    is_lhand, is_rhand = get_which_hands_inter(lcf_idx, rcf_idx)
    
    lcf_idx = proc_numpy(lcf_idx)
    lcov_idx = proc_numpy(lcov_idx)
    lchj_idx = proc_numpy(lchj_idx)
    ldist_value = proc_numpy(ldist_value)

    rcf_idx = proc_numpy(rcf_idx)
    rcov_idx = proc_numpy(rcov_idx)
    rchj_idx = proc_numpy(rchj_idx)
    rdist_value = proc_numpy(rdist_value)
    
    return (lcf_idx, lcov_idx, lchj_idx, ldist_value, 
            rcf_idx, rcov_idx, rchj_idx, rdist_value, 
            is_lhand, is_rhand)