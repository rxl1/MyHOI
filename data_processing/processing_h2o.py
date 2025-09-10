import os
import os.path as osp   # 智能拼接路径组件

import glob
import warnings         # 根据指定的文件名模式(通配符模式)来查找和匹配文件系统中的文件和目录路径
import numpy as np
import tqdm
import time
import json
from pathlib import Path

# 忽略指定警告：trimesh.visual.texture 的 RuntimeWarning(内容匹配"invalid value encountered in cast")
warnings.filterwarnings(
    action="ignore",          # 动作：忽略警告
    category=RuntimeWarning,  # 警告类型：RuntimeWarning(与报错一致)
    module="trimesh.visual.texture",  # 产生警告的模块(与报错路径一致)
    message="invalid value encountered in cast"  # 警告文本(完全复制报错中的提示)
)
# 再导入 trimesh(必须在设置过滤规则之后,否则警告可能提前触发)
import trimesh
import pickle
import warnings
import sys
sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

from collections import Counter

import torch
from lib.utils.file import load_config
from lib.models.object import build_object_model
from lib.models.mano import build_mano_hand
from constants.h2o_constants import (
    h2o_obj_name,           # H2O数据集中的物体名称
    action_list,            # 动作列表
    present_participle,     # 动词现在分词
    third_verb,             # 第三人称动词
    passive_verb,           # 被动动词
)

from lib.utils.processing_h2o_utils import(
    process_text_h2o,
    get_data_h2o,
    process_hand_pose_h2o,
    process_hand_trans_h2o
)
from lib.utils.processing_utils import (
    farthest_point_sample,  # 最远点采样
    get_contact_info,
    transform_hand_to_xdata,
    transform_xdata_to_joints,
    transform_obj_to_xdata
)
from lib.utils.frame import align_frame


def process_object():
    """ 处理物体 """

    # 忽略指定模块的特定警告
    warnings.filterwarnings(
        action="ignore",
        category=RuntimeWarning,
        module="numpy.lib.nanfunctions",
        message="All-NaN slice encountered",      
    )
    warnings.filterwarnings(
        action="ignore",
        category=RuntimeWarning,
        module="np.nanmedian",
        message="invalid value encountered",      
    )
    warnings.filterwarnings(
        action='ignore',
        category=RuntimeWarning,
        module='trimesh.visual.texture',
        message='All-NaN slice encountered'  # 只忽略包含该文本的警告
    )

    # 加载h2o配置文件
    # h2o_config = load_config("configs/dataset/h2o.yaml")
    h2o_config = load_config("configs/dataset/gpu_h2o.yaml")
    
    # objects_folder = glob.glob(osp.join(h2o_config.obj_root, "*"))  # data/h2o/object/book ... 
    objects_folder = list(Path(h2o_config.obj_root).glob("*"))   # data/h2o/object/book  ...
    

    # pc = point cloud 
    obj_path = {}           # 存储物体路径
    obj_pc_verts = {}       # 存储采样后的物体点云数据
    obj_pc_normals = {}     # 存储采样后的物体点云法线
    point_sets = {}         # 存储物体点云集合(记录从原始点云中选择了哪些点)

    for object_folder in tqdm.tqdm(objects_folder):
        # object_paths = glob.glob(osp.join(object_folder, "*.obj"))  # data/h2o/object/book/book.obj  ...
        object_paths = list(Path(object_folder).glob("*.obj"))    # data/h2o/object/book/book.obj
        for object_path in tqdm.tqdm(object_paths):
            mesh = trimesh.load(object_path, maintain_order=True)
            verts = torch.FloatTensor(mesh.vertices.copy()).unsqueeze(0).cuda()  # verts.shape=[1,2465,3]
            normal = torch.FloatTensor(mesh.vertex_normals.copy()).unsqueeze(0).cuda() # normal.shape=[1,2465,3]
            # 对物体点云的法向量进行L2 归一化(单位化)处理,确保每个法向量的模长(长度)为 1
            normal = normal / torch.norm(normal, dim=2, keepdim=True)
            # 对vert进行最远点采样
            point_set = farthest_point_sample(verts, 1024)  # point_set.shape=[1,1024]
            sampled_pc_verts= verts[0, point_set[0]].cpu().numpy()
            sampled_pc_normal = normal[0, point_set[0]].cpu().numpy()

            # object_name = object_path.split("/")[-2]  # object_name = book 
            object_name = object_path.parts[-2]     # object_name = book 
            print(f"object_name:{object_name}")
            key = f"{object_name}"
            obj_pc_verts[key] = sampled_pc_verts
            obj_pc_normals[key] = sampled_pc_normal
            point_sets[key] = point_set[0].cpu().numpy()
            obj_path[key] = "/".join(object_path.parts[-2:])

            # os.makedirs("data/h2o", exist_ok=True)
            os.makedirs("/root/xinglin-data/data/h2o", exist_ok=True)
            # with open("data/h2o/h2o_objcet.pkl", "wb") as f:
            with open("/root/xinglin-data/data/h2o/h2o_objcet.pkl", "wb") as f:
                pickle.dump(
                    {
                        "object_names": h2o_obj_name, 
                        "obj_pc_verts": obj_pc_verts, 
                        "obj_pc_normals": obj_pc_normals, 
                        "point_sets": point_sets, 
                        "obj_path": obj_path, 
                    }, f)

def process_hand_object():
    start_time = time.time()
    # h2o_config = load_config("configs/dataset/h2o.yaml")
    h2o_config = load_config("configs/dataset/gpu_h2o.yaml")
    
    data_root = h2o_config.root  # data/h2o/
    data_save_path = h2o_config.npz_data_path # data/h2o/h2o_HOI_data.npz

    # 对处理好的物体pkl数据 构造 物体模型
    object_model = build_object_model(h2o_config.data_obj_pc_path)  # data/h2o/h2o_objcet.pkl

    lhand_layer = build_mano_hand(is_rhand=False, flat_hand=h2o_config.flat_hand)
    lhand_layer = lhand_layer.cuda()
    rhand_layer = build_mano_hand(is_rhand=True, flat_hand=h2o_config.flat_hand)
    rhand_layer = rhand_layer.cuda()

    # 存储所有样本的左手 / 右手处理后的特征数据
    x_lhand_total, x_rhand_total = [], []  

    # 存储所有样本的左手 / 右手关节点(joints)坐标数据
    joints_lhand_total, joints_rhand_total = [], []  

    # 左/右根节点位置,存储所有样本的左右手根节点在空间中的坐标
    x_lhand_org_total, x_rhand_org_total = [], []  

    """ 
    cf可能为"contact finger"(接触手指),
    cov可能为 "coverage"(覆盖区域),
    chj可能为 "contact joint"(接触关节),
    dist_value为距离值 """
    # 左手 / 右手接触手指的索引
    lcf_idx_total, rcf_idx_total = [], [] 

    # 左手 / 右手接触区域的索引
    lcov_idx_total, rcov_idx_total = [], [] 

    # 左手 / 右手接触'关节'的索引
    lchj_idx_total, rchj_idx_total = [], [] 

    # 左手 / 右手与物体的接触距离值
    ldist_value_total, rdist_value_total = [], [] 

    # 存储每个样本中 "是否有左手参与交互"/"是否有右手参与交互" 的标志
    is_lhand_total, is_rhand_total = [], [] 

    # 存储所有样本的左手 / 右手形状参数(MANO 模型中beta用于描述手部的个性化形状)
    lhand_beta_total, rhand_beta_total = [], [] 

    # 存储所有样本的物体处理后的特征数据 
    x_obj_total = []

    # 存储所有样本对应的物体索引(用于关联预设的物体模型)
    object_idx_total = []
    # 存储每个样本对应的动作编号(整数标识)
    action_total = []
    # 存储每个样本对应的动作名称
    action_name_total = []
    # 存储每个样本包含的帧数
    nframes_total = []
    # 存储每个样本对应的受试者信息
    subject_total = []
    # 存储每个样本对应的背景信息
    background_total = []
    # 没有手部参与交互的动作名称
    no_inter_action_name = []
    '''
        subject, background, object class, cam, 
    '''

    # 将 data_root 转换为 Path 对象
    data_root_path = Path(data_root)  # data/h2o/
    # 定义要收集的受试者目录
    subjects = ["subject1", "subject2", "subject3"]  # subject3 可通过注释控制是否包含
    # 收集所有符合模式的路径（转换为字符串格式，保持与原代码一致）
    subject_cam_paths = []
    for subject in subjects:
        # 路径模式：subjectX/任意背景/任意物体类别/cam开头的相机目录
        pattern = f"{subject}/*/*/cam*"
        # 遍历匹配的路径并转换为字符串
        subject_cam_paths.extend(str(path) for path in data_root_path.glob(pattern))
    # 排序确保路径顺序一致
    subject_cam_paths.sort()  
    # print(f"enumerasubject_cam_paths:{subject_cam_paths[0:10]}") # '/root/xinglin-data/data/h2o/subject1/h1/0/cam4', ...

    for data_idx, data_path in enumerate(subject_cam_paths):
        hand_pose_manos, obj_pose_rts, cam_poses, action_labels \
            = get_data_h2o(data_path)
        print(f"data_path:{data_path}")
        # 初始化上一帧的动作标识为0,用于跟踪动作序列的变化
        prev_action = 0  
        # 存储当前动作序列中每一帧的手部相关数据
        lhand_pose_list, rhand_pose_list = [],[]    # 左/右手的姿态参数
        lhand_beta_list, rhand_beta_list = [],[]   # 左/右手手的形状参数
        lhand_trans_list, rhand_trans_list = [],[]   # 左/右手手的平移参数
        x_lhand_org_list, x_rhand_org_list = [],[]   # 左v手的原始根节点位置
        # 当前动作序列中每一帧的物体旋转矩阵(描述物体在 3D 空间中的旋转姿态
        object_rotmat_list = [] 

        """ 
        遍历数据集中的每一组帧（包含手部姿态、物体姿态、相机参数、动作标签），
        按动作序列（同一动作连续的帧）进行处理 
        """
        for hand_pose_mano, obj_pose_rt, cam_pose, action_label in \
            tqdm.tqdm(zip(hand_pose_manos, 
                          obj_pose_rts, 
                          cam_poses, 
                          action_labels), desc=f"{data_idx}/{len(subject_cam_paths)}", total=len(obj_pose_rts)):
            # 1.加载原始数据
            hand_pose_mano_data = np.loadtxt(hand_pose_mano)  # 手部姿态/形状/位置数据
            obj_pose_rt_data = np.loadtxt(obj_pose_rt)      # 物体姿态数据
            extrinsic_matrix = np.loadtxt(cam_pose).reshape(4, 4)   # 相机外参矩阵(3D坐标转换用)
            action = int(np.loadtxt(action_label))      # 当前帧的动作标签(整数标识)

            # 2.动作序列分割(检测动作变化)
            if action != prev_action and prev_action != 0:
                # 对长度大于20帧的动作序列计算手部与物体的接触交互信息
                if len(object_rotmat_list) > 20:
                    _, obj_pc, _, _ = object_model(int(obj_idx)) # 获取物体点云数据
                    # 计算接触信息
                    # lcf_idx:接触点索引 lcov_idx:覆盖点索引 lchj_idx:接触关节索引 ldist_value:与物体的距离值
                    lcf_idx, lcov_idx, lchj_idx, ldist_value, \
                    rcf_idx, rcov_idx, rchj_idx, rdist_value, \
                    is_lhand, is_rhand = get_contact_info(
                        lhand_pose_list, lhand_beta_list, lhand_trans_list, 
                        rhand_pose_list, rhand_beta_list, rhand_trans_list, 
                        object_rotmat_list, lhand_layer, rhand_layer,
                        obj_pc, 
                    )
                    # 过滤掉"无手部与物体交互"的动作序列
                    if is_lhand == 0 and is_rhand == 0:
                        # 记录无交互的动作名称
                        action_name = action_list[prev_action] 
                        no_inter_action_name.append(action_name)
                        # 清空当前序列的缓存数据
                        lhand_pose_list, rhand_pose_list = [], []
                        lhand_beta_list, rhand_beta_list = [], []
                        lhand_trans_list, rhand_trans_list = [], []
                        x_lhand_org_list, x_rhand_org_list = [], []
                        object_rotmat_list = []
                        # 更新状态并跳过后续处理
                        prev_action = action # 更新“上一个动作”为当前动作
                        continue    # 跳过当前序列的后续特征提取和保存步骤
                    
                    """ 对有效动作序列(帧长>20)进行数据转换,并将处理后的特征数据收集到总列表中 """
                    # 转换左手数据为模型输入格式
                    x_lhand = transform_hand_to_xdata(lhand_trans_list, lhand_pose_list)
                    # 转换右手数据为模型输入格式
                    x_rhand = transform_hand_to_xdata(rhand_trans_list, rhand_pose_list)
                    # 将左手输入数据转换为关节坐标
                    joint_lhand = transform_xdata_to_joints(x_lhand, lhand_layer)
                    # 将右手输入数据转换为关节坐标
                    joint_rhand = transform_xdata_to_joints(x_rhand, rhand_layer)
                    # 转换物体旋转矩阵列表为模型输入格式
                    x_obj = transform_obj_to_xdata(object_rotmat_list)
                    x_lhand_total.append(x_lhand)      # 左手模型输入特征
                    x_rhand_total.append(x_rhand)      # 右手模型输入特征
                    joints_lhand_total.append(joint_lhand) # 左手关节坐标
                    joints_rhand_total.append(joint_rhand) # 右手关节坐标
                    x_obj_total.append(x_obj)   # 物体模型输入特征
                    x_lhand_org_total.append(x_lhand_org_list) # 左手根节点位置
                    x_rhand_org_total.append(x_rhand_org_list) # 右手根节点位置
                    """ 手部 - 物体接触交互信息 """
                    lcf_idx_total.append(lcf_idx)   # 左手接触点索引
                    lcov_idx_total.append(lcov_idx) # 左手覆盖点索引
                    lchj_idx_total.append(lchj_idx) # 左手接触关节索引
                    ldist_value_total.append(ldist_value) # 左手与物体的距离值
                    rcf_idx_total.append(rcf_idx)
                    rcov_idx_total.append(rcov_idx)
                    rchj_idx_total.append(rchj_idx)
                    rdist_value_total.append(rdist_value)
                    is_lhand_total.append(is_lhand) # 左手是否交互（1/0）
                    is_rhand_total.append(is_rhand)
                    """ 辅助属性与元数据 """
                    lhand_beta_total.append(lhand_beta_list)  # 左手形状参数（影响手部 mesh 形态）
                    rhand_beta_total.append(rhand_beta_list)  # 右手形状参数
                    object_idx_total.append(int(obj_idx))     # 物体索引（标识交互物体类别
                    action_name = action_list[prev_action]    # 
                    action_total.append(prev_action)          # 动作标签（数字标识
                    action_name_total.append(action_name)     # 动作名称（如“pick up cup
                    nframes_total.append(len(object_rotmat_list)) # 动作序列长度（帧数
                    subject_total.append(data_path.split("/")[5]) # 数据所属主体（如“subject1
                    background_total.append(data_path.split("/")[6])  # 场景背景信息

                lhand_pose_list = []
                lhand_beta_list = []
                lhand_trans_list = []
                x_lhand_org_list = []
                rhand_pose_list = []
                rhand_beta_list = []
                rhand_trans_list = []
                x_rhand_org_list = []
                object_rotmat_list = []

            # 过滤掉动作标签为 0 的无效帧
            if action == 0:
                prev_action = action
                continue
            
            lhand_trans = hand_pose_mano_data[1:4]
            lhand_pose = hand_pose_mano_data[4:52]
            lhand_beta = hand_pose_mano_data[52:62]

            left_rotvec = process_hand_pose_h2o(lhand_pose, lhand_trans, extrinsic_matrix)
            lhand_pose[:3] = left_rotvec

            new_left_trans, lhand_origin = process_hand_trans_h2o(lhand_pose, lhand_beta, lhand_trans, extrinsic_matrix, lhand_layer)
            lhand_trans_list.append(new_left_trans)
            lhand_pose_list.append(lhand_pose)
            lhand_beta_list.append(lhand_beta)
            x_lhand_org_list.append(lhand_origin)

            rhand_trans = hand_pose_mano_data[63:66]
            rhand_pose = hand_pose_mano_data[66:114]
            rhand_beta = hand_pose_mano_data[114:124]

            right_rotvec = process_hand_pose_h2o(rhand_pose, rhand_trans, extrinsic_matrix)
            rhand_pose[:3] = right_rotvec

            new_right_trans, rhand_origin = process_hand_trans_h2o(rhand_pose, rhand_beta, rhand_trans, extrinsic_matrix, rhand_layer)
            rhand_trans_list.append(new_right_trans)
            rhand_pose_list.append(rhand_pose)
            rhand_beta_list.append(rhand_beta)
            x_rhand_org_list.append(rhand_origin)

            obj_idx = obj_pose_rt_data[0]
            object_ext = obj_pose_rt_data[1:].reshape(4, 4)

            new_object_matrix = np.dot(extrinsic_matrix, object_ext)
            object_rotmat_list.append(new_object_matrix)

            prev_action = action

        if len(object_rotmat_list) > 20:
            _, obj_pc, _, _ = object_model(int(obj_idx)) # 获取物体点云数据
            # 计算接触信息
            # lcf_idx:接触点索引 lcov_idx:覆盖点索引 lchj_idx:接触关节索引 ldist_value:与物体的距离值
            lcf_idx, lcov_idx, lchj_idx, ldist_value, \
            rcf_idx, rcov_idx, rchj_idx, rdist_value, \
            is_lhand, is_rhand = get_contact_info(
                lhand_pose_list, lhand_beta_list, lhand_trans_list, 
                rhand_pose_list, rhand_beta_list, rhand_trans_list, 
                object_rotmat_list, lhand_layer, rhand_layer,
                obj_pc, 
            )
            # 过滤掉"无手部与物体交互"的动作序列
            if is_lhand == 0 and is_rhand == 0:
                # 记录无交互的动作名称
                action_name = action_list[prev_action] 
                no_inter_action_name.append(action_name)
                # 清空当前序列的缓存数据
                lhand_pose_list, rhand_pose_list = [], []
                lhand_beta_list, rhand_beta_list = [], []
                lhand_trans_list, rhand_trans_list = [], []
                x_lhand_org_list, x_rhand_org_list = [], []
                object_rotmat_list = []
                # 更新状态并跳过后续处理
                prev_action = action # 更新“上一个动作”为当前动作
                continue    # 跳过当前序列的后续特征提取和保存步骤
            
            """ 对有效动作序列(帧长>20)进行数据转换,并将处理后的特征数据收集到总列表中 """
            # 转换左手数据为模型输入格式
            x_lhand = transform_hand_to_xdata(lhand_trans_list, lhand_pose_list)
            # 转换右手数据为模型输入格式
            x_rhand = transform_hand_to_xdata(rhand_trans_list, rhand_pose_list)
            # 将左手输入数据转换为关节坐标
            joint_lhand = transform_xdata_to_joints(x_lhand, lhand_layer)
            # 将右手输入数据转换为关节坐标
            joint_rhand = transform_xdata_to_joints(x_rhand, rhand_layer)
            # 转换物体旋转矩阵列表为模型输入格式
            x_obj = transform_obj_to_xdata(object_rotmat_list)
            x_lhand_total.append(x_lhand)      # 左手模型输入特征
            x_rhand_total.append(x_rhand)      # 右手模型输入特征
            joints_lhand_total.append(joint_lhand) # 左手关节坐标
            joints_rhand_total.append(joint_rhand) # 右手关节坐标
            x_obj_total.append(x_obj)   # 物体模型输入特征
            x_lhand_org_total.append(x_lhand_org_list) # 左手根节点位置
            x_rhand_org_total.append(x_rhand_org_list) # 右手根节点位置
            """ 手部 - 物体接触交互信息 """
            lcf_idx_total.append(lcf_idx)   # 左手接触点索引
            lcov_idx_total.append(lcov_idx) # 左手覆盖点索引
            lchj_idx_total.append(lchj_idx) # 左手接触关节索引
            ldist_value_total.append(ldist_value) # 左手与物体的距离值
            rcf_idx_total.append(rcf_idx)
            rcov_idx_total.append(rcov_idx)
            rchj_idx_total.append(rchj_idx)
            rdist_value_total.append(rdist_value)
            is_lhand_total.append(is_lhand) # 左手是否交互（1/0）
            is_rhand_total.append(is_rhand)
            """ 辅助属性与元数据 """
            lhand_beta_total.append(lhand_beta_list)  # 左手形状参数（影响手部 mesh 形态）
            rhand_beta_total.append(rhand_beta_list)  # 右手形状参数
            object_idx_total.append(int(obj_idx))     # 物体索引（标识交互物体类别
            action_name = action_list[prev_action]    # 
            action_total.append(prev_action)          # 动作标签（数字标识
            action_name_total.append(action_name)     # 动作名称（如“pick up cup
            nframes_total.append(len(object_rotmat_list)) # 动作序列长度（帧数
            subject_total.append(data_path.split("/")[5]) # 数据所属主体  subject1
            # print(f"subject_total:{subject_total}")
            background_total.append(data_path.split("/")[6])  # 场景背景信息  h1,h2,o1,o2,k1,k2...
            # print(f"background_total:{background_total}")
    
    total_dict = {
        "x_lhand": x_lhand_total,  # 左手 / 右手处理后的特征数据
        "x_rhand": x_rhand_total,
        "joints_lhand": joints_lhand_total,  # 左手 / 右手关节点(joints)坐标数据
        "joints_rhand": joints_rhand_total,
        "x_obj": x_obj_total,  # 所有样本的物体处理后的特征数据
        "lhand_beta": lhand_beta_total,  # 左手 / 右手形状参数
        "rhand_beta": rhand_beta_total,
        "lhand_org": x_lhand_org_total,  # 左/右根节点位置,存储所有样本的左右手根节点在空间中的坐标
        "rhand_org": x_rhand_org_total,  
    }

    final_dict = align_frame(total_dict)

    # 生成.npz文件
    np.savez(  # 将多个数组保存到一个压缩文件中的函数，生成的文件扩展名为 .npz
        data_save_path,
        **final_dict,
        lcf_idx=np.array(lcf_idx_total, dtype=object), 
        lcov_idx=np.array(lcov_idx_total, dtype=object), 
        lchj_idx=np.array(lchj_idx_total, dtype=object), 
        ldist_value=np.array(ldist_value_total, dtype=object), 
        rcf_idx=np.array(rcf_idx_total, dtype=object), 
        rcov_idx=np.array(rcov_idx_total, dtype=object), 
        rchj_idx=np.array(rchj_idx_total, dtype=object), 
        rdist_value=np.array(rdist_value_total, dtype=object), 
        is_lhand=np.array(is_lhand_total), 
        is_rhand=np.array(is_rhand_total), 
        object_idx=np.array(object_idx_total),
        action=np.array(action_total),
        action_name=np.array(action_name_total),
        nframes=np.array(nframes_total),
        subject=np.array(subject_total),
        background=np.array(background_total),
    )
    print("总共用时:", time.time()-start_time)

def process_text():
    # 加载h2o配置文件
    # h2o_config = load_config("configs/dataset/h2o.yaml")
    h2o_config = load_config("configs/dataset/gpu_h2o.yaml")
    text_json = h2o_config.text_json

    text_description = {}
    for action in action_list[1:]:
        text_left = f"{action} with left hand.".capitalize()  # Grab book with left hand
        text_right = f"{action} with right hand.".capitalize()
        text_both = f"{action} with both hands.".capitalize()
        # print(f"action:{action}")
        # print(f"text_left:{text_left}")
        # print(f"text_right:{text_right}")
        # print(f"text_both:{text_both}")

        # 现在进行时
        action_verb, action_object = " ".join(action.split(" ")[:-1]), action.split(" ")[-1]
        action_ving = present_participle[action_verb]
        text_left1 = f"{action_ving} {action_object} with left hand.".capitalize()
        text_right1 = f"{action_ving} {action_object} with right hand.".capitalize()
        text_both1 = f"{action_ving} {action_object} with both hands.".capitalize()
        # print(f"action_v:{action_verb}")
        # print(f"action_o:{action_object}")
        # print(f"text_left1:{text_left1}")
        # print(f"text_right1:{text_right1}")
        # print(f"text_both1:{text_both1}")

        # 第三人称
        action_3rd_v = third_verb[action_verb]
        text_left2 = f"Left hand {action_3rd_v} {action_object}."
        text_right2 = f"Right hand {action_3rd_v} {action_object}."
        text_both2 = f"Both hands {action_verb} {action_object}."
        
        # 过去式
        action_passive = passive_verb[action_verb]
        text_left3 = f"{action_object} {action_passive} with left hand.".capitalize()
        text_right3 = f"{action_object} {action_passive} with right hand.".capitalize()
        text_both3 = f"{action_object} {action_passive} with both hands.".capitalize()

        text_description[text_left] = [text_left, text_left1, text_left2, text_left3]
        text_description[text_right] = [text_right, text_right1, text_right2, text_right3]
        text_description[text_both] = [text_both, text_both1, text_both2, text_both3]

    with open(text_json, "w") as f:
        json.dump(text_description, f)

def process_balance_weights():
    # h2o_config = load_config("configs/dataset/h2o.yaml")
    h2o_config = load_config("configs/dataset/gpu_h2o.yaml")
    npz_data_path = h2o_config.npz_data_path
    balance_weights_path = h2o_config.balance_weights_path  # 平衡权重pkl文件
    text_count_json_path = h2o_config.text_count_json


    with np.load(npz_data_path, allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]  

    text_list = []
    for i in range(len(action_name)):   # place cappuccino
        text_key = process_text_h2o(
            action_name[i], is_lhand[i], is_rhand[i], 
            text_descriptions=None, return_key=True
        )
        text_list.append(text_key)  # Place cappuccino with right hand      text_list记录所有文本描述

    text_counter = Counter(text_list)
    # print(f"text_counter.type:{type(text_counter)}")
    # for text_key, count in text_counter.items():
    #     print(f"文本: {text_key}  出现次数: {count}")
    text_dict = dict(text_counter)
    # 打印测试
    # for k, v in text_dict.items():
    #     print(f"k:{k}")
    #     print(f"v:{v}")
    # print(f"{text_dict}")
    text_prob = {k:1/v for k, v in text_dict.items()}  # 计算权重（出现次数越多，权重越低，用于平衡样本）
    h2o_balance_weights = [text_prob[text] for text in text_list]  # 权重列表:为每个样本生成对应的平衡权重
    # with open(balance_weights_path, "wb") as f:  # 保存平衡权重到 pkl 文件（训练时使用）
    #     pickle.dump(h2o_balance_weights, f)     
    with open(text_count_json_path, "w") as f:  # 保存文本计数到 JSON 文件（用于统计分析）
        json.dump(text_dict, f)

def process_action_frame_length():
    """ 动作样本的帧长 """
    # h2o_config = load_config("configs/dataset/h2o.yaml")
    h2o_config = load_config("configs/dataset/gpu_h2o.yaml")
    npz_data_path = h2o_config.npz_data_path
    action_frame_length_json_path = h2o_config.action_frame_length_json
    
    with np.load(npz_data_path, allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]
        nframes = data["nframes"]   # 每个样本的帧数(动作序列长度)

    action_frame_length_dict = {}
    for i in range(len(action_name)):
        # 生成当前样本的文本关键字（如"Pick up cup with left hand."）
        text_key = process_text_h2o(
            action_name[i], is_lhand[i], is_rhand[i], 
            text_descriptions=None, return_key=True
        )

        # 获取当前样本的帧数，超过150则截断为150（限制最大序列长度
        num_frames = int(nframes[i])
        if num_frames > 150:
            num_frames = 150
        if text_key not in action_frame_length_dict:
            action_frame_length_dict[text_key] = [num_frames]
        else:
            action_frame_length_dict[text_key].append(num_frames)
    print(f"text_dict:{action_frame_length_dict}")

    with open(action_frame_length_json_path, "w") as f:
        json.dump(action_frame_length_dict, f)

def print_text_data_num():
    # h2o_config = load_config("configs/dataset/h2o.yaml")
    h2o_config = load_config("configs/dataset/gpu_h2o.yaml")
    npz_data_path = h2o_config.npz_data_path
    action_frame_length_json_path = h2o_config.action_frame_length_json
    
    with np.load(npz_data_path, allow_pickle=True) as data:
        action_name = data["action_name"]
    print(f"data num: {len(action_name)}")
    
    with open(action_frame_length_json_path, "r") as f:
        text = json.load(f)
    print(f"text num: {len(text)}")

if __name__ == '__main__':
    # process_object()
    # process_hand_object()
    # process_text()
    # process_balance_weights()
    # process_action_frame_length()

    print_text_data_num()


            
            
