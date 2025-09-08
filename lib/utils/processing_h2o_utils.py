import os
import os.path as osp

import glob
from scipy.spatial.transform import Rotation as R
import numpy as np
from pathlib import Path  # 需导入路径处理库pathlib的Path类

import torch

def process_text_h2o(
    action_name, is_lhand, is_rhand, 
    text_descriptions, return_key=False, 
):
    if is_lhand and is_rhand:
        text = f"{action_name} with both hands."
    elif is_lhand:
        text = f"{action_name} with left hand."
    elif is_rhand:
        text = f"{action_name} with right hand."
    text_key = text.capitalize() # Place cappuccino with right hand
    if return_key:
        return text_key
    else:
        text_description = text_descriptions[text_key]
        text = np.random.choice(text_description)
        return text


def get_data_path_h2o(data_path):
    # 将输入路径转换为Path对象
    data_path = Path (data_path) # data_path = data/h2o

    # 使用Path的glob方法获取对应目录下的所有txt文件，并转换为字符串路径
    hand_pose_manos = [str(p) for p in data_path.joinpath("hand_pose_mano").glob("*.txt")]
    obj_pose_rts = [str(p) for p in data_path.joinpath("obj_pose_rt").glob("*.txt")]
    cam_poses = [str(p) for p in data_path.joinpath("cam_pose").glob("*.txt")]
    action_labels = [str(p) for p in data_path.joinpath("action_label").glob("*.txt")]
    
    hand_pose_manos.sort()
    obj_pose_rts.sort()
    cam_poses.sort()
    action_labels.sort()
    return hand_pose_manos, obj_pose_rts, cam_poses, action_labels


def process_hand_pose_h2o(hand_pose, hand_trans, extrinsic_matrix):
    rot = R.from_rotvec(hand_pose[:3])
    mat = np.concatenate((np.concatenate((rot.as_matrix(), np.array(
        hand_trans[np.newaxis]).T), axis=1), [[0, 0, 0, 1]]), axis=0)
    mat_proj = np.dot(extrinsic_matrix, mat)
    rot_vec = R.from_matrix(mat_proj[:3, :3]).as_rotvec()
    return rot_vec


def process_hand_trans_h2o(hand_pose, hand_beta, hand_trans, extrinsic_matrix, hand_layer):
    mano_keypoints_3d = hand_layer(
        torch.FloatTensor(np.array([hand_beta])).cuda(),
        torch.FloatTensor(np.array([hand_pose[:3]])).cuda(),
        torch.FloatTensor(np.array([hand_pose[3:]])).cuda(),
    ).joints

    hand_origin = mano_keypoints_3d[0][0]
    origin = torch.unsqueeze(
        hand_origin, 1) + torch.tensor([hand_trans]).cuda().T
    origin = origin.float()
    extrinsic_matrix = torch.FloatTensor(extrinsic_matrix).cuda()
    mat_proj = torch.matmul(
        extrinsic_matrix, torch.cat((origin, torch.ones((1, 1)).cuda())))
    new_trans = mat_proj.T[0, :3] - hand_origin
    return new_trans.cpu().numpy(), hand_origin.cpu().numpy()

if __name__ == "__main__":
    test_path = r"data/h2o/subject1/h1/0/cam4"
    hand_pose_manos, obj_pose_rts, cam_poses, action_labels  = get_data_path_h2o(test_path)

    hand_pose_mano_path = hand_pose_manos[0]
    hand_pose_mano_data = np.loadtxt(hand_pose_mano_path)
    print(f"hand_pose_manos:{hand_pose_manos[0]}")
    print(f"hand_pose_mano_data:{hand_pose_mano_data}")
