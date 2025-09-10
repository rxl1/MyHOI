
import torch

from lib.utils.rot import(
    rot6d_to_axis_angle
)

# lib/utils/data_augment.py 数据增强用到该文件
def proc_numpy(d):
    if isinstance(d, torch.Tensor):
        if d.requires_grad:
            d = d.detach()
        if d.is_cuda:
            d = d.cpu()
        d = d.numpy()
    return d

def get_hand_layer_out(hand_params, hand_layer):
    bs, nframes = hand_params.shape[:2]
    pred_rot6d = hand_params[..., 3:]
    pred_pose = rot6d_to_axis_angle(pred_rot6d).reshape(-1, 48)
    out = hand_layer(
        torch.zeros(bs*nframes, 10).to(hand_params.device), 
        pred_pose[..., :3].reshape(-1, 3), 
        pred_pose[..., 3:].reshape(-1, 45), 
    )
    return out

def get_hand_joints_w_tip(hand_params, hand_layer):
    bs, nframes = hand_params.shape[:2]
    pred_trans = hand_params[..., :3]
    out = get_hand_layer_out(hand_params, hand_layer)
    hand_joints_w_tip = out.joints_w_tip.reshape(bs, nframes, 21, 3)
    hand_joints_w_tip = hand_joints_w_tip + pred_trans.unsqueeze(2)
    return hand_joints_w_tip