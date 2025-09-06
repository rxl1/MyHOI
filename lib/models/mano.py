from typing import NewType, Optional
import torch
from smplx import MANO
from lib.utils.file import load_config  # 配置加载工具
from dataclasses import dataclass, fields  # 用于定义数据类


# 加载MANO模型的配置文件（configs/mano.yaml）
config = load_config("configs/mano.yaml")
mano_config = config.mano
MODEL_DIR = mano_config.root                   # MANO模型文件的存储路径
SKELETONS = mano_config.skeletons              # 手部骨骼连接关系(不含指尖)
SKELETONS_W_TIP = mano_config.skeletons_w_tip  # 手部骨骼连接关系(含指尖)
left_hand_mean = mano_config.left_hand_mean    # 左手姿态均值
right_hand_mean = mano_config.right_hand_mean  # 右手姿态均值

Tensor = NewType('Tensor', torch.Tensor)  # 自定义Tensor类型注解

@dataclass
class MANO_Base_Out:
    vertices: Optional[Tensor] = None       # 手部网格顶点坐标
    joints: Optional[Tensor] = None         # 手部关节点坐标（不含指尖）
    joints_w_tip: Optional[Tensor] = None   # 含指尖的关节点坐标
    full_pose: Optional[Tensor] = None      # 完整姿态参数（全局旋转+手部姿态）
    global_orient: Optional[Tensor] = None  # 全局旋转参数
    transl: Optional[Tensor] = None         # 平移参数
    v_shaped: Optional[Tensor] = None       # 仅由形状参数决定的顶点坐标

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)] 
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


@dataclass
class MANO_Ext_Out(MANO_Base_Out):  # MANO_Extended_Output
    betas: Optional[Tensor] = None  # 形状参数（决定手部整体形状）
    hand_pose: Optional[Tensor] = None  # 手部姿态参数（除全局旋转外的关节旋转）
    skeletons: list = None  # 骨骼连接关系（不含指尖）
    skeletons_w_tip: list = None  # 骨骼连接关系（含指尖）


class MANO_Ext(MANO):  # MANO_Extended继承自smplx的MANO类
    def __init__(
        self, 
        model_path,  # MANO模型文件路径
        is_rhand=True,  # 是否为右手模型
        data_struct=None, 
        create_hand_pose=True,  # 是否创建手部姿态参数
        hand_pose=None,  # 手部姿态初始值
        use_pca=False,  # 不使用PCA降维（保留完整姿态参数）
        num_pca_comps=6, 
        flat_hand_mean=False,  # 是否使用"扁平手"均值姿态（简化手部初始姿态）
        batch_size=1, 
        dtype=torch.float32, 
        vertex_ids=None, 
        use_compressed=True, 
        ext='pkl', 
        **kwargs):
        # 调用父类MANO的初始化方法
        super().__init__(
            model_path, 
            is_rhand, 
            data_struct, 
            create_hand_pose, 
            hand_pose, 
            use_pca, 
            num_pca_comps, 
            flat_hand_mean, 
            batch_size, 
            dtype, 
            vertex_ids, 
            use_compressed, 
            ext,
            ** kwargs)
        
    def forward(
        self, 
        betas=None,  # 形状参数（B, 10），控制手部胖瘦等形状
        global_orient=None,  # 全局旋转（B, 3），控制手部整体朝向
        hand_pose=None,  # 手部关节姿态（B, 45），控制手指关节旋转（15个关节×3轴）
        transl=None,  # 平移参数（B, 3），控制手部位置
        return_verts=True,  # 是否返回顶点坐标
        return_full_pose=False,  # 是否返回完整姿态
        **kwargs) -> MANO_Ext_Out:
        
        # 调用父类forward，获取基础输出（顶点、关节点等）
        super_output = super().forward(betas, global_orient, hand_pose, transl, return_verts, return_full_pose,** kwargs)

        # 计算指尖位置（从顶点中提取特定索引的顶点作为指尖）
        thumb_tip = super_output.vertices[:, 745].unsqueeze(1)  # 拇指指尖顶点索引   形状：(B, 1, 3)  B 是批量大小，1 表示单个顶点，3 是三维坐标（x, y, z）
        index_tip = super_output.vertices[:, 317].unsqueeze(1)  # 食指指尖  形状：(B, 1, 3)
        middle_tip = super_output.vertices[:, 445].unsqueeze(1)  # 中指指尖 形状：(B, 1, 3)
        ring_tip = super_output.vertices[:, 556].unsqueeze(1)  # 无名指指尖 形状：(B, 1, 3)
        pinky_tip = super_output.vertices[:, 673].unsqueeze(1)  # 小指指尖  形状：(B, 1, 3)

        # 构建包含指尖的关节点（在原有关节点基础上拼接指尖）
        # 原有关节点 super_output.joints 的形状通常为 (B, J, 3)（J 是关节数量，如 MANO 模型默认有 16 个关节
        joints_w_tip = super_output.joints.clone() 
        joints_w_tip = torch.cat([
                joints_w_tip, index_tip, middle_tip, 
                pinky_tip, ring_tip, thumb_tip, 
            ], dim=1)  # 拼接之后的形状：(B, J+5, 3)

        # 封装输出为MANOOutput_C实例
        output = MANO_Ext_Out(
            **super_output,  # 继承父类输出（vertices, joints等）
            joints_w_tip=joints_w_tip,  # 新增含指尖的关节点
            skeletons=SKELETONS,  # 骨骼连接关系（不含指尖）
            skeletons_w_tip=SKELETONS_W_TIP,  # 骨骼连接关系（含指尖）
        )

        return output

def build_mano_hand(is_rhand, create_transl=False, flat_hand=False):
    return MANO_Ext(
        MODEL_DIR,  # 模型路径（从配置中获取）
        create_transl=create_transl,  # 是否创建平移参数（transl）
        use_pca=False,  # 不使用PCA，保留完整姿态参数
        flat_hand_mean=flat_hand,  # 是否使用扁平手均值姿态
        is_rhand=is_rhand,  # 区分左右手
    )