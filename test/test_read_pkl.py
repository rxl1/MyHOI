import warnings
# 忽略指定警告：trimesh.visual.texture 的 RuntimeWarning（内容匹配“invalid value encountered in cast”）
warnings.filterwarnings(
    action="ignore",          # 动作：忽略警告
    category=RuntimeWarning,  # 警告类型：RuntimeWarning（与报错一致）
    module="trimesh.visual.texture",  # 产生警告的模块（与报错路径一致）
    message="invalid value encountered in cast"  # 警告文本（完全复制报错中的提示）
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
import trimesh
import pickle
from easydict import EasyDict as edict
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))

def read_h2o_project_pkl():
    
    # 定义pkl文件路径（与保存路径对应）
    pkl_path = r"data\h2o\h2o_objcet.pkl"

    # 读取pkl文件
    with open(pkl_path, "rb") as f:
        # 加载数据（返回的是保存时的字典对象）
        data = pickle.load(f)

    print(f"data的类型:{type(data)}")
    print(f"data的所有key:{data.keys()}")

    print(f"object_names.type:{type(data['object_names'])}")
    print(f"object_names.len:{len(data['object_names'])}")
    print(f"object_names:{data['object_names']}")
    
    print(f"obj_pc_verts.type:{type(data['obj_pc_verts'])}")
    print(f"obj_pc_verts.keys:{data['obj_pc_verts'].keys()}")
    print(f"obj_pc_verts book.shape:{data['obj_pc_verts']['book'].shape}")
    print(f"obj_pc_verts book:{data['obj_pc_verts']['book']}")
    # print(f"obj_pc_verts:{data['obj_pc_verts']['h2o'].shape}")

    # 访问数据中的字段（根据保存时的键名）
    object_names = data["object_names"]  # 物体名称列表
    obj_pc_verts = data["obj_pc_verts"]          # 物体点云数据（字典，键为物体名称，值为点云坐标）
    obj_pc_normals = data["obj_pc_normals"]  # 点云法向量（字典）
    point_sets = data["point_sets"]    # 采样点索引（字典）
    obj_paths = data["obj_path"]       # 物体模型文件路径（字典）

    # 示例：打印第一个物体的点云信息
    if object_names:
        first_obj_name = object_names[1]
        print(f"示例...")
        print(f"第一个物体名称:{first_obj_name}")
        print(f"点云数据形状:{obj_pc_verts[first_obj_name].shape}")  # 应为 (1024, 3)（1024个3D点）
        print(f"法向量数据形状:{obj_pc_normals[first_obj_name].shape}")  # 应为 (1024, 3)
        print(f"point_sets数据形状:{point_sets[first_obj_name].shape}")  # 应为 (1024, 3)
        print(f"obj_paths:{obj_paths[first_obj_name]}")  # 应为 (1024, 3)（1024个3D点）

def read_milk_obj():
    path = r"data\h2o\object\milk\milk.obj"
    mesh = trimesh.load(path, maintain_order=True)
    mesh.show()

if __name__ == "__main__":
    # read_milk_obj()  # 测试milk.obj

    read_h2o_project_pkl()

    