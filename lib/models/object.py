
import pickle

class ObjectModel:
    def __init__(self, pkl_file):
        self.pkl_file = pkl_file   # data/h2o/h2o_objcet.pkl
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)  # 加载pkl文件
            self.object_names = data["object_names"]
            self.obj_pc_verts = data["obj_pc_verts"]
            self.obj_pc_normals = data["obj_pc_normals"]
            self.point_sets = data["point_sets"]
            self.obj_path = data["obj_path"]
            # 可选的物体顶部点云数据:可能用于特殊场景如"放置在物体顶部"的交互
            if "obj_pc_top" in data:
                self.obj_pc_top = data["obj_pc_top"]
            else:
                self.obj_pc_top = None

    # 可调用方法__call__:让ObjectModel的实例可以像函数一样被调用,根据输入的object_names获取对应的物体数据
    def __call__(self, object_names):
        # 若object_names是整数,则将其转换为对应的物体名称
        if isinstance(object_names, int):
            object_names = self.object_names[object_names]

        point_set = self.point_sets[object_names].copy() # (1024,)
        obj_pc_verts = self.obj_pc_verts[object_names].copy()     # (1024, 3)
        obj_pc_normals = self.obj_pc_normals[object_names].copy() # (1024, 3)
        obj_path = self.obj_path[object_names]    # book/book.obj

        if self.obj_pc_top is not None:
            obj_pc_top = self.obj_pc_top[object_names].copy()
            return point_set, obj_pc_verts, obj_pc_normals, obj_path, obj_pc_top
        else:
            return point_set, obj_pc_verts, obj_pc_normals, obj_path

def build_object_model(pkl_file):
    object_model = ObjectModel(pkl_file)
    return object_model

