import os
import trimesh
import torch
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))
from lib.utils.process_pointcloud import (
    farthest_point_sample  # 最远点采样
)

def process_object():
    path = r"F:\Chrome下载目录\Text2HOI相关\data\A_Text2HOI_data\h2o\object\book\book.obj"
    mesh = trimesh.load(path, maintain_order=True)
    # mesh.show()
    print(f"顶点数: {len(mesh.vertices)}")
    print(f"顶点形状:{mesh.vertices.shape}")
    print(f"面数: {len(mesh.faces)}")
    print(f"面的形状:{mesh.faces.shape}")
    verts = torch.FloatTensor(mesh.vertices.copy()).unsqueeze(0).cuda()
    print(f"verts的形状:{verts.shape}")
    normal = torch.FloatTensor(mesh.vertex_normals.copy()).unsqueeze(0).cuda()
    print(f"normal的形状:{normal.shape}")
    # print(f"normal数据:{normal[0][0:10]}")
    # normal = normal / torch.norm(normal, dim=2, keepdim=True)
    # print(f"归一化后normal数据:{normal[0][0:10]}")

def FPS():
    path = r"F:\Chrome下载目录\Text2HOI相关\data\A_Text2HOI_data\h2o\object\book\book.obj"
    mesh = trimesh.load(path, maintain_order=True)
    # mesh.show()
    verts = torch.FloatTensor(mesh.vertices.copy()).unsqueeze(0).cuda()
    # fps采样
    point_set = farthest_point_sample(verts, 1024)
    print(f"point_set形状:{point_set.shape}")
    print(f"采样结果 point_set:{point_set}")
    print(f"point_set[0] : {point_set[0]}")
    sampled_pointcloud = verts[0, point_set[0]].cpu().numpy()
    # print(f"sampled_pointcloud:{sampled_pointcloud.shape}")
    # print(f"采样结果 sampled_pointcloud:{sampled_pointcloud}")

def test_L2():
    vec = torch.tensor([1,2,3],dtype=torch.float32)
    norm = torch.norm(vec, dim=0, keepdim=False)
    normalized_vec = vec / norm
    print("原始向量：", vec)
    print("L2范数：", norm)
    print("归一化后向量：", normalized_vec)
    print("归一化后向量的模长（应接近1）：", torch.norm(normalized_vec))

if __name__=="__main__":
    # process_object()
    # test_L2()
    FPS()