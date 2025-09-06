
import torch

def func_1_mean_l2di_(a,b):
    """ 计算两个tensor之间的平均 L2 距离（欧氏距离） """
    x = torch.mean(torch.sqrt(torch.sum((a - b)**2, -1)))
    return x

def func_2_mean_l2di_(a,b):
    """ 计算两个tensor之间的平均 L2 距离（欧氏距离） """
    # 计算 a-b 在最后一个维度上的 L2 范数（即每个向量的欧氏距离）
    distances = torch.norm(a - b, p=2, dim=-1)
    # 对所有距离求平均
    return torch.mean(distances)

if __name__ == '__main__':
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[4.0, 6.0], [1.0, 1.0]])
    print("源代码：",func_1_mean_l2di_(a, b))
    print("新代码：",func_2_mean_l2di_(a, b))


