import torch

# 处理ARCTIC数据集需要用到该文件
def pad_tensor_list(v_list: list):
    """ 
    将由不同长度张量组成的列表填充为统一长度的张量
    并返回填充后的张量及各原始张量的长度信息。 
    """
    dev = v_list[0].device
    num_meshes = len(v_list)
    num_dim = 1 if len(v_list[0].shape) == 1 else v_list[0].shape[1]

    # 遍历输入列表，记录每个张量的长度（第一维大小）
    v_len_list = []
    for verts in v_list:
        v_len_list.append(verts.shape[0])

    # 确定需要填充的最大长度（所有张量中最长的那个）
    pad_len = max(v_len_list)
    dtype = v_list[0].dtype
    if num_dim == 1:
        padded_tensor = torch.zeros(num_meshes, pad_len, dtype=dtype)
    else:
        padded_tensor = torch.zeros(num_meshes, pad_len, num_dim, dtype=dtype)
    for idx, (verts, v_len) in enumerate(zip(v_list, v_len_list)):
        padded_tensor[idx, :v_len] = verts
    padded_tensor = padded_tensor.to(dev)
    v_len_list = torch.LongTensor(v_len_list).to(dev)

     # 返回填充后的张量和各原始张量的长度（张量形式）
    return padded_tensor, v_len_list