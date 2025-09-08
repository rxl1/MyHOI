import numpy as np



def get_frame_align_data_format(key, ndata, max_nframes):
    if "beta" in key:
        final_data = np.zeros((ndata, max_nframes, 10), dtype=np.float32)
    elif key == "x_lhand" or key == "x_rhand":
        final_data = np.zeros((ndata, max_nframes, 3+16*6), dtype=np.float32)
    elif key == "j_lhand" or key == "j_rhand":
        final_data = np.zeros((ndata, max_nframes, 21, 3), dtype=np.float32)
    elif key == "obj_alpha":
        final_data = np.zeros((ndata, max_nframes), dtype=np.float32)
    elif key == "x_obj":
        final_data = np.zeros((ndata, max_nframes, 3+6), dtype=np.float32)
    elif key == "x_obj_angle":
        final_data = np.zeros((ndata, max_nframes, 1), dtype=np.float32)
    elif "org" in key:
        final_data = np.zeros((ndata, max_nframes, 3), dtype=np.float32)
    elif "idx" in key or "dist" in key:
        final_data = np.zeros((ndata, max_nframes), dtype=np.float32)
    return final_data

def align_frame(total_dict):
    """ 
    将不同样本的帧数据对齐到相同长度 
    通过填充零值实现维度统一，方便后续模型处理（模型通常要求输入数据维度一致
    """
    max_nframes = 0
    value = next(iter(total_dict.values())) # ndata, frames, _  
    for _value in value:
        nframes = len(_value)
        if nframes > max_nframes:  # 记录所有样本中帧数的最大值
            max_nframes = nframes
    ndata = len(value)

    final_dict = {}
    for key, value in total_dict.items():
        final_data = get_frame_align_data_format(key, ndata, max_nframes)    
        for i, data in enumerate(value):
            nframes = len(data)
            if nframes == 0:
                continue
            final_data[i, :nframes] = data
        final_dict[key] = final_data
    return final_dict

if __name__ == "__main__":
    dict = {'a':[1,5,9], 'b':[9,10,60], 'c':[5,7,6]}
    iter = next(iter(dict.values()))
    for _value in iter:
        print(f"{_value}")