import numpy as np

def read_npz_file(file_path):
    """
    读取.npz文件并打印其中包含的数组信息及内容
    
    参数:
        file_path (str): .npz文件的路径
    """
    try:
        # 加载npz文件
        npz_data = np.load(file_path, allow_pickle=True)
        print(f"npz_data.type:{type(npz_data)}")
        
        # 获取文件中所有数组的名称
        array_names = npz_data.files
        print(f".npz文件中包含 {len(array_names)} 个数组: {array_names}\n")
        
        # 遍历并打印每个数组的信息
        for name in array_names:
            array_data = npz_data[name]
            print(f"数组名称: {name}")
            print(f"数组形状: {array_data.shape}")
            print(f"数组数据类型: {array_data.dtype}")
            print("数组前3个元素(如果是1D)或前2x2元素(如果是高维):")
            # 打印部分数据，避免输出过长
            if array_data.size > 0:
                print(array_data[:3] if array_data.ndim == 1 else array_data[:2, :2])
            print("-" * 50)
            
        # 关闭文件
        npz_data.close()
        
    except FileNotFoundError:
        print(f"错误: 未找到文件 {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")

def get_subject_from_npz(file_path):
    """
    从.npz文件中获取'subject'数组的内容
    
    参数:
        file_path (str): .npz文件的路径
    
    返回:
        numpy.ndarray: 'subject'数组的内容，如果不存在则返回None
    """
    try:
        # 加载.npz文件
        with np.load(file_path) as npz_data:
            # 检查'subject'数组是否存在
            if 'subject' in npz_data:
                subject_data = npz_data['subject']
                print("成功获取'subject'数组：")
                print(f"数组形状: {subject_data.shape}")
                print(f"数组数据类型: {subject_data.dtype}")
                print("数组内容:")
                print(subject_data)
                return subject_data
            else:
                print("错误：.npz文件中不存在'subject'数组")
                return None
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
        return None

def get_background_from_npz(file_path):
    """
    从.npz文件中获取'background'数组的内容
    
    参数:
        file_path (str): .npz文件的路径
    
    返回:
        numpy.ndarray: 'background'数组的内容，如果不存在则返回None
    """
    try:
        # 加载.npz文件，使用with语句自动关闭文件
        with np.load(file_path) as npz_data:
            # 检查'background'数组是否存在
            if 'background' in npz_data:
                background_data = npz_data['background']
                print("成功获取'background'数组：")
                print(f"数组形状: {background_data.shape}")
                print(f"数组数据类型: {background_data.dtype}")
                print("数组内容:")
                print(background_data)
                return background_data
            else:
                print("错误：.npz文件中不存在'background'数组")
                return None
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
        return None
    
if __name__ == "__main__":
    # 示例：替换为你的.npz文件路径
    npz_file_path = r"F:\Chrome下载目录\data.npz"  # 这里可以替换为实际的.npz文件路径
    read_npz_file(npz_file_path)
    # get_subject_from_npz(npz_file_path)
    # get_background_from_npz(npz_file_path)