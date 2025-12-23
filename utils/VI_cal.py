import numpy as np

# EVI 
G, C1, C2 = 2.5, 6, 7.5

def calculate_ndvi(data, start_time=0, end_time=None):
    if end_time is None:
        end_time = data.shape[1]

    Red = data[:, start_time:end_time, 2]
    Nir = data[:, start_time:end_time, 6]  #

    NDVI = (Nir - Red) / (Nir + Red + 1e-6)
    return NDVI


def calculate_gwcci(data, start_time=0, end_time=None):

    if end_time is None:
        end_time = data.shape[1]
    
    Red   =  data[:, start_time:end_time, 2]  
    Nir   =  data[:, start_time:end_time, 7] 
    Swir1 =  data[:, start_time:end_time, 8] 
    
    NDVI = (Nir - Red) / (Nir + Red)
    
    gwcci = NDVI * Swir1 
    return gwcci



def calculate_gwcci_2(data, start_time=0, end_time=None):

    if end_time is None:
        end_time = data.shape[1]

    Blue  =  data[:, start_time:end_time, 0] 
    Red   =  data[:, start_time:end_time, 2] 
    Nir   =  data[:, start_time:end_time, 7]
    Swir1 =  data[:, start_time:end_time, 8]
    
    EVI = (G * (Nir - Red) / (Nir + C1 * Red - C2 * Blue + 1 + 1e-3))
    NDSVI = ((Swir1 - Red) / (Swir1 + Red))
    
    gwcci2 = EVI * NDSVI
    return gwcci2



def calculate_SMCI(data, start_time=0, end_time=None):

    if end_time is None:
        end_time = data.shape[1]
    data = data + 0.1
    Blue =     data[:, start_time:end_time, 0] 
    Green =    data[:, start_time:end_time, 1]  
    Red  =     data[:, start_time:end_time, 2]  
    Re1 =      data[:, start_time:end_time, 3]
    Re2 =      data[:, start_time:end_time, 4]
    Re3 =      data[:, start_time:end_time, 5]
    Re4 =      data[:, start_time:end_time, 6]
    Nir  =     data[:, start_time:end_time, 7]
    Swir1 =    data[:, start_time:end_time, 8]
    
    EVI  = (G * (Nir - Red) / (Nir + C1 * Red - C2 * Blue + 1 + 1e-3))
    #GCVI = np.where(Green != 0, (Nir / Green) - 1, 0.0)
    GCVI =  (Nir / Green) - 1
    SMCI = (Swir1 + Nir + Re2 + Re3 + Re4) * EVI * GCVI
    return SMCI




def calculate_gcc(data, start_time=0, end_time=None):

    if end_time is None:
        end_time = data.shape[1]
        
    Blue =     data[:, start_time:end_time, 0] 
    Green =    data[:, start_time:end_time, 1]  
    Red  =     data[:, start_time:end_time, 2]  
    Re1 =      data[:, start_time:end_time, 3]
    Re2 =      data[:, start_time:end_time, 4]
    Re3 =      data[:, start_time:end_time, 5]
    Re4 =      data[:, start_time:end_time, 6]
    Nir  =     data[:, start_time:end_time, 7] 
    Swir1 =    data[:, start_time:end_time, 8]
    
    GCC = Green / (Blue + Red + Green)
    
    return GCC





def get_pseudo_label_with_threshold(data, vi='gwcci', 
                                    start_time=0, end_time=None, 
                                    thresholds=None,
                                    m = 'max'):
    """
    根据指定的植被指数（VI）计算伪标签。

    参数:
        data: ndarray, 输入数据，形状为 (N, T, B)
        vi: str, 植被指数类型，可选 'gwcci', 'gwcci_2', 'smci'
        start_time: int, 起始时间步
        end_time: int or None, 结束时间步（默认为数据末尾）
        thresholds: dict, 外部指定的阈值字典，格式如下:
            {
                'gwcci': (0.1, 0.05),
                'gwcci_2': (0.6, 0.5),
                'smci': (0.3, 0.2)
            }

    返回:
        pseudo_labels: ndarray, 伪标签数组，1 表示大豆，0 表示其他，-1 表示未确定
    """
    if end_time is None:
        end_time = data.shape[1]

    # 默认 VI 函数映射
    vi_functions = {
        'gwcci': calculate_gwcci,
        'gwcci_2': calculate_gwcci_2,
        'smci': calculate_SMCI
    }

    # 默认阈值
    default_thresholds = {
        'gwcci':   (0.17, 0.17),
        'gwcci_2': (0.56, 0.56),
        'smci':    (3.25, 3.25)
    }

    # 使用外部传入的 thresholds 或默认值
    thresholds = thresholds or default_thresholds

    if vi not in vi_functions or vi not in thresholds:
        raise ValueError(f"Invalid VI type or missing threshold: {vi}")

    # 计算 VI 时间序列值
    vi_ts = vi_functions[vi](data, start_time, end_time)

    # 获取阈值
    SOY_TH, OTHER_TH = thresholds[vi]

    # 取最大值作为代表值
    if m == 'max':
        vi_value = np.max(vi_ts, axis=1)
    elif m == 'mean':
        vi_value = np.mean(vi_ts, axis=1)
    else:
        raise ValueError("Invalid method. Use 'max' or 'mean'.")

    # 初始化伪标签为 -1
    pseudo_labels = np.full_like(vi_value, fill_value=-1, dtype=int)
    pseudo_labels[vi_value >= SOY_TH] = 1
    pseudo_labels[vi_value < OTHER_TH] = 0

    
    return pseudo_labels

