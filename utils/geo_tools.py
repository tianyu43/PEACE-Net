import os
import re
from osgeo import gdal
import numpy as np


START_V_I = 0
START_H_I = 0    #  
SIDE_LEN = 10000



def natural_sort_key(s):
    """
    按文件名的结构排序，即依次比较文件名的非数字和数字部分
    """
    sub_strings = re.split(r'(\d+)', s)
    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    return sub_strings


def _assert_suffix_match(suffix, path):
    assert re.search(r"\.{}$".format(suffix), path), "suffix mismatch"

def make_parent_dir(filepath):
    parent_path = os.path.dirname(filepath)
    if not os.path.isdir(parent_path):
        try:
            os.mkdir(parent_path)
        except FileNotFoundError:
            make_parent_dir(parent_path)
            os.mkdir(parent_path)
        print("[INFO] Make new directory: '{}'".format(parent_path))
        
def save_to_npy(data, path):
    _assert_suffix_match("npy", path)
    make_parent_dir(path)
    np.save(path, data)
    print("[INFO] Save as npy: '{}'".format(path))
    
    
    
    
    
    
def read_RS_tiffs(forderpath, file_format, file_bands, file_year, dtype=np.float32):
    files = [
        os.path.join(forderpath, f) 
        for f in os.listdir(forderpath) 
        if file_bands in f and file_year in f
        and f.endswith(file_format)
        ]
    
    files = sorted(files, key=natural_sort_key)
    
    n_t = len(files)
    n_bands = 10
    
    size = SIDE_LEN * SIDE_LEN
    output_array = np.empty((size, n_t, n_bands), dtype=dtype)
    
    xoff, yoff = START_H_I, START_V_I
    xsize = ysize = SIDE_LEN
    
    for t, fp in enumerate(files):
        dataset = gdal.Open(fp)
        i = 0
        if dataset is None:
            raise RuntimeError(f"Could not open {fp}")
        else:
            print(f'Successfully opened {fp}')


        arr = dataset.ReadAsArray(xoff, yoff, xsize, ysize)
        np.nan_to_num(arr, copy=False, nan=0.0)
        dataset = None
        
        arr2 = arr.reshape(n_bands, -1).T/10000
        
        output_array[:, t, :] = arr2.astype(dtype, copy=False)
        
        #print(arr2.shape)
        
        del arr, arr2

    return output_array








def read_label_tiffs(forderpath,file_format,file_bands,file_year):
    files = [
        os.path.join(forderpath, f) 
        for f in os.listdir(forderpath) 
        if file_bands in f and file_year in f
        and f.endswith(file_format)
        ]
    
    files = sorted(files, key=natural_sort_key)

    #print(files)
    array = []
    for file_path in files:
        dataset = gdal.Open(file_path)
        i = 0
        if dataset is None:
            print(f'Could not open {file_path}')
        else:
            print(f'Successfully opened {file_path}')
            i+1
        
        bands = dataset.ReadAsArray()[START_V_I:START_V_I+SIDE_LEN, START_H_I:START_H_I+SIDE_LEN]
        
        #bands = dataset.ReadAsArray()
        array.append(bands)
        dataset = None
        
    output_array = np.stack(array, axis=2)           #(W, H, B)
    #output_array = output_array.transpose(2,3,1,0)   
    output_array = output_array.reshape(-1,len(files))       #(size, B)
    
    return output_array
