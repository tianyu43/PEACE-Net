import os
import re
from osgeo import gdal
import numpy as np






def natural_sort_key(s):
    """
    natural string sorting.
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
    
    

START_V_I = 0
START_H_I = 0    
SIDE_LEN  = 5000

### 5009, 6777
    
    
def read_RS_tiffs(forderpath, file_format, file_bands, dtype=np.float32):
    
    files = [
        os.path.join(forderpath, f) 
        for f in os.listdir(forderpath) 
        if file_bands in f 
        and f.endswith(file_format)
        ]
    
    files = sorted(files, key=natural_sort_key)
    print(files)
    

    
    size = SIDE_LEN * SIDE_LEN
    size = 5009 * 6777
    output_array = np.empty((size, 22, 10), dtype=dtype)
    
    xoff, yoff = START_H_I, START_V_I
    xsize = ysize = SIDE_LEN
    
    for b, fp in enumerate(files):
        dataset = gdal.Open(fp)
        i = 0
        if dataset is None:
            raise RuntimeError(f"Could not open {fp}")
        else:
            print(f'Successfully opened {fp}')


        arr = dataset.ReadAsArray(xoff, yoff, xsize, ysize)
        #arr = dataset.ReadAsArray()
        print(arr.shape)
        np.nan_to_num(arr, copy=False, nan=0.0)
        dataset = None
        
        arr2 = arr.reshape(22, -1).T / 10000.0

        output_array[:, :, b] = arr2.astype(dtype, copy=False)
        
        #print(arr2.shape)
        
        del arr, arr2

    return output_array








def read_label_tiffs(forderpath,file_format,file_bands):
    files = [
        os.path.join(forderpath, f) 
        for f in os.listdir(forderpath) 
        if file_bands in f  
        and f.endswith(file_format)
        ]
    
    files = sorted(files, key=natural_sort_key)

    xoff, yoff = START_H_I, START_V_I
    xsize = ysize = SIDE_LEN
    
    #print(files)
    array = []
    for file_path in files:
        dataset = gdal.Open(file_path)
        if dataset is None:
            print(f'Could not open {file_path}')
        else:
            print(f'Successfully opened {file_path}')
        
        #label = dataset.ReadAsArray() 
        label = dataset.ReadAsArray(xoff, yoff, xsize, ysize)

        dataset = None

    output_array = label.reshape(-1)
    
    return output_array
