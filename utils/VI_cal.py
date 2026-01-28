import numpy as np


def gaussian_kernel(size, sigma=1):
    x = np.arange(-size//2 + 1, size//2 + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

window_size = 5
sigma = 1.0
kernel = gaussian_kernel(window_size, sigma)



# EVI 
G, C1, C2 = 2.5, 6, 7.5

def calculate_ndvi(data, start_time=0, end_time=None):
    if end_time is None:
        end_time = data.shape[1]

    Red = data[:, start_time:end_time, 2]
    Nir = data[:, start_time:end_time, 7]

    NDVI = (Nir - Red) / (Nir + Red + 1e-6)
    del Red, Nir

    '''
    NDVI_s = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode='same'), axis=1, arr=NDVI
        )
    del NDVI
    '''
    
    return NDVI


def calculate_gwcci(data, start_time=0, end_time=None):

    if end_time is None:
        end_time = data.shape[1]
    
    Red   =  data[:, start_time:end_time, 2]  
    #Nir   =  data[:, start_time:end_time, 7] 
    Nir   =  data[:, start_time:end_time, 6]     
    Swir1 =  data[:, start_time:end_time, 8] 
    
    NDVI = (Nir - Red) / (Nir + Red + 1e-6)
    gwcci = NDVI * Swir1 
    del NDVI, Red, Nir, Swir1

    '''
    gwcci_s = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode='same'), axis=1, arr=gwcci
        )
    del gwcci
    '''
    
    return gwcci



def calculate_gwcci_2(data, start_time=0, end_time=None):

    if end_time is None:
        end_time = data.shape[1]

    Blue  =  data[:, start_time:end_time, 0] 
    Red   =  data[:, start_time:end_time, 2] 
    #Nir   =  data[:, start_time:end_time, 7]
    Nir   =  data[:, start_time:end_time, 6]
    Swir1 =  data[:, start_time:end_time, 8]
    
    EVI = (G * (Nir - Red) / (Nir + (C1 * Red) - (C2 * Blue) + 1 + 1e-6))
    NDSVI = ((Swir1 - Red) / (Swir1 + Red + 1e-6))
    
    gwcci2 = EVI * NDSVI
    del EVI, NDSVI, Blue, Red, Nir, Swir1, 
    
    '''
    gwcci2_s = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode='same'), axis=1, arr=gwcci2
        )
    del gwcci2
    '''
    
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
    #Nir  =     data[:, start_time:end_time, 7]
    #Re4 =      data[:, start_time:end_time, 7]
    Nir  =     data[:, start_time:end_time, 6]
    

    Swir1 =    data[:, start_time:end_time, 8]
    
    EVI = (G * (Nir - Red) / (Nir + (C1 * Red) - (C2 * Blue) + 1 + 1e-6))
    #GCVI = np.where(Green != 0, (Nir / Green) - 1, 0.0)
    GCVI =  (Nir / Green) - 1
    SMCI = (Swir1 + Nir + Re2 + Re3 + Re4) * EVI * GCVI
    del EVI, GCVI, Blue, Green, Red, Re1, Re2, Re3, Re4, Nir, Swir1

    '''
    SMCI_s = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode='same'), axis=1, arr=SMCI
        )
    del SMCI  
    '''
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
    
    GCC = Green / (Blue + Red + Green + 1e-6)
    
    return GCC
