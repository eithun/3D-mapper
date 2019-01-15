import scipy.misc
import os
import numpy as np


# Covery npy file to tiff stack
def npy_to_tiff(f, out_location, dim=-1, c = None):
    V = np.load(f)
    voxel_to_tiff(V, out_location, dim, c)

# Convert loaded numpy matrix 
def voxel_to_tiff(V, out_location, dim=-1, c = None):
    # Create folder
    if not os.path.exists(out_location):
        os.makedirs(out_location)
    
    # Replace nan values with zero
    V[np.isnan(V)] = 0


    if c == None:
        cmin, cmax = np.min(V),np.max(V)
    else:
        cmin, cmax = c
        
    sliceN = V.shape[dim]
    fill = len(str(sliceN))
    
    for i in range(sliceN):
        im = scipy.misc.toimage(V[:,:,i], cmin=cmin, cmax=cmax)
        scipy.misc.imsave(out_location + str(i).zfill(fill) + ".tiff", im)
    
    
