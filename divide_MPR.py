import os
import numpy as np
import json
import nibabel as nib
from matplotlib import cm
import matplotlib.pyplot as plt
import re
from scipy.interpolate import interpn
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
import cv2
import skimage
import scipy.io as scio
from utils.tools import load_centerline, find_plaque
from vis.vis import vis_MPR
from MPR import colored_mask
def divide_MPR(fp, nii_c, artery_c, start, end, lines_fp):
    with open(lines_fp, 'r') as f:
        lines = f.readlines()
    f.close()

    lines_fp = lines_fp.split('.txt')[0] + '_divide' + '.txt'

    nii_c_artery_c_list = [int(re.findall('\d+', i)[-1]) for i in lines if i.split('/')[0] == str(nii_c)]
    max_artery_c = np.max(nii_c_artery_c_list)

    artery_c_t1 = str(max_artery_c + 1)
    artery_c_t2 = str(max_artery_c + 2)

    MPR_I = scio.loadmat(os.path.join(fp, 'mat/image', '{}_{}.mat'.format(nii_c, artery_c)))['MPR_I']
    MPR_MASK = scio.loadmat(os.path.join(fp, 'mat/mask', '{}_{}.mat'.format(nii_c, artery_c)))['MPR_MASK']
    MPR_VEC = scio.loadmat(os.path.join(fp, 'mat/mask_vec', '{}_{}.mat'.format(nii_c, artery_c)))['MPR_MASK_VEC']

    MPR_I_1, MPR_I_2 = MPR_I[start:end, :, :], MPR_I[end:, :, :]
    MPR_MASK_1, MPR_MASK_2 = MPR_MASK[start:end, :, :], MPR_MASK[end:, :, :]
    MPR_VEC_1, MPR_VEC_2 = MPR_VEC[start:end], MPR_MASK[end:]

    scio.savemat(os.path.join(fp, 'mat/image', '{}_{}.mat'.format(nii_c, artery_c_t1)), {'MPR_I': MPR_I_1})
    scio.savemat(os.path.join(fp, 'mat/mask', '{}_{}.mat'.format(nii_c, artery_c_t1)), {'MPR_MASK': MPR_MASK_1})
    scio.savemat(os.path.join(fp, 'mat/mask_vec', '{}_{}.mat'.format(nii_c, artery_c_t1)), {'MPR_VEC': MPR_VEC_1})

    scio.savemat(os.path.join(fp, 'mat/image', '{}_{}.mat'.format(nii_c, artery_c_t2)), {'MPR_I': MPR_I_2})
    scio.savemat(os.path.join(fp, 'mat/mask', '{}_{}.mat'.format(nii_c, artery_c_t2)), {'MPR_MASK': MPR_MASK_2})
    scio.savemat(os.path.join(fp, 'mat/mask_vec', '{}_{}.mat'.format(nii_c, artery_c_t2)), {'MPR_VEC': MPR_VEC_2})


    with open(lines_fp, 'a') as f:
        f.writelines('{}/{}.txt\n'.format(nii_c, artery_c_t1))
        f.writelines('{}/{}.txt\n'.format(nii_c, artery_c_t2))
    f.close()




if __name__ == '__main__':
    divide_MPR('../DATA/22_oct_Coronary_Artery_CT/MPR_2/',
               nii_c=13, artery_c=38,
               start=0, end=327,
               lines_fp='../DATA/22_oct_Coronary_Artery_CT/plaque/lines.txt')
