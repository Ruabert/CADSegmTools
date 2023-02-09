"""
Created on 01/09/2023

@author: ZhangRuarua
"""

from vis.vis import plot_mask
from utils.tools import *
import nibabel as nib
import os

if __name__ == '__main__':
    nii_n = 2
    cl_n = 22

    plaque_label = 254
    step = 2

    thresh = 5

    data_root_fp = '../DATA/22_oct_Coronary_Artery_CT/nii/'

    mask_fp = os.path.join(data_root_fp, 'mask/{}.nii.gz'.format(nii_n))
    txt_fp = os.path.join(data_root_fp, 'vis/{}/each_label/{}_{}.txt'.format(nii_n, nii_n, cl_n))

    mask = nib.load(mask_fp).get_fdata().transpose(2, 1, 0)  # 为了符合一般习惯，transpose一下 z,y,x
    mask = mask.astype(int)

    plaque_mask = find_plaque(mask)
    # coordinates, coordinate_range, bbox = find_bbox(plaque_mask, label='254')
    # Zmin, Zmax = coordinate_range[0, 0], coordinate_range[0, 1]
    # Ymin, Ymax = coordinate_range[1, 0], coordinate_range[1, 1]
    # Xmin, Xmax = coordinate_range[2, 0], coordinate_range[2, 1]
    cl_points = load_centerline(txt_fp)

    l = cl_points.shape[0]

    # cross_plaque = 0
    for i in range(0, l, step):
        z = int(cl_points[i, 0])
        y = int(cl_points[i, 1])
        x = int(cl_points[i, 2])

        # bool1 = z <= Zmax and z >= Zmin
        # bool2 = y <= Ymax and y >= Ymin
        # bool3 = x <= Xmax and x >= Xmin
        #
        # if bool1 and bool2 and bool3:
        #     cross_plaque = cross_plaque + 1

        plaque_mask[z, y, x] = cl_n

    # if cross_plaque>=thresh:
    #     print("cross plaque")

    plot_mask(plaque_mask, plaque=True, save_fp=None)
