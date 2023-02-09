"""
Created on 01/09/2023

@author: ZhangRuarua
"""

import os
import skimage
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from utils.tools import find_centerline
import matplotlib.colors as mcolors
import random
import colorsys

from sklearn.cluster import KMeans

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def random_colors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors

def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)

# 绘制带有血管和斑块的散点图
def plot_mask(
        cl_mask,
        # root_fp='../DATA/nii', f_mode='.nii.gz', code='10',
        plaque=True, start_points=None, cross_points=None, save_fp=None, only_one_label=-1):

    masks = []
    labels = np.unique(cl_mask)

    # print(labels)
    ax = plt.figure().add_subplot(111, projection='3d')

    colors = list(map(lambda x: color(tuple(x)), random_colors(len(labels))))

    for i, n in enumerate(labels):
        if n == 0:
            continue
        if n == 254 and not plaque:
            continue
        if only_one_label > 0 and n != only_one_label and n != 254:
            continue

        l_n_mask = np.where(cl_mask == n, n, 0)  # 提取标签为 n的绘制散点图

        # 绘制血管
        axis = np.argwhere(l_n_mask == n)
        z = axis[:, 0]
        y = axis[:, 1]
        x = axis[:, 2]


        if n == 254:
            ax.scatter(y, x, z, c='k', marker='.', s=1, alpha=0.3)
        else:
            ax.scatter(y, x, z, c=colors[i], marker='.', s=1, alpha=0.5)


    if start_points is not None:
        for start_point in start_points:
            ax.scatter(start_point[1], start_point[2], start_point[0], c='b', marker='.')

    if cross_points is not None:
        for cross_point in cross_points:
            ax.scatter(cross_point[1], cross_point[2], cross_point[0], c='g', marker='.')

    ax.set_xlabel('y Label')
    ax.set_ylabel('x Label')
    ax.set_zlabel('z Label')

    if save_fp is not None:
        plt.savefig(save_fp)
        plt.close()
    else:
        plt.show()
        plt.close()

def draw_points(P_list, y_p):
    Z = [P[0] for P in P_list]
    Y = [P[1] for P in P_list]
    X = [P[2] for P in P_list]


    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(Y, X, Z, c=y_p, marker='*')
    plt.show()

if __name__ == '__main__':
    data_root_fp = '../../DATA/22_oct_Coronary_Artery_CT/nii/'

    nii_n = 10
    mask_fp = os.path.join(data_root_fp, 'mask/{}.nii.gz'.format(nii_n))


    mask = nib.load(mask_fp).get_fdata().transpose(2, 1, 0)  # 为了符合一般习惯，transpose一下 z,y,x
    mask = mask.astype(int)


    mask = find_centerline(mask)

    plot_mask(mask, plaque=False)

