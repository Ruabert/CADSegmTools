"""
Created on 02/07/2023

@author: ZhangRuarua

"""

import os
import numpy as np
import nibabel as nib
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
import cv2
import skimage
import scipy.io as scio
import re
import json
from utils.tools import find_start_cross_points, find_plaque, find_bbox, find_centerline, Parse_mask_to_txt
from utils.region_crowing import cal_artery_R, Region_Growing2, Mask_Filter
from vis.vis import plot_mask
from extract_plaque_info import find_lines_for_MPR
import random

# 尝试重新写一下整个流程
if __name__ == '__main__':
    # 需要的路径
    with open('parameters.json') as f:
        parameters = json.load(f)

    CT_data_root = parameters['data_path']['CT_data_root']
    mask_data_fp = os.path.join(CT_data_root, parameters['data_path']['mask_path'])

    # 所有病例的mask路径
    patient_mask_fp_list = [os.path.join(mask_data_fp, i) for i in
                       sorted(os.listdir(mask_data_fp),
                              key=lambda x:int(re.findall('\d+', x)[0]))]

    print('patient num: {}'.format(len(patient_mask_fp_list)))

    # 逐个病例处理
    for patient_mask_fp in tqdm(patient_mask_fp_list, ncols=50):
        nii_c = re.findall('\d+', patient_mask_fp.split('\\')[-1])[0]

        if os.path.exists(os.path.join(CT_data_root, parameters['data_path']['cl_save_path'], nii_c)) \
                and len(os.listdir(os.path.join(CT_data_root, parameters['data_path']['cl_save_path'], nii_c))):
            continue

        # 读取mask
        patient_mask = nib.load(patient_mask_fp).get_fdata().transpose(2, 1, 0)  # 为了符合一般习惯，transpose一下 z,y,x
        patient_mask = patient_mask.astype(int)

        ### 获得中心线点集
        cl_mask = find_centerline(patient_mask)
        coordinates, _, _ = find_bbox(cl_mask)
        cl_p_num = coordinates.shape[0]

        ### 计算每个中心点在MASK上面的转化距离
        dst_path = os.path.join(CT_data_root, 'tmpfile')
        if os.path.exists(os.path.join(dst_path, '{}_dst.npy'.format(nii_c))):
            c_dst_dict = np.load(os.path.join(dst_path, '{}_dst.npy'.format(nii_c)), allow_pickle=True).item()
        else:
            c_dst_dict = cal_artery_R(coordinates, patient_mask)
            np.save(os.path.join(dst_path, '{}_dst.npy'.format(nii_c)), c_dst_dict)

        ### 做区域生长
        # 提取斑块
        plaque_mask = find_plaque(patient_mask)
        # 提取一些空间信息
        coordinate, _, _ = find_bbox(cl_mask, cl=False)
        # 提取冠状动脉的末梢，作为 start_points
        start_points, cross_points = find_start_cross_points(cl_mask, coordinate)
        # 生长
        seed_mask, points_dict = Region_Growing2(cl_mask, start_points, cross_points, coordinates, c_dst_dict)
        seed_mask, points_dict = Mask_Filter(seed_mask, points_dict, 30)
        # plot_mask(seed_mask, plaque=False, start_points=start_points, save_fp=None)

        print('{}`s RC segement rate: {}'.format(nii_c, len(np.where(seed_mask>0)[0]) / cl_p_num))

        # 构建中心线存储文件
        cl_save_path = os.path.join(CT_data_root, parameters["data_path"]["cl_save_path"])
        cl_save_path = os.path.join(cl_save_path, nii_c)

        if not os.path.exists(cl_save_path):
            os.makedirs(cl_save_path)

        ###  存储中心线文件为 .txt
        Parse_mask_to_txt(points_dict, cl_save_path)

        ### 中心线可视化存储
        cl_vis_path = os.path.join(CT_data_root, parameters["data_path"]["cl_vis_path"])
        cl_vis_path = os.path.join(cl_vis_path, nii_c)

        if not os.path.exists(cl_vis_path):
            os.makedirs(cl_vis_path)

        # 逐个分支绘图
        print('draw centerline ...')

        for l in tqdm(np.unique(seed_mask), ncols=50):

            if l == 0:
                continue
            # 绘图
            vis_mask = cl_mask + np.where(seed_mask == l, 150, 0) + plaque_mask
            l_vis_path = os.path.join(cl_vis_path, '{}.png'.format(int(l + 1)))
            plot_mask(vis_mask, plaque=True, save_fp=l_vis_path)

        ### 根据斑块信息，确定通过斑块的中心线
        plaque_info_path = os.path.join(CT_data_root, parameters["data_path"]["plaque_info_path"])
        lines_path = os.path.join(plaque_info_path, 'lines.txt')
        plaque_info_path = os.path.join(plaque_info_path, nii_c)

        if not os.path.exists(plaque_info_path):
            os.makedirs(plaque_info_path)

        find_lines_for_MPR(patient_mask_fp, plaque_info_path, cl_save_path, lines_path)





