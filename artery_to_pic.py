"""
Created on 01/09/2023

@author: ZhangRuarua
"""
import os
import numpy as np
import re
from tqdm import tqdm
from vis.vis import plot_mask, draw_points
from utils.tools import find_centerline, find_bbox, find_start_cross_points, find_plaque, Parse_mask_to_txt
from utils.region_crowing import Region_Growing, Mask_Filter
import nibabel as nib


if __name__ == '__main__':
    nii_mask_root = '../DATA/22_oct_Coronary_Artery_CT/nii/mask/'
    # debug_root = ../DATA/22_oct_Coronary_Artery_CT/TRY/mask/'

    do_root = nii_mask_root

    # 读取排好序的文件列表
    fp_list = sorted(os.listdir(do_root), key=lambda fp: int(re.findall('\d+', fp)[-1]))

    # 逐个处理病例
    for mask_fp in tqdm(fp_list):

        # 需要处理的病例路径
        mask_fp = do_root + mask_fp
        sp_vis = False
        cp_vis = False
        plaque_vis = True

        f_c = re.findall('\d+', mask_fp)[-1]

        save_fp = '/'.join(do_root.split('/')[:-2])+'/vis'
        save_fp = save_fp + '/' + f_c + '/'
        try:
            os.makedirs(save_fp)
        except:
            pass

        if int(f_c) == 6:
            continue

        mask = nib.load(mask_fp).get_fdata().transpose(2, 1, 0)  # 为了符合一般习惯，transpose一下 z,y,x
        mask = mask.astype(int)

        # 提取中心线
        cl_mask = find_centerline(mask)

        # 提取斑块
        plaque_mask = find_plaque(mask)

        # 提取一些空间信息
        coordinate, _, _ = find_bbox(cl_mask, cl=False)

        # 提取冠状动脉的末梢，作为 start_points
        start_points, cross_points = find_start_cross_points(cl_mask, coordinate)

        # 生长
        seed_mask, points_dict = Region_Growing(cl_mask, start_points, cross_points)

        # 目前的生长算法，会导致很多分支丢失，所以说与原始中心线mask相加，把丢失的点补充回来
        seed_mask = seed_mask + cl_mask

        # 保存过滤后的蒙版
        seed_mask, points_dict = Mask_Filter(seed_mask, points_dict, thresh=30)

        print('first Region Grow completion')

        # 处理遗漏的点
        one_mask = np.where(seed_mask == 1, 1, 0)
        coordinate, _, _ = find_bbox(one_mask, cl=False)
        start_points, cross_points = find_start_cross_points(one_mask, coordinate)
        one_mask, one_points_dict = Region_Growing(one_mask, start_points, cross_points)
        one_mask, one_points_dict = Mask_Filter(one_mask, one_points_dict, thresh=20)
        # plot_mask(one_mask, plaque=False, start_points=start_points, save_fp=None)

        print('second Region Grow completion')

        # 合并两个points_dict
        keys = list(points_dict.keys())
        last_key = keys[-1]
        for k, v in one_points_dict.items():
            points_dict[k+last_key] = one_points_dict[k]

        one_mask = one_mask + last_key
        one_mask = one_mask - np.where(one_mask == last_key, last_key, 0)
        seed_mask = seed_mask + one_mask
        # plot_mask(seed_mask, plaque=False, save_fp=None)

        # 绘图中加入 plaque
        vis_mask2 = seed_mask + plaque_mask
        save_fp2 = save_fp + '{}_with_plaque.png'
        plot_mask(vis_mask2, plaque=True, save_fp=save_fp2.format(f_c))

        # 逐个分支绘图
        for l in tqdm(np.unique(vis_mask2)):
            if l == 0:
                continue

            try:
                os.makedirs(save_fp + 'each_label/')
            except: pass

            save_fp3 = save_fp + 'each_label/' + '{}_{}.png'.format(f_c, int(l))
            plot_mask(vis_mask2, plaque=True, save_fp=save_fp3, only_one_label=l)

        # 保存不同label的点集
        save_txt_fp = save_fp + 'each_label/{}_'.format(f_c)
        Parse_mask_to_txt(points_dict, save_txt_fp)













