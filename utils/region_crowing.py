"""
Created on 01/09/2023

@author: ZhangRuarua
"""

import numpy as np
import cv2
import nibabel as nib
from utils.tools import find_plaque, find_centerline, find_bbox
from vis.vis import plot_mask
from utils.static import NBH_Creator

def Cal_vec_angle(v1, v2):
    # 计算向量模
    l_v1 = np.sqrt(v1.dot(v1))
    l_v2 = np.sqrt(v2.dot(v2))

    # 计算向量的点积
    # 夹角余弦
    cos_ = v1.dot(v2) / (l_v1 * l_v2)

    # 弧度
    angle_h = np.arccos(cos_)
    angle_d = angle_h * 180 / np.pi

    return angle_d

# 区域生长法
def Region_Growing(mask, start_points, cross_points, thresh=0):
    # 区域生长的一些基本的属性
    MAXZ, MAXY, MAXX = mask.shape
    NBH = NBH_Creator()

    # 一些mask
    seed_mask = np.zeros(mask.shape)  # 一个空白的mask

    # sp, cp
    sp_list = start_points.copy()
    sp_labels = []

    for i, sp in enumerate(sp_list):
        sp_labels.append((i+1, sp))

    cp_list = cross_points.copy()

    # 入cp点的空间向量
    enter_cp_vec_dict = {}

    # 中心线集合
    points_dict = {}

    for label, sp in sp_labels:
        # 点集
        points = []

        # 区域生长点列表
        seed_list = []

        # 将sp加入到seed_list
        seed_list.append(sp)

        while(len(seed_list) > 0):
            # 获得当前生长点
            z, y, x = seed_list.pop(0)
            points.append(np.array([z, y, x], dtype=int))
            seed_mask[z, y, x] = label

            if (z, y, x) in cp_list:
                enter_cp_vec = enter_cp_vec_dict[(z, y, x)]
                assert enter_cp_vec is not None, "{}".format((z, y, x))
                out_cp_vec_dict = {}

                for i in range(len(NBH)):
                    tmp_z = z + NBH[i][0]
                    tmp_y = y + NBH[i][1]
                    tmp_x = x + NBH[i][2]

                    if mask[z, y, x] and mask[tmp_z, tmp_y, tmp_x] and seed_mask[tmp_z, tmp_y, tmp_x] == 0:
                        out_cp_vec = (tmp_z - z, tmp_y - y, tmp_x - x)
                        out_cp_vec_dict.update({(tmp_z, tmp_y, tmp_x): out_cp_vec})

                min_angle = 180
                op = (0, 0, 0)
                for out_p, out_cp_vec in out_cp_vec_dict.items():
                    angle = Cal_vec_angle(
                        v1=np.array(enter_cp_vec),
                        v2=np.array(out_cp_vec)
                    )

                    if np.abs(angle - 180) <= min_angle:
                        min_angle = np.abs(angle - 180)
                        op = out_p
                # print(min_angle)
                tmp_z, tmp_y, tmp_x = op

                if (tmp_z, tmp_y, tmp_x) in cp_list:
                    enter_cp_vec = (tmp_z - z, tmp_y - y, tmp_x - x)
                    enter_cp_vec_dict.update({(tmp_z, tmp_y, tmp_x): enter_cp_vec})

                seed_list.append((tmp_z, tmp_y, tmp_x))

                # seed_mask[tmp_z, tmp_y, tmp_x] = label
                # print('join {} to branch_{}'.format((tmp_z, tmp_y, tmp_x), label))


            else:
                for i in range(len(NBH)):
                    tmp_z = z + NBH[i][0]
                    tmp_y = y + NBH[i][1]
                    tmp_x = x + NBH[i][2]

                    # 空间限制
                    if tmp_z < 0 or tmp_y < 0 or tmp_x < 0 or \
                            tmp_z >= MAXZ or tmp_y >= MAXY or tmp_x >= MAXX:
                        continue

                    # 限制生长条件
                    # 如果在cl_mask上，当前点值为 1，且邻域点为 1，且在seed_mask未被标记, 可以对其进行生长
                    if mask[z, y, x] and mask[tmp_z, tmp_y, tmp_x] and seed_mask[tmp_z, tmp_y, tmp_x] == 0:

                        # # 在seed_maks上面标记它
                        # seed_mask[tmp_z, tmp_y, tmp_x] = label

                        if (tmp_z, tmp_y, tmp_x) in cp_list:
                            enter_cp_vec = (tmp_z - z, tmp_y - y, tmp_x - x)
                            enter_cp_vec_dict.update({(tmp_z, tmp_y, tmp_x): enter_cp_vec})

                        # 加入到生长
                        seed_list.append((tmp_z, tmp_y, tmp_x))
                        # print('join {} to branch_{}'.format((tmp_z, tmp_y, tmp_x), label))

        points_dict.update({label: points})

    seed_mask[0, 0, 0] = 0

    return seed_mask, points_dict

def Mask_Filter(mask, points_dict, thresh = 30):
    points_dict_c = points_dict.copy()
    mask = mask.copy()
    labels = np.unique(mask)

    for l in labels:
        if l == 0:
            continue
        if l == 254:
            continue

        z, y, x = np.where(mask == int(l))
        if len(z) < thresh:
            for i in range(len(z)):
                mask[z[i], y[i], x[i]] = 0

    for k, v in points_dict.items():
        if len(v) < thresh:
            points_dict_c.pop(k)

    return mask, points_dict_c