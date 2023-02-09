"""
Created on 01/09/2023

@author: ZhangRuarua
"""

import os.path
import SimpleITK as sitk
import skimage.morphology
import numpy as np
from tqdm import tqdm
from utils.static import NBH_Creator
import nibabel as nib
import pandas as pd
# from vis.vis import plot_mask
import re

# 从txt读取中心线点集
def load_centerline(fp):
    with open(fp, 'r') as f:
        content = f.readlines()
        points = [x.split('\n')[0].split(',') for x in content]
        points = [[int(c) for c in p] for p in points]

    return np.array(points)

# 提取只有管腔中心线的mask
def find_centerline(mask):
    # 提取只有管腔中心线的mask
    mask = np.where(mask == 1, 1, 0)

    coordinate, coordinate_range, bbox = find_bbox(mask, label='1')

    cl_mask = skimage.morphology.skeletonize_3d(mask)
    cl_mask = np.where(cl_mask > 0, 1, 0)

    return cl_mask

# 提取只有斑块的mask
def find_plaque(mask, t_v=254):
    plaque = np.where(mask == 254, t_v, 0)

    return plaque

# 提取包围盒：
# CT mask中，label与 bg的样本量极其不平衡，这个函数用来提取 mask中能包围某个 label所有 coordinate的最小 bbox
def find_bbox(mask, label = '1', cl = False):

    # 是否需要做中心线提取
    if int(label) and cl:
        print('extracting centerline')
        mask = find_centerline(mask)

    z, y, x = np.where(mask == int(label))
    zmin, zmax = np.min(z), np.max(z)
    ymin, ymax = np.min(y), np.max(y)
    xmin, xmax = np.min(x), np.max(x)
    bbox = mask[zmin:zmax, ymin:ymax, xmin:xmax]

    # 返回坐标区间
    coordinate_range = np.array([(zmin, zmax), (ymin, ymax), (xmin, xmax)])

    # 提取中心线矢量有用 返回z,y,x
    coordinates = []
    for i in range(len(z)):
        coordinates.append(np.array([z[i], y[i], x[i]]))

    coordinates = np.array(coordinates)
    return coordinates, coordinate_range, bbox

# 找到血管末梢点和交叉点
def find_start_cross_points(mask, coordinate):
    MAXZ, MAXY, MAXX = mask.shape
    NBH = NBH_Creator()
    start_points = []
    cross_points = []
    for z, y, x in coordinate:
        count = 0

        for i in range(len(NBH)):
            tmp_z = z + NBH[i][0]
            tmp_y = y + NBH[i][1]
            tmp_x = x + NBH[i][2]

            if z < 0 or y < 0 or x < 0 or \
                tmp_z >= MAXZ or tmp_y >= MAXY or tmp_x >= MAXX:
                continue

            if mask[z, y, x] and mask[tmp_z, tmp_y, tmp_x]:
                count = count + 1

        if count == 1:
            start_points.append((z, y, x))

        if count > 2:
            cross_points.append((z, y, x))

    return start_points, cross_points

def Parse_mask_to_txt(points_dict, fp):

    # with open("test.txt", "w") as f:
    for k, v in points_dict.items():
        name = str(k + 1) + '.txt'
        txt_fp = os.path.join(fp, name)
        with open(txt_fp, "w") as f:
            lines = []
            for coordinate in v:
                if coordinate[0] == 0 and coordinate[1] == 0 and coordinate[2] == 0:
                    continue
                coordinate_str = str(coordinate[0]) + ',' + str(coordinate[1]) + ',' + str(coordinate[2]) + '\n'
                lines.append(coordinate_str)
            f.writelines(lines)

def Parse_csv_to_mat(csv_fp, label_list):
    df = pd.read_csv(csv_fp)
    line_list = []
    # 找到我们需要的分支所对应的df的index
    for label in label_list:
        p_index = -1
        # 拿到该分支的 index
        for index, l in enumerate(df.iloc[0, 1:]):
            if int(float(l)) == label:
                p_index = index + 1

        # 拿到这个index对应的点集
        points_str = ''.join(df.iloc[1, p_index])[1:-1]
        points = re.findall('\d+, \d+, \d+', points_str)

        # 转化下类型
        points_list = []
        for p in points:
            zyx = re.findall('\d+', p)
            z = int(zyx[0])
            y = int(zyx[1])
            x = int(zyx[2])
            points_list.append(np.array([x + 1, y + 1, z + 1], dtype=int))

        # 获得点集
        points_list = np.array(points_list, dtype=int)
        line_list.append(points_list)
    line_list.append(np.array([], dtype=float))
    line_list = np.array([line_list])
    return line_list

def A_to_I(array, Origin, Spacing, Direction):
    # 将 array 转换为 Image
    img = sitk.GetImageFromArray(array)

    img.SetOrigin(Origin)
    img.SetSpacing(Spacing)
    img.SetDirection(Direction)

    return img

def write_to_nii(data, root_fp):
    for i in tqdm(range(data.__len__())):
        sample = data[i]
        img = A_to_I(sample['img'], Origin=sample['Origin'], Spacing=sample['Spacing'], Direction=sample['Direction'])
        mask = A_to_I(sample['mask'], Origin=sample['Origin'], Spacing=sample['Spacing'], Direction=sample['Direction'])

        # cl_mask = A_to_I(sample['cl'], Origin=sample['Origin'], Spacing=sample['Spacing'], Direction=sample['Direction'])

        sitk.WriteImage(img, os.path.join(root_fp, 'nii/image', sample['code'] + '.nii.gz'))
        sitk.WriteImage(mask, os.path.join(root_fp, 'nii/mask', sample['code'] + '.nii.gz'))
        # sitk.WriteImage(cl_mask, os.path.join(root_fp, 'nii/cl_mask', sample['code'] + '.nii.gz'))


def statistics_points_num_of_each_line(fp):
    fp_l = [os.path.join(fp, i) for i in sorted(os.listdir(fp), key=lambda x:int(x))]
    fp_l = [os.path.join(i, 'each_label') for i in fp_l]
    for fp in fp_l:

        count = 0
        max = 0
        min = 100000

        txt_l = [i for i in os.listdir(fp) if i.endswith('.txt')]
        txt_l = sorted(txt_l, key=lambda x: int(x.split('.txt')[0].split('_')[-1]))
        txt_l = [os.path.join(fp, i) for i in txt_l]

        num = len(txt_l)

        for txt in txt_l:
            count_c = len(open(txt,'rU').readlines())
            if count_c < min:
                min = count_c
            if count_c > max:
                max = count_c
            count = count + count_c

        avg = count / num

        print('{}'.format(fp.split('\\')[0].split('/')[-1]))
        print('avg: {}, min: {}, max: {}'.format(avg, min, max))
