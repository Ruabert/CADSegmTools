"""
Created on 02/07/2023

@author: ZhangRuarua

"""

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


######## 预处理部分 ########

# 对原始图像进行灰度处理：灰度截断
def gray_preprocess(I, max, min):
    l_200_I = np.where(I< min, min, 0)
    mid_I = np.where(I >= min, I, 0)
    mid_I = np.where(mid_I < max, mid_I, 0)
    h_3000_I = np.where(I >= max, max, 0)

    out = l_200_I + mid_I + h_3000_I

    # print('I gray range is {} to {}'.format(np.min(I), np.max(I)))
    # print('I_pre gray range is {} to {}'.format(np.min(out), np.max(out)))

    return out

# 中心线预处理：插值
def cl_interpolate(points, N_min, N_max):

    n = len(points)
    if 3*n <= N_min:
        m = N_min
    elif 3*n >= N_max:
        m = N_max
    else:
        m = 3*n

    z_new = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), m)

    zy = points[:, [0, 1]]
    _, s_index = np.unique(zy[:, 0], return_index=True)
    y_new = make_interp_spline(zy[s_index, 0], zy[s_index, 1])(z_new)
    # y_new = [int(i) for i in y_new]

    zx = points[:, [0, 2]]
    _, s_index = np.unique(zx[:, 0], return_index=True)
    x_new = make_interp_spline(zx[s_index, 0], zx[s_index, 1])(z_new)
    # x_new = [int(i) for i in x_new]

    # z_new = [int(i) for i in z_new]


    return np.array([z_new, y_new, x_new]).T

# # 获得中心线的包围盒
# def get_cl_bbox(I, points):
#     z = [p[0] for p in points]
#     y = [p[1] for p in points]
#     x = [p[2] for p in points]
#
#     zmin, zmax = np.round(np.min(z) - 5), np.round(np.max(z) + 5)
#     ymin, ymax = np.round(np.min(y) - 5), np.round(np.max(y) + 5)
#     xmin, xmax = np.round(np.min(x) - 5), np.round(np.max(x) + 5)
#
#     I_cut = I[int(zmin):int(zmax), int(ymin):int(ymax), int(xmin):int(xmax)]
#
#     print("centerline bbox shape is {}".format(I_cut.shape))
#
#     z = np.array(range(int(zmin), int(zmax)))
#     y = np.array(range(int(ymin), int(ymax)))
#     x = np.array(range(int(xmin), int(xmax)))
#
#     bbox = [z, y, x]
#     print('bbox z range {} to {}'.format(zmin, zmax))
#     print('bbox y range {} to {}'.format(ymin, ymax))
#     print('bbox x range {} to {}'.format(xmin, xmax))
#
#     return bbox, I_cut

# 计算每个点的切向量
def cal_tangent_vector(points):
    z_t = np.gradient(points[:, 0])
    y_t = np.gradient(points[:, 1])
    x_t = np.gradient(points[:, 2])
    T = [[z_t[i], y_t[i], x_t[i]] for i in range(len(x_t))]
    # T = T / np.linalg.norm(T)
    T = np.array(T)
    return T

# 计算每个点的V_1向量
def cal_V_1(points, T):
    # V_1 = norm(Vref*T)
    Vref = points
    V_1 = np.cross(Vref, T)
    for i, V_1_i in enumerate(V_1):
        V_1_i = V_1_i / np.linalg.norm(V_1_i)
        V_1[i, :] = V_1_i
    V_1 = np.array(V_1)
    return V_1

# 计算每个点的V_2向量
def cal_V_2(V_1, T):
    # V_2 = norm(V_1*T)
    V_2 = np.cross(V_1, T)
    for i, V_2_i in enumerate(V_2):
        V_2_i = V_2_i / np.linalg.norm(V_2_i)
        V_2[i, :] = V_2_i
    V_2 = np.array(V_2)
    return V_2

# 构建待插值空间
def Pi_slice(points, slice_width, V_1, V_2):
    interpolate_space = []
    for i, pi in enumerate(points):
        # 平面基向量
        V_1_i = V_1[i,:]
        V_2_i = V_2[i,:]

        steps = np.arange(np.negative(slice_width/2), slice_width/2)

        # 构造V_1为row，V_2为col的面
        slice = np.zeros([slice_width, slice_width, 3])

        for i, k in enumerate(steps):
            for j, q in enumerate(steps):
                P_i = pi + k*V_1_i + q*V_2_i
                slice[i ,j] = np.array(P_i)
        interpolate_space.append(slice)

    return np.array(interpolate_space)

# 对待图像插值空间进行插值
def Interpolate(interpolate_space, bbox_points, I_cut, M = 'linear'):
    slice_width = interpolate_space.shape[1]
    out = np.zeros(interpolate_space.shape[:-1])
    out = out.reshape(out.shape[0], -1)
    for i in tqdm(range(interpolate_space.shape[0])):
        slice = interpolate_space[i]
        slice = slice.reshape(-1, 3)
        for j in range(slice.shape[0]):
            p = slice[j]

            # 边界控制
            if p[0] >= I_cut.shape[0] - 2:
                p[0] = I_cut.shape[0] - 3
            if p[1] >= I_cut.shape[1] - 2:
                p[1] = I_cut.shape[1] - 3
            if p[2] >= I_cut.shape[2] - 2:
                p[2] = I_cut.shape[2] - 3

            if p[0] < 0:
                p[0] = 0
            if p[1] < 0:
                p[1] = 0
            if p[2] < 0:
                p[2] = 0

            result = interpn(bbox_points, I_cut, p, method=M)
            out[i, j] = result

    return out.reshape(out.shape[0], slice_width, -1)

# 对待mask插值空间进行插值
def Interpolate_Mask(interpolate_space, bbox_points, I_cut, plaque_label = 254 ,M = 'nearest'):
    out = np.zeros([interpolate_space.shape[0], 1])

    for i in tqdm(range(interpolate_space.shape[0])):
        slice = interpolate_space[i]
        slice = slice.reshape(-1, 3)

        count = 0
        for j in range(slice.shape[0]):
            p = slice[j]

            # 边界控制
            if p[0] >= I_cut.shape[0] - 2:
                p[0] = I_cut.shape[0] - 3
            if p[1] >= I_cut.shape[1] - 2:
                p[1] = I_cut.shape[1] - 3
            if p[2] >= I_cut.shape[2] - 2:
                p[2] = I_cut.shape[2] - 3

            if p[0] < 0:
                p[0] = 0
            if p[1] < 0:
                p[1] = 0
            if p[2] < 0:
                p[2] = 0

            result = interpn(bbox_points, I_cut, p, method=M)
            if result[0] == plaque_label:
                count = count + 1
                # out[i] = 1
                # break
        if count >= 50:
            out[i] = 1

    return out

# mask_vec着色
def colored_mask_vec(mask_vec, slice_width):
    colored_mask = np.zeros([mask_vec.shape[0], slice_width, 3])
    colored_mask_vec = np.where(mask_vec == 1, (0, 0, 255), (0, 0, 0))

    for i in range(slice_width):
        colored_mask[:, i] = colored_mask_vec

    return colored_mask



if __name__ == '__main__':
    with open('parameters.json') as f:
        parameters = json.load(f)

    # 需要的路径
    CT_data_root = parameters['data_path']['CT_data_root']
    cl_save_path = parameters['data_path']['cl_save_path']
    plaque_info_path = parameters['data_path']['plaque_info_path']

    image_path = os.path.join(parameters['data_path']['CT_data_root'], parameters['data_path']['image_path'])
    mask_path = os.path.join(parameters['data_path']['CT_data_root'], parameters['data_path']['mask_path'])

    MPR_save_path = os.path.join(parameters['data_path']['CT_data_root'], parameters['data_path']['MPR_save_path'])
    MPR_vis_path = os.path.join(parameters['data_path']['CT_data_root'], parameters['data_path']['MPR_vis_path'])

    with open(os.path.join(CT_data_root, plaque_info_path, 'lines.txt'), 'r') as f:
    # with open('./debug_tmpfile/lines.txt', 'r') as f:
        lines_cross_the_plaque_fp_list = f.readlines()

    lines_cross_the_plaque_fp_list = [os.path.join(CT_data_root, cl_save_path, i.strip()) for i in lines_cross_the_plaque_fp_list]

    for line_fp in lines_cross_the_plaque_fp_list:
        line_name = re.findall('\d+/\d+.txt', line_fp)[0].split('.txt')[0]
        line_name = re.sub('/', '_', line_name)

        print('{} under the process'.format(line_name))
        # 病例号
        nii_c = re.findall('\d+', line_fp)[-2]

        ### 从txt中读取中心线
        cl_points = load_centerline(line_fp)

        # 中心线插值
        cl_points = cl_interpolate(cl_points,
                                   N_min=parameters['preprocessing']['cl_interpolate_min'],
                                   N_max=parameters['preprocessing']['cl_interpolate_max'])

        ### 读取image和mask
        # image路径
        image_fp = os.path.join(image_path, '{}.nii.gz'.format(nii_c))
        # mask路径
        mask_fp = os.path.join(mask_path, '{}.nii.gz'.format(nii_c))

        I = nib.load(image_fp).get_fdata().transpose(2, 1, 0)
        MASK = nib.load(mask_fp).get_fdata().transpose(2, 1, 0)

        ### 灰度截断
        I_pre = gray_preprocess(I, max=parameters['preprocessing']['gray_max'], min=parameters['preprocessing']['gray_min'])

        Z = np.array(range(I_pre.shape[0]))
        Y = np.array(range(I_pre.shape[1]))
        X = np.array(range(I_pre.shape[1]))
        bbox = [Z, Y, X]

        ### 计算向量
        T = cal_tangent_vector(cl_points)
        V_1 = cal_V_1(cl_points, T)
        V_2 = cal_V_2(V_1, T)

        ### 构建待插值平面
        slice_width = parameters['MPR']['slice_width']
        interpolate_space = Pi_slice(cl_points, slice_width, V_1, V_2)



        # 构建下文件夹
        if not os.path.exists(os.path.join(MPR_save_path, 'image')):
            os.makedirs(os.path.join(MPR_save_path, 'image'))
        if not os.path.exists(os.path.join(MPR_save_path, 'mask')):
            os.makedirs(os.path.join(MPR_save_path, 'mask'))
        if not os.path.exists(os.path.join(MPR_save_path, 'mask_vec')):
            os.makedirs(os.path.join(MPR_save_path, 'mask_vec'))

        ### 插值得到image和mask的MPR
        print("Interpolate image and mask")
        if os.path.exists(os.path.join(MPR_save_path, 'image', line_name+'.mat')) and os.path.exists(os.path.join(MPR_save_path, 'mask' ,line_name+'.mat')):
            # 如果已经生成过了，就直接读取CPR_I.mat和CPR_MASK.mat
            MPR_I = scio.loadmat(os.path.join(MPR_save_path, 'image', line_name+'.mat'))['MPR_I']
            MPR_MASK = scio.loadmat(os.path.join(MPR_save_path, 'mask', line_name+'.mat'))['MPR_MASK']
        else:
            # 得到MPR_I and MPR_MASK
            MPR_I = Interpolate(interpolate_space, bbox, I_pre)
            MPR_MASK = Interpolate(interpolate_space, bbox, MASK, M='nearest')

            # 存储CPR_I.mat和CPR_MASK.mat
            scio.savemat(os.path.join(MPR_save_path, 'image', line_name+'.mat'), {'MPR_I': MPR_I})
            scio.savemat(os.path.join(MPR_save_path, 'mask', line_name+'.mat'), {'MPR_MASK': MPR_MASK})

        ### 插值得到mask_vec
        print("Interpolate mask vec")
        if os.path.exists(os.path.join(MPR_save_path, 'mask_vec', line_name+'.mat')):
            mask_vec = scio.loadmat(os.path.join(MPR_save_path, 'mask_vec', line_name+'.mat'))['MPR_MASK_VEC']
        else:
            mask_vec = Interpolate_Mask(interpolate_space, bbox, MASK)
            # 存储mask_vec
            scio.savemat(os.path.join(MPR_save_path, 'mask_vec', line_name+'.mat'), {'MPR_MASK_VEC': mask_vec})



        ### 存图，只存血管剖面
        # 文件夹
        if not os.path.exists(os.path.join(MPR_vis_path, 'image')):
            os.makedirs(os.path.join(MPR_vis_path, 'image'))
        if not os.path.exists(os.path.join(MPR_vis_path, 'mask')):
            os.makedirs(os.path.join(MPR_vis_path, 'mask'))
        if not os.path.exists(os.path.join(MPR_vis_path, 'i_m_add')):
            os.makedirs(os.path.join(MPR_vis_path, 'i_m_add'))

        t_fp_1 = os.path.join(os.path.join(MPR_vis_path, 'image'), line_name + '.png')
        t_fp_2 = os.path.join(os.path.join(MPR_vis_path, 'mask'), line_name + '.png')
        t_fp_3 = os.path.join(os.path.join(MPR_vis_path, 'i_m_add'), line_name + '.png')


        pos = int(slice_width / 2)
        # 着色
        # MPR_slice = (MPR_I[:, pos, :] + MPR_I[:, :, pos]) / 2
        MPR_slice = MPR_I[:, pos, :]
        MPR_colored_slice = np.zeros([MPR_slice.shape[0], MPR_slice.shape[1], 3])
        for i in range(3):
            MPR_colored_slice[:, :, i] = MPR_slice

        colored_mask = colored_mask_vec(mask_vec, slice_width)

        size = (int(MPR_slice.shape[1] * 4), int(MPR_slice.shape[0] * 4))

        # 存图
        cv2.imwrite(t_fp_1, cv2.resize(MPR_slice, size))
        cv2.imwrite(t_fp_2, cv2.resize(colored_mask, size))
        img_add = cv2.addWeighted(MPR_colored_slice, 0.7, colored_mask, 0.3, gamma=0.0)
        cv2.imwrite(t_fp_3, cv2.resize(img_add, size))





