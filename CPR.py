"""
Created on 01/09/2023

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

# # 获得中心线的包围盒，带transpose
# def get_cl_bbox(I, points):
#     z = [p[0] for p in points]
#     y = [p[1] for p in points]
#     x = [p[2] for p in points]
#
#     temp_dict = {0:-1, 1:-1, 2:-1}
#     zmin, zmax = np.round(np.min(z) - 5), np.round(np.max(z) + 5)
#     temp_dict[0] = zmax-zmin
#     ymin, ymax = np.round(np.min(y) - 5), np.round(np.max(y) + 5)
#     temp_dict[1] = ymax-ymin
#     xmin, xmax = np.round(np.min(x) - 5), np.round(np.max(x) + 5)
#     temp_dict[2] = xmax-xmin
#
#     temp_dict = sorted(temp_dict.items(), key=lambda x:x[-1])
#     transpose_index = [x[0] for x in temp_dict]
#
#     sorted_dict = dict()
#     sorted_dict.update(temp_dict)
#
#     print('bbox z range {} to {}'.format(zmin, zmax))
#     print('bbox y range {} to {}'.format(ymin, ymax))
#     print('bbox x range {} to {}'.format(xmin, xmax))
#
#     I_cut = I[int(zmin):int(zmax), int(ymin):int(ymax), int(xmin):int(xmax)]
#     I_cut = I_cut.transpose(transpose_index)
#
#     print("centerline bbox shape is {}".format(I_cut.shape))
#
#     z = np.array(range(int(zmin), int(zmax)))
#     y = np.array(range(int(ymin), int(ymax)))
#     x = np.array(range(int(xmin), int(xmax)))
#
#     # 输入 interpn的points的维度必须为递增的
#     temp_list = [z, y, x]
#     bbox = [0, 0, 0]
#     for i in range(3):
#         bbox[i] = temp_list[transpose_index[i]]
#
#     return bbox, I_cut, transpose_index

# 获得中心线的包围盒
def get_cl_bbox(I, points):
    z = [p[0] for p in points]
    y = [p[1] for p in points]
    x = [p[2] for p in points]

    zmin, zmax = np.round(np.min(z) - 5), np.round(np.max(z) + 5)
    ymin, ymax = np.round(np.min(y) - 5), np.round(np.max(y) + 5)
    xmin, xmax = np.round(np.min(x) - 5), np.round(np.max(x) + 5)

    I_cut = I[int(zmin):int(zmax), int(ymin):int(ymax), int(xmin):int(xmax)]

    print("centerline bbox shape is {}".format(I_cut.shape))

    z = np.array(range(int(zmin), int(zmax)))
    y = np.array(range(int(ymin), int(ymax)))
    x = np.array(range(int(xmin), int(xmax)))

    bbox = [z, y, x]
    print('bbox z range {} to {}'.format(zmin, zmax))
    print('bbox y range {} to {}'.format(ymin, ymax))
    print('bbox x range {} to {}'.format(xmin, xmax))

    return bbox, I_cut

######## CPR ########
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

            result = interpn(bbox_points, I_cut, p, method=M)
            out[i, j] = result

    return out.reshape(out.shape[0], slice_width, -1)

# 对待mask插值空间进行插值
def Interpolate_Mask(interpolate_space, bbox_points, I_cut, plaque_label = 254 ,M = 'nearest'):
    slice_width = interpolate_space.shape[1]

    out = np.zeros([interpolate_space.shape[0], 1])

    for i in tqdm(range(interpolate_space.shape[0])):
        slice = interpolate_space[i]
        slice = slice.reshape(-1, 3)
        for j in range(slice.shape[0]):
            p = slice[j]

            result = interpn(bbox_points, I_cut, p, method=M)
            if np.unique(result) == plaque_label:
                out[i] = 1
                break

    return out

# mask_vec着色
def colored_mask_vec(mask_vec, slice_width):
    colored_mask = np.zeros([mask_vec.shape[0], slice_width, 3])
    colored_mask_vec = np.where(mask_vec == 1, (0, 0, 255), (0, 0, 0))

    for i in range(slice_width):
        colored_mask[:, i] = colored_mask_vec

    return colored_mask


def make_cpr(nii_n, cl_n, slice_width=32):
    nii_n = nii_n
    cl_n = cl_n

    # 读取中心线
    fp = './DATA/nii/vis/{}/each_label/{}_{}.txt'.format(nii_n, nii_n, cl_n)
    cl_points = load_centerline(fp)

    N_min = 80
    N_max = 600
    slice_width = slice_width

    # 对中心线进行插值
    cl_points = cl_interpolate(cl_points, N_min, N_max)

    ######## 处理得到nii图像文件和mask的CPR ########

    # 读取图像和mask
    I_root = './DATA/nii/image/{}.nii.gz'.format(nii_n)
    I = nib.load(I_root).get_fdata().transpose(2, 1, 0)
    MASK_root = './DATA/nii/mask/{}.nii.gz'.format(nii_n)
    MASK = nib.load(MASK_root).get_fdata().transpose(2, 1, 0)

    # 图像预处理：灰度截断
    I_pre = gray_preprocess(I, max=3000, min=-200)

    # I_cut = I_pre
    I_cut = I_pre
    MASK_cut = MASK

    Z = np.array(range(I_pre.shape[0]))
    Y = np.array(range(I_pre.shape[1]))
    X = np.array(range(I_pre.shape[1]))
    bbox = [Z, Y, X]

    # 计算向量
    T = cal_tangent_vector(cl_points)
    V_1 = cal_V_1(cl_points, T)
    V_2 = cal_V_2(V_1, T)

    # 构建待插值平面
    interpolate_space = Pi_slice(cl_points, slice_width, V_1, V_2)

    # 插值得到图像和mask的CPR
    print("Interpolate image and mask")

    if os.path.exists('./DATA/CPR/mat/{}/{}_{}.mat'.format(nii_n, nii_n, cl_n)) and os.path.exists('./DATA/CPR/mat/{}/{}_{}_MASK.mat'.format(nii_n, nii_n, cl_n)):
        # 如果已经生成过了，就直接读取CPR_I.mat和CPR_MASK.mat
        CPR_I = scio.loadmat('./DATA/CPR/mat/{}/{}_{}.mat'.format(nii_n, nii_n, cl_n))['CPR_I']
        CPR_MASK = scio.loadmat('./DATA/CPR/mat/{}/{}_{}_MASK.mat'.format(nii_n, nii_n, cl_n))['CPR_MASK']
    else:
        try:
            os.makedirs('./DATA/CPR/mat/{}'.format(nii_n))
        except:pass

        CPR_I = Interpolate(interpolate_space, bbox, I_cut)
        CPR_MASK = Interpolate(interpolate_space, bbox, MASK_cut, M='nearest')

        # 存储CPR_I.mat和CPR_MASK.mat
        scio.savemat('./DATA/CPR/mat/{}/{}_{}.mat'.format(nii_n, nii_n, cl_n), {'CPR_I': CPR_I})
        scio.savemat('./DATA/CPR/mat/{}/{}_{}_MASK.mat'.format(nii_n, nii_n, cl_n), {'CPR_MASK': CPR_MASK})

    ######## 处理得到MASK_VEC ########
    mask_cut = MASK
    mask_bbox = bbox

    print("Interpolate mask vec")
    if os.path.exists('./DATA/CPR/mat/{}/{}_{}_MASK_VEC.mat'.format(nii_n, nii_n, cl_n)):
        mask_vec = scio.loadmat('./DATA/CPR/mat/{}/{}_{}_MASK_VEC.mat'.format(nii_n, nii_n, cl_n))
        mask_vec = mask_vec['MASK_VEC']
    else:
        mask_vec = Interpolate_Mask(interpolate_space, bbox, I_cut)
        # 存储mask_vec
        scio.savemat('./DATA/CPR/mat/{}/{}_{}_MASK_VEC.mat'.format(nii_n, nii_n, cl_n), {'MASK_VEC': mask_vec})

    colored_mask = colored_mask_vec(mask_vec, slice_width)

    ######## 存储 ########
    # 存血管轴面
    t_fp = './DATA/CPR'.format(nii_n, cl_n)

    t_fp_1 = os.path.join(t_fp, 'sCPR/{}_{}.jpg'.format(nii_n, cl_n))
    t_fp_2 = os.path.join(t_fp, 'sCPR_mask_vec/{}_{}_mask_vec.jpg'.format(nii_n, cl_n))
    t_fp_3 = os.path.join(t_fp, 'sCPR_add/{}_{}_add.jpg'.format(nii_n, cl_n))

    pos = int(slice_width/2)
    CPR_slice = (CPR_I[:, pos, :] + CPR_I[:, :, pos]) / 2
    CPR_T_slice = np.zeros([CPR_slice.shape[0], CPR_slice.shape[1], 3])
    for i in range(3):
        CPR_T_slice[:, :, i] = CPR_slice

    size = (int(CPR_slice.shape[1]*2), int(CPR_slice.shape[0]*2))

    cv2.imwrite(t_fp_1, cv2.resize(CPR_slice, size))
    cv2.imwrite(t_fp_2, cv2.resize(colored_mask, size))

    img_add = cv2.addWeighted(CPR_T_slice, 0.7, colored_mask, 0.3, gamma=0.0)
    cv2.imwrite(t_fp_3, cv2.resize(img_add, size))

    # 存血管截面
    t_fp_4 = os.path.join(t_fp, 'sCPR_section/{}_{}'.format(nii_n, cl_n))
    t_fp_5 = os.path.join(t_fp, 'sCPR_section_mask/{}_{}_mask'.format(nii_n, cl_n))
    t_fp_6 = os.path.join(t_fp, 'sCPR_section_add/{}_{}_add'.format(nii_n, cl_n))

    try:os.makedirs(t_fp_4)
    except:pass

    try:os.makedirs(t_fp_5)
    except:pass

    try:os.makedirs(t_fp_6)
    except:pass

    for i in np.arange(0, CPR_I.shape[0])[::5]:
        CPR_I_i = CPR_I[i, :, :]
        CPR_I_T = np.zeros([CPR_I_i.shape[0], CPR_I_i.shape[1], 3])
        for j in range(3):
            CPR_I_T[:, :, j] = CPR_I_i

        CPR_MASK_i = CPR_MASK[i, :, :]
        CPR_MASK_T = np.zeros([CPR_MASK_i.shape[0], CPR_MASK_i.shape[1], 3])
        for j in range(3):
            CPR_MASK_T[:, :, j] = CPR_MASK_i


        MASK_1 = np.where(CPR_MASK_T == [1, 1, 1], [45, 10, 254], [0, 0, 0])
        MASK_254 = np.where(CPR_MASK_T == [254, 254, 254], [10, 230, 0], [0, 0, 0])
        CPR_MASK_T = MASK_1 + MASK_254
        CPR_MASK_T = CPR_MASK_T.astype(np.float64)

        CPR_ADD_i = cv2.addWeighted(CPR_I_T, 0.6, CPR_MASK_T, 0.4, gamma=0.0)

        size = (CPR_I_T.shape[0] * 2, CPR_I_T.shape[1] * 2)
        cv2.imwrite(os.path.join(t_fp_4, '{}.jpg'.format(i)), cv2.resize(CPR_I_T, size))
        cv2.imwrite(os.path.join(t_fp_5, '{}.jpg'.format(i)), cv2.resize(CPR_MASK_T, size))
        cv2.imwrite(os.path.join(t_fp_6, '{}.jpg'.format(i)), cv2.resize(CPR_ADD_i, size))


if __name__ == '__main__':
    data_root_fp = '../DATA/22_oct_Coronary_Artery_CT/nii/'
    lines_fp = os.path.join(data_root_fp, 'plaque_info/lines.txt')

    with open(lines_fp, 'r') as f:
        content = f.readlines()
        for nii_cl in content:
            nii_cl = nii_cl.strip()
            nii_cl = nii_cl.split('.txt')[0]
            nii_n = int(nii_cl.split('_')[0])
            cl_n = int(nii_cl.split('_')[-1])
            print("nii: {}, cl: {}".format(nii_n, cl_n))
            try:
                make_cpr(nii_n, cl_n, slice_width=32)
            except:pass

    # make_cpr(nii_n=2, cl_n=11, slice_width=32)









