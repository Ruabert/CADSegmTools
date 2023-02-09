"""
Created on 01/09/2023

@author: ZhangRuarua
"""

import os
import scipy.io as scio
import datetime as dt
from utils.tools import *
from shutil import copyfile


####################################
######### ALL TWIGS TO MAT #########
####################################

def txt_all_label_to_mat(data_root_fp, txt_fp_list, cl_mat_fp, img_mat_fp, f_name, step=2):
    lines = []
    for txt_fp in txt_fp_list:

        # 读取txt文件写入mat
        with open(txt_fp, 'r') as f:
            rows = f.readlines()
            rows_step = rows[::step]
            line = [0 for r in range(len(rows_step))]

            for i, r in enumerate(rows_step):
                z, y, x = [int(x) for x in re.findall('\d+', r)]
                line[i] = np.array([y+1, x+1, z+1], dtype=int)

            lines.append(line)
            f.close()

    lines.append(np.array([], dtype=float))
    lines = np.array([lines])

    # 读取我们使用的mat模板
    cl_mat = scio.loadmat('1_ctline_split_2_ex.mat')
    now_time = dt.datetime.now().strftime('%F %T')
    header = 'MATLAB 5.0 MAT-file Platform: nt, Created on: {}'.format(now_time)
    header = bytes(header, encoding='UTF-8')

    cl_mat['__header__'] = header
    cl_mat['centerline'] = lines
    scio.savemat(cl_mat_fp, cl_mat)

    # 读取img存入mat
    img_mat = scio.loadmat('1.mat')
    img_name = re.findall('\d+', img_mat_fp)[-1]

    assert str(f_name) == img_name

    if os.path.exists(img_mat_fp) == False:
        img = nib.load(os.path.join(data_root_fp, 'nii/image/{}.nii.gz'.format(img_name))).get_fdata()
        img_mat['img'] = img
        scio.savemat(img_mat_fp, img_mat)


# def csv_all_label_to_mat(csv_fp, cl_mat_fp, img_mat_fp, f_name, step=2):
#     # 读取之前处理好的CSV
#     df = pd.read_csv(csv_fp).T.iloc[1:, :]
#
#     lines = []
#     for i, s in df.iterrows():
#         # 从csv里面取点
#         label = s[0]
#         points = s[1]
#         points = ''.join(points)[1:-1]
#         points = re.findall('\d+, \d+, \d+', points)
#
#         # 间隔取点
#         line = points[0::step]
#         if len(line) == 0:
#             continue
#         print("line: {}, points_num: {}".format(label, len(line)))
#
#
#         for i, v in enumerate(line):
#             z, y, x = v.split(', ')
#             z, y, x = int(z), int(y), int(x)
#             line[i] = np.array([y+1, x+1, z+1], dtype=int)
#
#         # lines
#         lines.append(line)
#
#     lines.append(np.array([], dtype=float))
#     lines = np.array([lines])
#
#     # 读取我们使用的mat模板
#     cl_mat = scio.loadmat('1_ctline_split_2_ex.mat')
#     now_time = dt.datetime.now().strftime('%F %T')
#     header = 'MATLAB 5.0 MAT-file Platform: nt, Created on: {}'.format(now_time)
#     header = bytes(header, encoding='UTF-8')
#
#     cl_mat['__header__'] = header
#     cl_mat['centerline'] = lines
#     scio.savemat(cl_mat_fp, cl_mat)
#
#     # 读取img存入mat
#     img_mat = scio.loadmat('1.mat')
#     img_name = re.findall('\d+', img_mat_fp)[-1]
#
#     assert str(f_name) == img_name
#
#     if os.path.exists(img_mat_fp) == False:
#         img = nib.load(os.path.join(data_root_fp, './DATA/nii/image/{}.nii.gz'.format(img_name)).get_fdata()
#         img_mat['img'] = img
#         scio.savemat(img_mat_fp, img_mat)

if __name__ == '__main__':

    # 基本参数
    f_name = 2
    step = 5

    # 存储的文件夹
    data_root_fp = '../DATA/22_oct_Coronary_Artery_CT/'
    save_dir = '2D_DATA_BULID_2'

    # 一些路径
    # 中心线 txt路径
    txt_root = os.path.join(data_root_fp, 'nii/vis/{}/each_label/'.format(f_name))
    all_fp_list = os.listdir(txt_root)
    txt_fp_list = []

    for x in all_fp_list:
        if x.split('.')[-1] == 'txt':
            txt_fp_list.append(x)

    txt_fp_list.sort(key=lambda x: int(x.split('.txt')[0].split('_')[-1]))
    txt_fp_list = [os.path.join(txt_root, txt) for txt in txt_fp_list]


    # 输出的中心线 .mat文件的路径
    all_cl_mat_fp = os.path.join(data_root_fp, '2D_DATA/{}/IMAGE/{}/nii_{}_all_line.mat'.format(save_dir, f_name, f_name))
    # 输出的3D图像的 .mat路径
    img_mat_fp = os.path.join(data_root_fp, '2D_DATA/{}/IMAGE/{}/nii_{}.mat'.format(save_dir, f_name, f_name))
    # 输出的3DMask的 .mat路径
    mask_mat_fp = os.path.join(data_root_fp, '2D_DATA/{}/MASK/{}/nii_{}.mat'.format(save_dir, f_name, f_name))


    # 存储下 f_name、 cl_label 和 save_dir
    f_filename = os.path.join(data_root_fp, '2D_DATA/TMP_FILE/n.txt')
    with open(f_filename, 'w', encoding='utf-8') as f:
        f.write(str(f_name))

    cl_filename = os.path.join(data_root_fp, '2D_DATA/TMP_FILE/line.txt')
    with open(cl_filename, 'w', encoding='utf-8') as f:
        f.write('0')

    save_fp = os.path.join(data_root_fp, '2D_DATA/TMP_FILE/save_dir.txt')
    with open(save_fp, 'w', encoding='utf-8') as f:
        f.write(str(save_dir))

    print("save f_name: {} to {}".format(f_name, f_filename))

    # 若文件夹未能建立，建立下
    dir_path = re.split('/nii_\d+_all_line.mat', all_cl_mat_fp)[0]
    out_path = os.path.join(dir_path, 'out')
    TMP_path = os.path.join(dir_path, 'TMP')

    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)
    if os.path.exists(dir_path) and os.path.exists(out_path) == False:
        os.makedirs(out_path)
    if os.path.exists(dir_path) and os.path.exists(TMP_path) == False:
        os.makedirs(TMP_path)

    dir_path = re.split('/nii_\d+.mat', mask_mat_fp)[0]
    out_path = os.path.join(dir_path, 'out')

    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)
    if os.path.exists(dir_path) and os.path.exists(out_path) == False:
        os.makedirs(out_path)

    # 读取所有txt转为mat
    txt_all_label_to_mat(data_root_fp, txt_fp_list, cl_mat_fp=all_cl_mat_fp,
                     img_mat_fp=img_mat_fp, f_name=f_name,
                     step=step)


    print("save cl.mat to {}".format(all_cl_mat_fp))
    print("save img.mat to {}".format(img_mat_fp))


    # 读取 .mat文件的模板
    template_mat = scio.loadmat('1.mat')
    # 读取 mask的 .nii文件
    mask = nib.load(os.path.join(data_root_fp, 'nii/mask/{}.nii.gz'.format(f_name))).get_fdata()

    # 处理
    mask_1 = np.where(mask == 1, 1, 0)
    mask_254 = np.where(mask == 254, 254, 0)
    out_mask = mask_1 + mask_254

    # 存储
    template_mat['img'] = out_mask
    scio.savemat(mask_mat_fp, template_mat)

    print("save mask.mat to {}".format(mask_mat_fp))

    src = os.path.join(data_root_fp, "2D_DATA/{}/IMAGE/{}/nii_{}_all_line.mat".format(save_dir, f_name, f_name))
    tgt = os.path.join(data_root_fp, "2D_DATA/{}/MASK/{}/nii_{}_all_line.mat".format(save_dir, f_name, f_name))

    copyfile(src, tgt)
    print("cp cl.mat to {}".format(tgt))
