import os
import numpy as np
import nibabel as nib
import time
from utils.tools import load_centerline, find_plaque
from utils.dbscan import getClusters
import skimage

# 对斑块进行聚类
def plaque_clustering(plaque_mask):
    # 对只含有plaque的mask进行聚类，区分出不同位置的plaque
    plaque_points = np.where(plaque_mask == 254)
    plaque_points = [(plaque_points[0][i], plaque_points[1][i], plaque_points[2][i]) for i in range(len(plaque_points[0]))]
    plaque_points = np.array(plaque_points)

    start = time.time()

    # 耗时
    c_dict = getClusters(plaque_points, eps=6, minpts=4)

    # print('time wasted: %.2f  min' % ((time.time() - start) / 60))
    points_c = []
    for c, points in c_dict.items():
        if len(points) == 0:
            break

        points = [np.array([plaque_points[i, 0], plaque_points[i, 1], plaque_points[i, 2]]) for i in points]
        points_c.append(points)

    return points_c, len(points_c)

# 对斑块进行保存
def extract_plaque_info(nii_fp, nii_tfp):

    # 读取mask
    mask = nib.load(nii_fp).get_fdata().transpose(2, 1, 0)
    plaque_mask = find_plaque(mask)
    plaque_mask = skimage.morphology.skeletonize_3d(plaque_mask)

    # print('clustering plaques')
    points_c, cluster = plaque_clustering(plaque_mask)

    # 存这些 plaque
    # print('save plaques')
    for i in range(cluster):
        points_c_i = points_c[i]
        points_c_i = np.array(points_c_i)

        zmin, zmax = np.min(points_c_i[:, 0]), np.max(points_c_i[:, 0])
        ymin, ymax = np.min(points_c_i[:, 1]), np.max(points_c_i[:, 1])
        xmin, xmax = np.min(points_c_i[:, 2]), np.max(points_c_i[:, 2])
        bbox = [(zmin-4, zmax+4), (ymin-4, ymax+4), (xmin-4, xmax+4)]

        # with open(nii_tfp + 'plaque_{}.txt'.format(i), 'w') as f:
        #     lines = []
        #     for coordinate in points_c_i:
        #         coordinate_str = str(coordinate[0]) + ',' + str(coordinate[1]) + ',' + str(coordinate[2]) + '\n'
        #         lines.append(coordinate_str)
        #     f.writelines(lines)
        # f.close()
        tfp = os.path.join(nii_tfp, 'plaque_{}_bbox.txt'.format(i))
        with open(tfp, 'w') as f:
            lines = []
            for i in range(3):
                l = str(bbox[i][0]) + ',' + str(bbox[i][1]) + '\n'
                lines.append(l)
            f.writelines(lines)
        f.close()
    return points_c, cluster

def is_line_across_plaque(cl_points, plaque, thresh=10):
    with open(plaque, 'r') as f:
        contents = f.readlines()
    f.close()
    bbox = [(int(c.strip().split(',')[0]), int(c.strip().split(',')[1])) for c in contents]

    Zmin, Zmax = bbox[0]
    Ymin, Ymax = bbox[1]
    Xmin, Xmax = bbox[2]

    cross_plaque = 0
    for coordinate in cl_points:
        z = coordinate[0]
        y = coordinate[1]
        x = coordinate[2]

        bool1 = z >= Zmin and z <= Zmax
        bool2 = y >= Ymin and y <= Ymax
        bool3 = x >= Xmin and x <= Xmax

        if bool1 and bool2 and bool3:
            cross_plaque = cross_plaque + 1

    return cross_plaque > thresh

def find_lines_for_MPR(mask_fp, nii_tfp, cl_fp, lines_fp):

    # 读取mask文件
    mask_fp = mask_fp
    nii_tfp = nii_tfp
    nii_n = nii_tfp.split('\\')[-1]

    # 通过聚类，提取出不同斑块的bbox信息
    # print('extract plaques from nii {}'.format(nii_n))
    _, cluster = extract_plaque_info(mask_fp, nii_tfp)
    # print('find {} plaques from nii {}'.format(cluster, nii_n))


    cl_fp = cl_fp
    plaque_fp = nii_tfp
    lines_fp = lines_fp

    with open(lines_fp, 'r') as f:
        existed_lines = f.readlines()
    f.close()

    existed_lines = [el.strip() for el in existed_lines]

    cl_txt_list = [i for i in os.listdir(cl_fp) if i.endswith('.txt')]
    cl_txt_list = sorted(cl_txt_list, key=lambda x: int(x.split('.')[0]))
    # cl_txt_fp = [cl_fp + '/' + i for i in cl_txt]

    plaque_txt_list = [i for i in os.listdir(plaque_fp) if i.split('.')[0].split('_')[-1] == 'bbox']
    # plaque_txt = [plaque_fp + '/' + i for i in plaque_txt]

    # 找到经过斑块的中心线
    wait_to_be_wrote = []
    for cl_txt in cl_txt_list:
        cl_points = load_centerline(fp=cl_fp + '/' + cl_txt)
        for plaque_txt in plaque_txt_list:
            if is_line_across_plaque(cl_points, plaque=plaque_fp + '/' + plaque_txt):
                cl_txt = nii_n + '/' + cl_txt
                if len(existed_lines) == 0 or cl_txt not in existed_lines:
                    wait_to_be_wrote.append(cl_txt)

                break

    wait_to_be_wrote = np.unique(wait_to_be_wrote)
    print('find centerlines {}, across the plaque'.format([i.strip().split('.txt')[0] for i in wait_to_be_wrote]))

    if len(wait_to_be_wrote):
        with open(lines_fp, 'a') as f:
            for l in list(wait_to_be_wrote):
                if l.strip() not in existed_lines:
                    f.writelines(l + '\n')

    f.close()






