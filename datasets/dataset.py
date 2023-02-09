"""
Created on 01/09/2023

@author: ZhangRuarua
"""

import os
import SimpleITK as Sitk
from torch.utils.data import Dataset
import numpy as np
import re


class CT_image(Dataset):
    def __init__(self, root_fp, img_set='train'):
        super(CT_image, self).__init__()

        # 图像文件的根路径
        self.root_fp = root_fp
        # .raw的路径
        self.raw_fp_list = []
        # .mhd的路径
        self.mhd_fp_list = []

        # mask文件的根路径
        # .raw的路径
        self.mask_raw_fp_list = []
        # .mhd的路径
        self.mask_mhd_fp_list = []

        self.raw_fp_list, self.mhd_fp_list = self.get_fp_index(set='image')
        self.mask_raw_fp_list, self.mask_mhd_fp_list = self.get_fp_index(set='mask')


    def get_fp_index(self, set='image'):
        raw_fp_list = []
        mhd_fp_list = []

        _root_fp = os.path.join(self.root_fp, set)

        for fp in os.listdir(_root_fp):
            if os.path.splitext(fp)[1] == '.raw' or os.path.splitext(fp)[1] == '.zraw':
                fp = os.path.join(_root_fp, fp)
                raw_fp_list.append(fp)

            if os.path.splitext(fp)[1] == '.mhd':
                fp = os.path.join(_root_fp, fp)
                mhd_fp_list.append(fp)

        return raw_fp_list, mhd_fp_list

    def __getitem__(self, idx):
        # 拿到 item fp
        mhd_fp = self.mhd_fp_list[idx]
        mask_mhd_fp = self.mask_mhd_fp_list[idx]

        # 拿到图片和mask
        raw = Sitk.ReadImage(mhd_fp)
        mask_raw = Sitk.ReadImage(mask_mhd_fp)
        image_array = Sitk.GetArrayFromImage(raw)
        mask_array = Sitk.GetArrayFromImage(mask_raw)

        # 读入需要的属性
        Size = image_array.shape
        Direction = raw.GetDirection()
        Origin = raw.GetOrigin()
        Spacing = raw.GetSpacing()


        # 返回样本实例
        sample = {'img': image_array,
                  'mask': mask_array,
                  # 'cl': np.zeros(mask_array.shape),
                  'size': Size,
                  'Direction': Direction,
                  'Origin': Origin,
                  'Spacing': Spacing,
                  'code': re.findall('\d+',mhd_fp)[-1]}

        return sample

    def __len__(self):
        return len(self.mhd_fp_list)


