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
from utils.tools import write_to_nii
import datasets.dataset as Data

if __name__ == '__main__':
    ct_data = Data.CT_image(root_fp='../DATA/ZeRui_Coronary_Artery_CT')
    write_to_nii(data=ct_data, root_fp='../DATA/ZeRui_Coronary_Artery_CT')