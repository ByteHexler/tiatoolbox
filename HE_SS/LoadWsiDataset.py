import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

from tiatoolbox import rcParam
from tiatoolbox.tools.tissuemask import MorphologicalMasker
from tiatoolbox.models.dataset.classification import WSIPatchDataset
from tiatoolbox.models.dataset.wsiclassification import WSILabelPatchDataset

import os
import pandas as pd

mpl.rcParams['figure.dpi'] = 300 # for high resolution figure in notebook

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if device=='cuda:0': ON_GPU = True

wsi_dir = 'V:\Comparative-Computational\Pancreas\scans'
excel_dir = 'V:\Comparative-Computational\Pancreas\scans\Training_dataset_list_formatiert.xlsx'

class_dict={0: 'SS', 1: 'FFPE'}

def get_data(wsi_dir, excel_dir, class_dict):
    class_dict = {v: k for k, v in class_dict.items()}
    df=pd.read_excel(excel_dir, sheet_name="Slides").sort_values(by="ImageID")
    img_idx = list(df["ImageID"])
    paths=[]
    labels=[]
    for i, img_id in enumerate(img_idx):
        path = wsi_dir+os.sep+str(img_id)+".svs"
        label = class_dict[df.iloc[i]["Label"]]
        if os.path.isfile(path):
            paths.append(path)
            labels.append(label)
    return paths, labels

wsi_paths, labels = get_data(wsi_dir, excel_dir, class_dict)

res = 5
un = 'power'
patch_input_shape=[256, 256]
stride_shape=[240, 240]

train_ds = WSILabelPatchDataset(
    img_paths=wsi_paths[:35],
    img_labels=labels[:35],
    class_dict=class_dict,
    mode="wsi",
    save_dir="C:\\Users\\ge63kug\\source\\repos\\tiatoolbox\\HE_SS\\out_data",
    patch_input_shape=patch_input_shape,
    stride_shape=stride_shape,
    #preproc_func=None,
    auto_get_mask=True,
    resolution=res,
    units=un,
    thres=0.5,
    #ignore_background=False,
    #auto_mask_method="morphological",
    #kernel_size = 128,
    #min_region_size = 1000,
)

test_ds = WSILabelPatchDataset(
    img_paths=wsi_paths[35:],
    img_labels=labels[35:],
    class_dict=class_dict,
    mode="wsi",
    save_dir="C:\\Users\\ge63kug\\source\\repos\\tiatoolbox\\HE_SS\\out_data",
    patch_input_shape=patch_input_shape,
    stride_shape=stride_shape,
    #preproc_func=None,
    auto_get_mask=True,
    resolution=res,
    units=un,
    thres=0.5,
    #ignore_background=False,
    #auto_mask_method="morphological",
    #kernel_size = 128,
    #min_region_size = 1000,
)
