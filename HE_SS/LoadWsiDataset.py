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

ds = WSILabelPatchDataset(
    img_paths=wsi_paths[:3],
    img_labels=labels[:3],
    class_dict=class_dict,
    mode="wsi",
    patch_input_shape=patch_input_shape,
    stride_shape=stride_shape,
    #preproc_func=None,
    auto_get_mask=True,
    resolution=res,
    units=un,
    thres=0.5,
    #auto_mask_method="morphological",
    #kernel_size = 128,
    #min_region_size = 1000,
)

def overlay_patches(ds, img_id):
    fig, ax = plt.subplots()
    mask = ds.masks[img_id]
    mask = np.ma.masked_where(mask>0.0, 1-mask)
    img = ds.reader[img_id].thumbnail
    ax.imshow(img)
    ax.imshow(mask, alpha=0.8, cmap='Reds', vmin=0, vmax=1.3)
    ax.title.set_text("\"{}\"\nLabel: \"{}\" | Patch number:{}".format(ds.img_paths[img_id], class_dict[ds.img_labels[img_id]], ds.num_patches[img_id]))
    scale = ds.masks[img_id].shape[0]/ds.reader[img_id].slide_dimensions(resolution=res, units=un)[1]
    for x, y in zip(ds.inputs[img_id][:,0]*scale, ds.inputs[img_id][:,1]*scale):
         rect=mpl.patches.Rectangle(xy=(x, y), width=patch_input_shape[0]*scale, height=patch_input_shape[1]*scale, linewidth=0.5, edgecolor='g', facecolor='none') 
         ax.add_patch(rect)
    plt.show()

overlay_patches(ds, 0)
