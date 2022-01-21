import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from tiatoolbox import rcParam
from tiatoolbox.tools.tissuemask import MorphologicalMasker
from tiatoolbox.models.dataset.classification import WSIPatchDataset
from tiatoolbox.models.dataset.wsiclassification import WSILabelPatchDataset

mpl.rcParams['figure.dpi'] = 300 # for high resolution figure in notebook

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if device=='cuda:0': ON_GPU = True

wsi_path = './HE_SS/test_slides/Slide-002.svs'

res = 5
un = 'power'
patch_input_shape=[256, 256]
stride_shape=[240, 240]

ds = WSILabelPatchDataset(
    img_paths=[wsi_path],
    img_labels=[0],
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
    ax.imshow(ds.masks[img_id])
    scale = ds.masks[img_id].shape[0]/ds.reader[img_id].slide_dimensions(resolution=res, units=un)[1]
    for x, y in zip(ds.inputs[img_id][:,0]*scale, ds.inputs[img_id][:,1]*scale):
         rect=mpl.patches.Rectangle(xy=(x, y), width=patch_input_shape[0]*scale, height=patch_input_shape[1]*scale, linewidth=0.5, edgecolor='r', facecolor='none') 
         ax.add_patch(rect)
    plt.show()

overlay_patches(ds, 0)
