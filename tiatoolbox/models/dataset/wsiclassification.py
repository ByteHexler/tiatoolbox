# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****


import os
import pathlib
import warnings

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

import json

import cv2
import numpy as np
import PIL
import torchvision.transforms as transforms

from tiatoolbox.models.dataset import abc
from tiatoolbox.tools.patchextractionThres import PatchExtractor
from tiatoolbox.utils.misc import imread
from tiatoolbox.wsicore.wsimeta import WSIMeta
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, get_wsireader

from tiatoolbox.utils.misc import save_as_json


class _TorchPreprocCaller:
    """Wrapper for applying PyTorch transforms.

    Args:
        preprocs (list): List of torchvision transforms for preprocessing the image.
          The transforms will be applied in the order that they are given in the
          list. For more informaion, visit the following link:
          https://pytorch.org/vision/stable/transforms.html.

    """

    def __init__(self, preprocs):
        self.func = transforms.Compose(preprocs)

    def __call__(self, img):
        img = PIL.Image.fromarray(img)
        img = self.func(img)
        img = img.permute(1, 2, 0)
        return img


def predefined_preproc_func(dataset_name):
    """Get the preprocessing information used for the pretrained model.

    Args:
        dataset_name (str): Dataset name used to determine what preprocessing was used.
    Returns:
        preproc_func (_TorchPreprocCaller): Preprocessing function for transforming
          the input data.

    """
    preproc_dict = {
        "kather100k": [
            transforms.ToTensor(),
        ],
        "pcam": [
            transforms.ToTensor(),
        ],
    }

    if dataset_name not in preproc_dict:
        raise ValueError(
            f"Predefined preprocessing for dataset `{dataset_name}` does not exist."
        )

    preprocs = preproc_dict[dataset_name]
    preproc_func = _TorchPreprocCaller(preprocs)
    return preproc_func


class PatchDataset(abc.PatchDatasetABC):
    """Defines a simple patch dataset, which inherits from the
      torch.utils.data.Dataset class.

    Attributes:
        inputs: Either a list of patches, where each patch is a ndarray or a list of
          valid path with its extension be
          (".jpg", ".jpeg", ".tif", ".tiff", ".png") pointing to an image.
        labels: List of label for sample at the same index in `inputs`.
          Default is `None`.
        preproc_func: Preprocessing function used to transform the input data.

    Examples:
        >>> # an user defined preproc func and expected behavior
        >>> preproc_func = lambda img: img/2  # reduce intensity by half
        >>> transformed_img = preproc_func(img)
        >>> # create a dataset to get patches preprocessed by the above function
        >>> ds = PatchDataset(
        ...     inputs=['/A/B/C/img1.png', '/A/B/C/img2.png'],
        ...     preproc_func=preproc_func
        ... )

    """

    def __init__(self, inputs, labels=None):
        super().__init__()

        self.data_is_npy_alike = False

        self.inputs = inputs
        self.labels = labels

        # perform check on the input
        self._check_input_integrity(mode="patch")

    def __getitem__(self, idx):
        patch = self.inputs[idx]

        # Mode 0 is list of paths
        if not self.data_is_npy_alike:
            patch = self.load_img(patch)

        # Apply preprocessing to selected patch
        patch = self._preproc(patch)

        data = {
            "image": patch,
        }
        if self.labels is not None:
            data["label"] = self.labels[idx]
            return data

        return data


class WSIPatchDataset(abc.PatchDatasetABC):
    """Defines a WSI-level patch dataset.

    Attributes:
        reader (:class:`.WSIReader`): an WSI Reader or Virtual Reader
          for reading pyramidal image or large tile in pyramidal way.
        inputs: List of coordinates to read from the `reader`,
          each coordinate is of the form [start_x, start_y, end_x, end_y].
        patch_input_shape: a tuple(int, int) or ndarray of shape (2,).
          Expected size to read from `reader` at requested `resolution`
          and `units`. Expected to be (height, width).
        resolution: check (:class:`.WSIReader`) for details.
        units: check (:class:`.WSIReader`) for details.
        preproc_func: Preprocessing function used to transform the input data.
          If supplied, then torch.Compose will be used on the input preprocs.
          preprocs is a list of torchvision transforms for preprocessing the
          image. The transforms will be applied in the order that they are
          given in the list. For more information, visit the following link:
          https://pytorch.org/vision/stable/transforms.html.

    """

    def __init__(
        self,
        img_path,
        img_label=None,
        mode="wsi",
        mask_path=None,
        patch_input_shape=None,
        stride_shape=None,
        resolution=None,
        units=None,
        thres=0.5,
        auto_get_mask=True,
        auto_mask_method="otsu",
        **masker_kwargs,
    ):
        """Create a WSI-level patch dataset.
        Args:
            mode (str): can be either `wsi` or `tile` to denote the image to read is
              either a whole-slide image or a large image tile.
            img_path (:obj:`str` or :obj:`pathlib.Path`): valid to pyramidal
              whole-slide image or large tile to read.
            mask_path (:obj:`str` or :obj:`pathlib.Path`): valid mask image.
            patch_input_shape: a tuple (int, int) or ndarray of shape (2,).
              Expected shape to read from `reader` at requested `resolution` and
              `units`. Expected to be positive and of (height, width). Note, this
              is not at `resolution` coordinate space.
            stride_shape: a tuple (int, int) or ndarray of shape (2,).
              Expected stride shape to read at requested `resolution` and `units`.
              Expected to be positive and of (height, width). Note, this is not at
              level 0.
            resolution: check (:class:`.WSIReader`) for details. When `mode='tile'`,
              value is fixed to be `resolution=1.0` and `units='baseline'`
              units: check (:class:`.WSIReader`) for details.
            preproc_func: Preprocessing function used to transform the input data.

        Examples:
            >>> # an user defined preproc func and expected behavior
            >>> preproc_func = lambda img: img/2  # reduce intensity by half
            >>> transformed_img = preproc_func(img)
            >>> # create a dataset to get patches from WSI with above
            >>> # preprocessing function
            >>> ds = WSIPatchDataset(
            ...     img_path='/A/B/C/wsi.svs',
            ...     mode="wsi",
            ...     patch_input_shape=[512, 512],
            ...     stride_shape=[256, 256],
            ...     auto_get_mask=False,
            ...     preproc_func=preproc_func
            ... )

        """
        super().__init__()

        # Is there a generic func for path test in toolbox?
        if not os.path.isfile(img_path):
            raise ValueError("`img_path` must be a valid file path.")
        if mode not in ["wsi", "tile"]:
            raise ValueError(f"`{mode}` is not supported.")
        patch_input_shape = np.array(patch_input_shape)
        stride_shape = np.array(stride_shape)

        self.label = img_label

        if (
            not np.issubdtype(patch_input_shape.dtype, np.integer)
            or np.size(patch_input_shape) > 2
            or np.any(patch_input_shape < 0)
        ):
            raise ValueError(f"Invalid `patch_input_shape` value {patch_input_shape}.")
        if (
            not np.issubdtype(stride_shape.dtype, np.integer)
            or np.size(stride_shape) > 2
            or np.any(stride_shape < 0)
        ):
            raise ValueError(f"Invalid `stride_shape` value {stride_shape}.")

        img_path = pathlib.Path(img_path)
        if mode == "wsi":
            self.reader = get_wsireader(img_path)
        else:
            warnings.warn(
                (
                    "WSIPatchDataset only reads image tile at "
                    '`units="baseline"` and `resolution=1.0`.'
                )
            )
            units = "baseline"
            resolution = 1.0
            img = imread(img_path)
            axes = "YXS"[: len(img.shape)]
            # initialise metadata for VirtualWSIReader.
            # here, we simulate a whole-slide image, but with a single level.
            # ! should we expose this so that use can provide their metadata ?
            metadata = WSIMeta(
                mpp=np.array([1.0, 1.0]),
                axes=axes,
                objective_power=10,
                slide_dimensions=np.array(img.shape[:2][::-1]),
                level_downsamples=[1.0],
                level_dimensions=[np.array(img.shape[:2][::-1])],
            )
            # hack value such that read if mask is provided is through
            # 'mpp' or 'power' as varying 'baseline' is locked atm
            units = "mpp"
            resolution = 1.0
            self.reader = VirtualWSIReader(
                img,
                info=metadata,
            )

        # may decouple into misc ?
        # the scaling factor will scale base level to requested read resolution/units
        wsi_shape = self.reader.slide_dimensions(resolution=resolution, units=units)

        # use all patches, as long as it overlaps source image
        self.inputs = PatchExtractor.get_coordinates(
            image_shape=wsi_shape,
            patch_input_shape=patch_input_shape[::-1],
            stride_shape=stride_shape[::-1],
            input_within_bound=False,
        )

        mask_reader = None
        if mask_path is not None:
            if not os.path.isfile(mask_path):
                raise ValueError("`mask_path` must be a valid file path.")
            mask = imread(mask_path)  # assume to be gray
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask = np.array(mask > 0, dtype=np.uint8)

            mask_reader = VirtualWSIReader(mask)
            mask_reader.info = self.reader.info
        elif auto_get_mask and mode == "wsi" and mask_path is None:
            # if no mask provided and `wsi` mode, generate basic tissue
            # mask on the fly
            mask_reader = self.reader.tissue_mask(method=auto_mask_method, resolution=1.25, units="power", **masker_kwargs)
            # ? will this mess up  ?
            mask_reader.info = self.reader.info

        self.mask = mask_reader.img

        if mask_reader is not None:
            selected = PatchExtractor.filter_coordinates(
                mask_reader,  # must be at the same resolution
                self.inputs,  # must already be at requested resolution
                resolution=resolution,
                units=units,
                thres=thres
            )
            self.inputs = self.inputs[selected]

        if len(self.inputs) == 0:
            raise ValueError("No coordinate remain after tiling!")

        self.patch_input_shape = patch_input_shape
        self.resolution = resolution
        self.units = units

        # Perform check on the input
        self._check_input_integrity(mode="wsi")

    def __getitem__(self, idx):
        coords = self.inputs[idx]
        # Read image patch from the whole-slide image
        patch = self.reader.read_bounds(
            coords,
            resolution=self.resolution,
            units=self.units,
            pad_constant_values=255,
            coord_space="resolution",
        )

        # Apply preprocessing to selected patch
        patch = self._preproc(patch)

        if self.label is not None:
            return patch, self.label
        else:
            data = {"image": patch, "coords": np.array(coords)}
            return data


class WSILabelPatchDataset(abc.PatchDatasetABC):
    """Defines a WSI-level patch dataset.

    Attributes:
        reader (:class:`.WSIReader`): an WSI Reader or Virtual Reader
          for reading pyramidal image or large tile in pyramidal way.
        inputs: List of coordinates to read from the `reader`,
          each coordinate is of the form [start_x, start_y, end_x, end_y].
        patch_input_shape: a tuple(int, int) or ndarray of shape (2,).
          Expected size to read from `reader` at requested `resolution`
          and `units`. Expected to be (height, width).
        resolution: check (:class:`.WSIReader`) for details.
        units: check (:class:`.WSIReader`) for details.
        preproc_func: Preprocessing function used to transform the input data.
          If supplied, then torch.Compose will be used on the input preprocs.
          preprocs is a list of torchvision transforms for preprocessing the
          image. The transforms will be applied in the order that they are
          given in the list. For more information, visit the following link:
          https://pytorch.org/vision/stable/transforms.html.

    """

    def __init__(
        self,
        img_paths,
        img_labels=None,
        class_dict=None,
        mode="wsi",
        mask_paths=None,
        save_output=True,
        save_dir=None,
        patch_input_shape=None,
        stride_shape=None,
        resolution=None,
        units=None,
        ignore_background=True,
        background_idx=-1,
        thres=0.5,
        auto_get_mask=True,
        auto_mask_method="otsu",
        **masker_kwargs,
    ):
        """Create a WSI-level patch dataset.
        Args:
            mode (str): can be either `wsi` or `tile` to denote the image to read is
              either a whole-slide image or a large image tile.
            img_path (:obj:`str` or :obj:`pathlib.Path`): valid to pyramidal
              whole-slide image or large tile to read.
            mask_path (:obj:`str` or :obj:`pathlib.Path`): valid mask image.
            patch_input_shape: a tuple (int, int) or ndarray of shape (2,).
              Expected shape to read from `reader` at requested `resolution` and
              `units`. Expected to be positive and of (height, width). Note, this
              is not at `resolution` coordinate space.
            stride_shape: a tuple (int, int) or ndarray of shape (2,).
              Expected stride shape to read at requested `resolution` and `units`.
              Expected to be positive and of (height, width). Note, this is not at
              level 0.
            resolution: check (:class:`.WSIReader`) for details. When `mode='tile'`,
              value is fixed to be `resolution=1.0` and `units='baseline'`
              units: check (:class:`.WSIReader`) for details.
            preproc_func: Preprocessing function used to transform the input data.

        Examples:
            >>> # an user defined preproc func and expected behavior
            >>> preproc_func = lambda img: img/2  # reduce intensity by half
            >>> transformed_img = preproc_func(img)
            >>> # create a dataset to get patches from WSI with above
            >>> # preprocessing function
            >>> ds = WSIPatchDataset(
            ...     img_path='/A/B/C/wsi.svs',
            ...     mode="wsi",
            ...     patch_input_shape=[512, 512],
            ...     stride_shape=[256, 256],
            ...     auto_get_mask=False,
            ...     preproc_func=preproc_func
            ... )

        """
        super().__init__()
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.class_dict = class_dict
        self.save_dir = save_dir
        self.reader = []
        self.inputs = []
        self.masks = []
        self.idx_map = []
        self.num_patches = []
        self.num_tissue_patches = []
        self.patch_input_shape = patch_input_shape
        self.stride_shape = stride_shape
        self.resolution = resolution
        self.units = units
        self.ignore_background = ignore_background
        self.background_idx = background_idx
        self.wsi_labels = list(img_labels)

        if save_output:

            if self.save_dir is None:
                self.save_dir = os.path.join(os.path.dirname(img_path), "out_data")

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        if class_dict is None:
            class_dict= {l: l for _ in set(img_labels)}

        if mask_paths is None:
            mask_paths = [None for _ in range(len(img_paths))]

        if not len(mask_paths)==len(img_paths):
                raise ValueError("number of `mask_paths` must match the number of `img_paths`.")

        for img_idx, (img_path, mask_path, label) in enumerate(zip(img_paths, mask_paths, img_labels)):
            print("{} | Working on \"{}\" | Label: \"{}\" \t({}/{})".format(datetime.now().strftime("%X"), img_path, class_dict[label], img_idx+1, len(img_labels)))

            img_code = os.path.basename(img_path)
            save_path = os.path.join(str(save_dir), img_code)
            raw_save_path = f"{save_path}.raw.json"

            if os.path.isfile(raw_save_path):
                print("Found data of previous run, using existing data... to avoid this behaviour in the future, delete \"{}\"".format(raw_save_path))
                with open(raw_save_path, 'r') as f:
                    json_data = json.loads(f.read())
                self.reader.append(get_wsireader(img_path))
                self.inputs.append(json_data["coords"])
                self.img_labels[img_idx] = json_data["label"]

                if self.resolution is not None and self.resolution != json_data["resolution"] or self.units is not None and self.units != json_data["units"]:
                    print("Warning: different resolutions detected, using most recent...")
                    self.resolution = json_data["resolution"]
                    self.units = json_data["units"]

                if self.patch_input_shape != json_data["patch_shape"] or self.stride_shape!=json_data["stride_shape"]:
                    print("Warning: different patch/stride shapes detected, using most recent...")
                    self.patch_input_shape = json_data["patch_shape"]
                    self.stride_shape = json_data["stride_shape"]

                self.idx_map = np.append(self.idx_map, np.full(len(self.inputs[img_idx]), img_idx)).astype(int)
                
                self.num_patches.append(len(self.inputs[img_idx]))
                if type(self.img_labels[img_idx]) != type(3):
                    self.num_tissue_patches.append(self.img_labels[img_idx].count(label))
                else: self.num_tissue_patches.append(len(self.inputs[img_idx]))

                self.masks.append(None)

                continue

            # Is there a generic func for path test in toolbox?
            if not os.path.isfile(img_path):
                raise ValueError("`img_path` must be a valid file path.")
            if mode not in ["wsi", "tile"]:
                raise ValueError(f"`{mode}` is not supported.")
            patch_input_shape = np.array(patch_input_shape)
            stride_shape = np.array(stride_shape)

            if (
                not np.issubdtype(patch_input_shape.dtype, np.integer)
                or np.size(patch_input_shape) > 2
                or np.any(patch_input_shape < 0)
            ):
                raise ValueError(f"Invalid `patch_input_shape` value {patch_input_shape}.")
            if (
                not np.issubdtype(stride_shape.dtype, np.integer)
                or np.size(stride_shape) > 2
                or np.any(stride_shape < 0)
            ):
                raise ValueError(f"Invalid `stride_shape` value {stride_shape}.")
            if mode == "wsi":
                self.reader.append(get_wsireader(img_path))
            else:
                warnings.warn(
                    (
                        "WSIPatchDataset only reads image tile at "
                        '`units="baseline"` and `resolution=1.0`.'
                    )
                )
                units = "baseline"
                resolution = 1.0
                img = imread(img_path)
                axes = "YXS"[: len(img.shape)]
                # initialise metadata for VirtualWSIReader.
                # here, we simulate a whole-slide image, but with a single level.
                # ! should we expose this so that use can provide their metadata ?
                metadata = WSIMeta(
                    mpp=np.array([1.0, 1.0]),
                    axes=axes,
                    objective_power=10,
                    slide_dimensions=np.array(img.shape[:2][::-1]),
                    level_downsamples=[1.0],
                    level_dimensions=[np.array(img.shape[:2][::-1])],
                )
                # hack value such that read if mask is provided is through
                # 'mpp' or 'power' as varying 'baseline' is locked atm
                units = "mpp"
                resolution = 1.0
                self.reader.append(
                    VirtualWSIReader(
                    img,
                    info=metadata,
                    )
                )
            # may decouple into misc ?
            # the scaling factor will scale base level to requested read resolution/units
            wsi_shape = self.reader[img_idx].slide_dimensions(resolution=resolution, units=units)

            # use all patches, as long as it overlaps source image
            self.inputs.append(
                PatchExtractor.get_coordinates(
                image_shape=wsi_shape,
                patch_input_shape=patch_input_shape[::-1],
                stride_shape=stride_shape[::-1],
                input_within_bound=False,
                )
            )

            mask_reader = None
            if mask_path is not None:
                if not os.path.isfile(mask_path):
                    raise ValueError("`mask_path` must be a valid file path.")
                mask = imread(mask_path)  # assume to be gray
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                mask = np.array(mask > 0, dtype=np.uint8)

                mask_reader = VirtualWSIReader(mask)
                mask_reader.info = self.reader.info
            elif auto_get_mask and mode == "wsi" and mask_path is None:
                # if no mask provided and `wsi` mode, generate basic tissue
                # mask on the fly
                mask_reader = self.reader[img_idx].tissue_mask(method=auto_mask_method, resolution=1.25, units="power", **masker_kwargs)
                # ? will this mess up  ?
                mask_reader.info = self.reader[img_idx].info

            self.masks.append(mask_reader.img)

            if mask_reader is not None:
                selected = PatchExtractor.filter_coordinates(
                    mask_reader,  # must be at the same resolution
                    self.inputs[img_idx],  # must already be at requested resolution
                    resolution=resolution,
                    units=units,
                    thres=thres,
                )
                if self.ignore_background:
                    self.inputs[img_idx] = self.inputs[img_idx][selected]
                    self.num_patches.append(len(self.inputs[img_idx]))
                    self.num_tissue_patches.append(len(self.inputs[img_idx]))
                else:
                    self.img_labels[img_idx] = np.full(len(self.inputs[img_idx]), background_idx)
                    self.img_labels[img_idx][selected] = label
                    self.num_patches.append(self.inputs[img_idx])
                    self.num_tissue_patches.append(self.img_labels[img_idx][selected])


            if len(self.inputs[img_idx]) == 0:
                raise ValueError("No coordinate remain after tiling!")
            self.idx_map = np.append(self.idx_map, np.full(len(self.inputs[img_idx]), img_idx)).astype(int)

            if save_output:
                
                #save_info["raw"] = raw_save_path
                output = {
                    #"path": img_path,
                    #"wsi_dims": wsi_shape,
                    "coords": self.inputs[img_idx],
                    "label": self.img_labels[img_idx],
                    #"num_patches": self.num_patches[img_idx],
                    "resolution": self.resolution,
                    "units": self.units,
                    #"mask": self.masks[img_idx],
                    #"thumbnail": self.reader[img_idx].thumbnail,
                    "patch_shape": self.patch_input_shape,
                    "stride_shape": self.stride_shape,
                    }
                save_as_json(output, raw_save_path)
                self.overlay_patches(img_idx)
                #file_dict[str(img_path)] = save_info
        
        #self.inputs = np.vstack(inputs)

        # Perform check on the input
        self._check_input_integrity(mode="wsi")
        print("{} | Done".format(datetime.now().strftime("%X")))

    def overlay_patches(self, img_id):
        fig, ax = plt.subplots()
        mask = self.masks[img_id]
        mask = np.ma.masked_where(mask>0.0, 1-mask)
        img = self.reader[img_id].thumbnail
        ax.imshow(img)
        ax.imshow(mask, alpha=0.8, cmap='Reds', vmin=0, vmax=1.3)
        ax.title.set_text("\"{}\"\nLabel: \"{}\" | Patch number /w tissue: {} | Patch number in total: {}".format(self.img_paths[img_id], self.class_dict[self.wsi_labels[img_id]], self.num_tissue_patches[img_id], self.num_patches[img_id])) #""", self.class_dict[self.wsi_labels[img_id]]"""
        scale = self.masks[img_id].shape[0]/self.reader[img_id].slide_dimensions(resolution=self.resolution, units=self.units)[1]
        for i, (x, y) in enumerate(zip(self.inputs[img_id][:,0]*scale, self.inputs[img_id][:,1]*scale)):
            if type(self.img_labels[img_id]) == type(3) or self.img_labels[img_id][i] != self.background_idx:
                rect=mpl.patches.Rectangle(
                    xy=(x, y), width=self.patch_input_shape[0]*scale,
                    height=self.patch_input_shape[1]*scale,
                    linewidth=0.5, edgecolor='g', facecolor='none'
                    ) 
                ax.add_patch(rect)
        img_path = self.img_paths[img_id]
        img_code = os.path.basename(img_path)
        save_path = os.path.join(str(self.save_dir), img_code+"_sum.png")
        plt.savefig(save_path)
        plt.clf()

    def __getitem__(self, idx):
        img_idx = self.idx_map[idx]
        patch_idx = idx - sum(self.num_patches[:img_idx])
        coords = self.inputs[img_idx][patch_idx]
        if type(self.img_labels[img_idx]) != type(3):
            label = self.img_labels[img_idx][patch_idx]
        else:
            label = self.img_labels[img_idx]
        # Read image patch from the whole-slide image
        patch = self.reader[img_idx].read_bounds(
            coords,
            resolution=self.resolution,
            units=self.units,
            pad_constant_values=255,
            coord_space="resolution",
        )

        # Apply preprocessing to selected patch
        patch = self._preproc(patch)
        if self.img_labels is None:
            data = {"image": patch, "coords": np.array(coords), "img_id": img_idx}
            return data
        else:
            return patch, label

    def __len__(self):
        return sum(self.num_patches)
