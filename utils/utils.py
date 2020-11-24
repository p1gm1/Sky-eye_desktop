"""Handles the utilities
to handle machine learning 
data."""

# Local
import importlib
from typing import Any
import collections
import random
import cv2

# Data
import numpy as np

# Omegaconf
from omegaconf import OmegaConf

# torch
import torch

# Torchvision
import torchvision.transforms as transforms

def get_cfg():
    """Get cfg object, with the preloaded
    configs.
    """

    cfg =  {'logging': {'log': True},
                        'data': { 'num_workers': 0, 
                        'batch_size': 12},
            'augmentation': {'valid': {'augs': [{'class_name': 'albumentations.pytorch.transforms.ToTensorV2', 
                                                    'params': {'p': 1.0}}],
                                        'bbox_params': {'format': 'pascal_voc', 
                                                        'label_fields': ['labels']}}}}

    return OmegaConf.create(cfg)


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, 
            including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have 
            the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path  
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`."
        )
    return getattr(module_obj, obj_name)


def collate_fn(batch):
    """Handles the pack of
    batches in a dataloder.
    """
    return tuple(zip(*batch))


def flatten_omegaconf(d, sep="_"):
    """Unroll the omegaconf."""
    d = OmegaConf.to_container(d)

    obj = collections.OrderedDict()

    def recurse(t, parent_key=""):

        if isinstance(t, list):
            for i in range(len(t)):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)
    obj = {k: v for k, v in obj.items() if type(v) in [int, float]}

    return obj


def eval_model(test_loader, 
               results, 
               detection_threshold, 
               device, 
               lit_model):
    """Evaluates an image on
    the model.
    """
    for images, _, image_ids in test_loader:

        images = list(image.to(device) for image in images)
        outputs = lit_model(images)

        for i, image in enumerate(images):

            boxes = outputs[i]['boxes'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()
            
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold]
            image_id = image_ids[i]

            for s, b in zip(scores, boxes.astype(int)):
                result = {
                'image_id': image_id,
                'x1': b[0],
                'y1': b[1],
                'x2': b[2],
                'y2': b[3],
                'score': s,
                }

            
                results.append(result)
    
    return results