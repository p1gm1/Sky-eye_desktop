"""Handles the image data
parsed in the form.
"""

# Torch
import torch
from torch.utils.data import Dataset

# pytorch-lighting
import pytorch_lightning as pl

# Third-party
import numpy as np
import pandas as pd

# albumentations
from albumentations.core.composition import Compose

# Omegaconf
from omegaconf import DictConfig

# utils
from utils import load_obj

# Local
import os
import cv2


class ImgDataset(Dataset):

    def __init__(self,
                 dataframe: pd.DataFrame = None,
                 mode: str = 'train',
                 image_dir: str = '',
                 cfg: DictConfig = None,
                 transforms: Compose = None):
        """

        Args:
            dataframe: dataframe with image id and bboxes
            mode: train/val/test
            cfg: config with parameters
            image_dir: path to images
            transforms: albumentations
        """
        self.image_dir = image_dir
        self.df = dataframe
        self.mode = mode
        self.cfg = cfg
        self.image_ids = os.listdir(self.image_dir) if self.df is None else self.df['image_id'].unique()
        self.transforms = transforms

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx].split('.')[0]
        #print(image_id)
        image = cv2.imread(f'{self.image_dir}/{image_id}.JPG', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # normalization.
        image /= 255.0

        # test dataset must have some values so that transforms work.
        target = {'labels': torch.as_tensor([[0]], dtype=torch.float32),
                  'boxes': torch.as_tensor([[0, 0, 0, 0]], dtype=torch.float32)}

        # for train and valid test create target dict.
        if self.mode != 'test':
            image_data = self.df.loc[self.df['image_id'] == image_id]
            boxes = image_data[['x', 'y', 'x1', 'y1']].values

            areas = image_data['area'].values
            areas = torch.as_tensor(areas, dtype=torch.float32)

            # there is only one class
            labels = torch.ones((image_data.shape[0],), dtype=torch.int64)
            iscrowd = torch.zeros((image_data.shape[0],), dtype=torch.int64)

            target['boxes'] = boxes
            target['labels'] = labels
            target['image_id'] = torch.tensor([idx])
            target['area'] = areas
            target['iscrowd'] = iscrowd

            if self.transforms:
                image_dict = {
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                }
                image_dict = self.transforms(**image_dict)
                image = image_dict['image']
                target['boxes'] = torch.as_tensor(image_dict['bboxes'], 
                                                  dtype=torch.float32)

        else:
            image_dict = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': target['labels']
            }
            image = self.transforms(**image_dict)['image']

        return image, target, image_id

    def __len__(self) -> int:
        return len(self.image_ids)


class LitImg(pl.LightningModule):

    def __init__(self, hparams: DictConfig = None, cfg: DictConfig = None, model = None):
        super(LitImg, self).__init__()
        self.cfg = cfg
        self.hparams = hparams
        self.model = model

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        if 'decoder_lr' in self.cfg.optimizer.params.keys():
            params = [
                {'params': self.model.decoder.parameters(), 'lr': self.cfg.optimizer.params.lr},
                {'params': self.model.encoder.parameters(), 'lr': self.cfg.optimizer.params.decoder_lr},
            ]
            optimizer = load_obj(self.cfg.optimizer.class_name)(params)

        else:
            optimizer = load_obj(self.cfg.optimizer.class_name)(self.model.parameters(), **self.cfg.optimizer.params)
        scheduler = load_obj(self.cfg.scheduler.class_name)(optimizer, **self.cfg.scheduler.params)

        return [optimizer], [{"scheduler": scheduler,
                              "interval": self.cfg.scheduler.step,
                              'monitor': self.cfg.scheduler.monitor}]