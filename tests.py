"""Module of unitary tests."""

# Unitests
import unittest

# Utils
from utils import get_cfg, load_obj, collate_fn

# Albumentations
import albumentations as A

# Machine Learning
from ml import ImgDataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


class TestCounter(unittest.TestCase):
    """Handles the test for the view
    of the counter.
    """

    def setUp(self) -> None:
        """Set't up the tests
        variables.
        """

        imageDir = '/home/p1gm1/Documents/DroneSky/test_img/oil_palm'

        cfg = get_cfg()

        valid_augs_list = [load_obj(i['class_name'])(**i['params']) for i in cfg['augmentation']['valid']['augs']]
        valid_bbox_params = OmegaConf.to_container((cfg['augmentation']['valid']['bbox_params']))
        valid_augs = A.Compose(valid_augs_list, bbox_params=valid_bbox_params)
        
        self.test_dataset = ImgDataset(None,
                                 'test',
                                 imageDir,
                                 cfg,
                                 valid_augs)
        
        self.test_loader = DataLoader(self.test_dataset,
                                 batch_size=cfg.data.batch_size,
                                 num_workers=cfg.data.num_workers,
                                 shuffle=False,
                                 collate_fn=collate_fn)

    def test_dataset_exists(self):
        """Test if the dataset is created."""
        self.assertIsNotNone(self.test_dataset)

    def test_dataloader_exists(self):
        """Test if the dataloader is created."""
        self.assertIsNotNone(self.test_loader)

    def test_identity_of_image_id(self):
        """Test the identity of the var
        image_id.
        """
        for images, _, image_ids in self.test_loader:

            for i, image in enumerate(images):
                self.image_id = image_ids[i]
                
        self.assertEqual(self.image_id, 'DJI_0108')

if __name__=='__main__':
    unittest.main()