import lmdb
import sys
import six
from torch.utils.data import Dataset
from PIL import Image

import torch
import numpy as np
from torchvision import transforms
from imgaug import augmenters as iaa


class lmdbDataset(Dataset):
    """LMDB dataset for raw images.

    Args:
        root (str): Root path for lmdb files.
        transform (callable, optional): A function/transform that  takes in an
            PIL image and returns a transformed version.
    """

    def __init__(self, root: str = None, transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
        self.transform = transform

        self.augmentor = self.sequential_aug()
        mean = std = 0.5
        self.aug_transformer = transforms.Compose([
                transforms.Resize((32, 128), interpolation=3),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ])

    def __len__(self):
        return self.nSamples
    
    def sequential_aug(self):
        aug_transform = transforms.Compose([
        iaa.Sequential(
            [
            iaa.SomeOf((3, None),
            [
                iaa.LinearContrast((0.5, 1.0)),
                iaa.GaussianBlur((0.5, 1.5)),
                iaa.Crop(percent=((0, 0.1),
                                (0, 0.0),
                                (0, 0.1),
                                (0, 0.0)),
                                keep_size=True), # top and down
                iaa.Crop(percent=((0, 0.0),
                                (0, 0.02),
                                (0, 0.0),
                                (0, 0.02)),
                                keep_size=True),  # left and right
                iaa.Sharpen(alpha=(0.0, 0.5),
                            lightness=(0.0, 0.5)),
                # iaa.AdditiveGaussianNoise(scale=(0, 0.15*255), per_channel=True),
                iaa.Rotate((-15, 15), fit_output=True),  # can reach 15
                # iaa.Cutout(nb_iterations=1, size=(0.15, 0.25), squared=True),
                iaa.PiecewiseAffine(scale=(0.03, 0.05), mode='edge'),  # 0.03-0.05
                iaa.PerspectiveTransform(scale=(0.02, 0.06)),  #crop 0.1 too large 0.02-0.06
                iaa.Solarize(1, threshold=(32, 128), invert_above_threshold=0.5, per_channel=False),
                iaa.Grayscale(alpha=(0.0, 1.0)),
            ],
            random_order=True)
            ]
        ).augment_image,
        ])
        return aug_transform

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            # iaa aug
            aug_img = self.augmentor(np.asarray(img))
            aug_img = Image.fromarray(np.uint8(aug_img))
            aug_img = self.aug_transformer(aug_img)

            img = self.transform(img)

        return img, aug_img, 'test'
