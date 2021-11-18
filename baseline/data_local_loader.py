from torch.utils import data
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import pandas as pd
# from torchvision.io import read_image
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sklearn.model_selection import train_test_split


def aug_transform(image1, image2, image3, random_crop=True, mode='2d'):
    """
    albumentations transform. Crop part of image containing hand(s).
    If keypoints labeling is wrong, replace crop with random crop.
    """
    augmentations = [A.Resize(width=256, height=256, p=1)]
    if random_crop:
        augmentations.extend([A.RandomResizedCrop(width=224, height=224, p=1),
                              A.HorizontalFlip(p=0.25),
                              A.ShiftScaleRotate(shift_limit=0.10,
                                                 scale_limit=0.10,
                                                 rotate_limit=5,
                                                 border_mode=cv2.BORDER_CONSTANT,
                                                 p=0.5),
                              A.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                          std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
                              ToTensorV2(),
                              ])
    else:
        augmentations.append(A.CenterCrop(width=224, height=224, p=1))

    transform = A.Compose(augmentations, additional_targets={'image2':'image', 'image3':'image'})
    transformed = transform(image=image1, image2=image2, image3=image3)
    image1 = transformed['image']
    image2 = transformed['image2']
    image3 = transformed['image3']

    if mode == '3d':
        image = torch.stack([image1, image2, image3], dim=1)
    elif mode == '2d':
        image = torch.cat([image1, image2, image3])

    return image


def get_transform(random_crop=True):
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = []
    transform.append(transforms.Resize(256))
    if random_crop:
        transform.append(transforms.RandomResizedCrop(224))
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2)))
    else:
        transform.append(transforms.CenterCrop(224))
    transform.append(normalize)
    return transforms.Compose(transform)


class CustomDataset(data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, split='train', mode='2d'):
        self.img_labels = dict([l.replace('\n', '').split(' ') for l in open(annotations_file).readlines()])
        self.img_dir = img_dir
        self.transform = transform
        self.split = split  # train or val
        self.mode = mode  # 2d or 3d

    def __len__(self):
        return len(self.img_labels)

    def set_mode(self, mode):
        self.mode = mode

    def set_split(self, split):
        self.split = split

    def __getitem__(self, idx):
        img_origin_path = list(self.img_labels.keys())[idx]
        img_path_S1 = os.path.join(self.img_dir, img_origin_path) + '-S01.jpg'
        img_path_M1 = os.path.join(self.img_dir, img_origin_path) + '-M01.jpg'
        img_path_E1 = os.path.join(self.img_dir, img_origin_path) + '-E01.jpg'
        image_id = img_path_S1.split('-')[0].split('/')[-1]
        image_S1 = cv2.cvtColor(cv2.imread(img_path_S1), cv2.COLOR_BGR2RGB)
        image_M1 = cv2.cvtColor(cv2.imread(img_path_M1), cv2.COLOR_BGR2RGB)
        image_E1 = cv2.cvtColor(cv2.imread(img_path_E1), cv2.COLOR_BGR2RGB)
        label = self.img_labels[img_origin_path]

        if self.mode == '2d':
            if self.split == 'train':
                image = aug_transform(image_S1, image_M1, image_E1, random_crop=True, mode='2d')
            elif self.split == 'val':
                image = aug_transform(image_S1, image_M1, image_E1, random_crop=False, mode='2d')
        elif self.mode == '3d':
            if self.split == 'train':
                image = aug_transform(image_S1, image_M1, image_E1, random_crop=True, mode='3d')
            elif self.split == 'val':
                image = aug_transform(image_S1, image_M1, image_E1, random_crop=False, mode='3d')
        label = int(label)

        return image_id, image, label


class TestDataset(data.Dataset):
    def __init__(self, root, transform=None, mode='2d'):
        self.image_dir = root.replace('label','data')
        self.image_list = sorted(list(set(map(lambda x: x.split('-')[0], os.listdir(root)))))
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_id  = self.image_list[idx]
        img_path_S1 = os.path.join(self.image_dir, img_id) + '-S01.jpg'
        img_path_M1 = os.path.join(self.image_dir, img_id) + '-M01.jpg'
        img_path_E1 = os.path.join(self.image_dir, img_id) + '-E01.jpg'
        image_S1 = cv2.cvtColor(cv2.imread(img_path_S1), cv2.COLOR_BGR2RGB)
        image_M1 = cv2.cvtColor(cv2.imread(img_path_M1), cv2.COLOR_BGR2RGB)
        image_E1 = cv2.cvtColor(cv2.imread(img_path_E1), cv2.COLOR_BGR2RGB)

        if self.mode == '2d':
            image = aug_transform(image_S1, image_M1, image_E1, random_crop=False, mode='2d')
        elif self.mode == '3d':
            image = aug_transform(image_S1, image_M1, image_E1, random_crop=False, mode='3d')

        return img_id, image


def test_data_loader(root, phase='train', batch_size=64):
    if phase == 'train':
        is_train = True
    elif phase == 'test':
        is_train = False
    else:
        raise KeyError

    dataset = TestDataset(
        root.replace('label','data'),
        transform=get_transform(random_crop=is_train)
    )
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=is_train)


def data_loader_with_split(root, train_split=0.9, batch_size=64, val_label_file='./val_label', data_mode='2d'):
    print("loading train and validation data...")
    if root[-1]=='/':
        mode = root.split('/')[-2]
    else:
        mode = root.split('/')[-1]
    dataset = CustomDataset(
        os.path.join(root, mode+'_label'),
        os.path.join(root, mode+'_data'),
        transform=True,
    )
    print()
    split_size = int(len(dataset) * train_split)
    train_set, valid_set = data.random_split(dataset, [split_size, len(dataset) - split_size])
    print("train set contains {} samples".format(len(train_set)))
    print("validation set contains {} samples".format(len(valid_set)))

    #train_set.set_split('train')
    #valid_set.set_split('val')
    #print("data mode: {}".format(data_mode))
    #if data_mode == '2d':
    #    train_set.set_mode('2d')
    #    valid_set.set_mode('2d')
    #elif data_mode == '3d':
    #    train_set.set_mode('3d')
    #    valid_set.set_mode('3d')
    #else:
    #    raise ValueError('data_mode mst be either 2d or 3d')

    tr_loader = data.DataLoader(dataset=train_set,
                                batch_size=batch_size,
                                num_workers=4,
                                shuffle=True)
    val_loader = data.DataLoader(dataset=valid_set,
                                 batch_size=batch_size,
                                 num_workers=4,
                                 shuffle=False)


    print('generate val labels')
    gt_labels = {valid_set[idx][0]: valid_set[idx][2] for idx in tqdm(range(len(valid_set)))}
    gt_labels_string = [' '.join([str(s) for s in l]) for l in tqdm(list(gt_labels.items()))]
    with open(val_label_file, 'w') as file_writer:
        file_writer.write("\n".join(gt_labels_string))


    return tr_loader, val_loader, val_label_file
