import glob
import random

import torchvision
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class DDTI(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_path = os.path.join(self.data_dir, 'p_image')
        self.mask_path = os.path.join(self.data_dir, 'p_mask')
        self.image = os.listdir(self.image_path)
        self.transform = torchvision.transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
        )

    def __getitem__(self, item):
        img_path = os.path.join(self.image_path, self.image[item])
        mask_path = os.path.join(self.mask_path, self.image[item])
        img = Image.open(img_path).convert('RGB')

        mask = Image.open(mask_path)

        img = self.transform(img)
        mask = self.transform(mask)
        return img, mask

    def __len__(self):
        return len(self.image)


class ACDC(Dataset):
    def __init__(self, img_path, mask_path):
        super(ACDC, self).__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.img = os.listdir(self.img_path)
        self.transform = torchvision.transforms.Compose([
            # transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.ToTensor(),

        ]
        )

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_path, self.img[item])
        mask_path = os.path.join(self.mask_path, self.img[item])
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        img, mask = Image.open(img_path).convert("RGB"), Image.open(mask_path)
        mask = np.array(mask)
        mask[mask == 85] = 1
        mask[mask == 170] = 2
        mask[mask == 255] = 3
        mask = Image.fromarray(mask)
        img = self.transform(img)
        torch.random.manual_seed(seed)
        mask = self.transform(mask)
        return img, mask * 255


class ACDC_test(Dataset):
    def __init__(self, img_path, mask_path):
        super(ACDC_test, self).__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.img = os.listdir(self.img_path)
        # self.img.remove('.ipynb_checkpoints')
        self.transform = torchvision.transforms.Compose([
            transforms.ToTensor(),

        ]
        )

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_path, self.img[item])
        mask_path = os.path.join(self.mask_path, self.img[item])
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        img, mask = Image.open(img_path).convert("RGB"), Image.open(mask_path)
        img0 = img
        mask = np.array(mask)
        mask[mask == 85] = 1
        mask[mask == 170] = 2
        mask[mask == 255] = 3
        mask = Image.fromarray(mask)
        img = self.transform(img)
        torch.random.manual_seed(seed)
        mask = self.transform(mask)
        return img, mask * 255, img0

class Masks(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.vids = os.listdir(data_dir)
        self.transform = torchvision.transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
        )

    def __getitem__(self, index):
        vid = os.path.join(self.data_dir, self.vids[index])

        frames_paths = sorted(glob.glob(vid + '/Frames' + '/*.png'))
        masks_paths = sorted(glob.glob(vid + '/Masks' + '/*.png'))
        nframes = len(frames_paths)
        item = random.sample(list(range(1, nframes)), 1)
        img_source = Image.open(frames_paths[item[0]]).convert('RGB')
        img_target = Image.open(frames_paths[0]).convert('RGB')
        mask = Image.open(masks_paths[0]).convert('L')
        img_source = self.transform(img_source)
        img_target = self.transform(img_target)
        mask = self.transform(mask)
        return torch.stack((img_source, img_target)), mask
        # return img_source, img_target,

    def __len__(self):
        return len(self.vids)


# class Piccolo_train(Dataset):
#     def __init__(self, train_path):
#         super().__init__()
#         self.base_path = train_path
#         self.polyps = sorted(os.listdir(os.path.join(self.base_path, 'polyps')))
#         self.masks = sorted(os.listdir(os.path.join(self.base_path, 'masks')))
#         self.transform = torchvision.transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomVerticalFlip(p=0.5),
#             transforms.RandomRotation(45),
#             transforms.ToTensor(),

#         ])

#     def __len__(self):
#         return self.polyps.__len__()

#     def __getitem__(self, item):
#         img_path = os.path.join(os.path.join(self.base_path, 'polyps'), self.polyps[item])
#         mask_path = os.path.join(os.path.join(self.base_path, 'masks'), self.masks[item])
#         img, mask = Image.open(img_path), Image.open(mask_path)
#         seed = torch.random.seed()
#         torch.random.manual_seed(seed)
#         img = self.transform(img)
#         torch.random.manual_seed(seed)
#         mask = self.transform(mask)
#         img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
#         return img, mask


class Piccolo_train(Dataset):
    def __init__(self, train_path, val_path):
        super().__init__()
        self.base_path = train_path
        self.base_val_path = val_path
        self.polyps = []
        for i in os.listdir(os.path.join(self.base_path, 'polyps')):
            self.polyps.append(os.path.join(self.base_path, os.path.join('polyps', i)))
        for j in os.listdir(os.path.join(self.base_val_path, 'polyps')):
            self.polyps.append(os.path.join(self.base_val_path, os.path.join('polyps', j)))
        self.polyps = sorted(self.polyps)
        self.masks = []
        for i in os.listdir(os.path.join(self.base_path, 'masks')):
            self.masks.append(os.path.join(os.path.join(self.base_path, 'masks'), i))
        for j in os.listdir(os.path.join(self.base_val_path, 'masks')):
            self.masks.append(os.path.join(self.base_val_path, os.path.join('masks', j)))
        self.masks = sorted(self.masks)
        self.transform = torchvision.transforms.Compose([
            transforms.Resize((288, 288)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.polyps.__len__()

    def __getitem__(self, item):
        img_path = self.polyps[item]
        mask_path = self.masks[item]
        img, mask = Image.open(img_path), Image.open(mask_path)
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        img = self.transform(img)
        torch.random.manual_seed(seed)
        mask = self.transform(mask)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img, mask


class Piccolo_val(Dataset):
    def __init__(self, val_path):
        super().__init__()
        self.base_path = val_path
        self.polyps = sorted(os.listdir(os.path.join(self.base_path, 'polyps')))
        self.masks = sorted(os.listdir(os.path.join(self.base_path, 'masks')))
        self.transform = torchvision.transforms.Compose([
            transforms.Resize((288, 288)),
            transforms.ToTensor(),

        ])

    def __len__(self):
        return self.polyps.__len__()

    def __getitem__(self, item):
        img_path = os.path.join(os.path.join(self.base_path, 'polyps'), self.polyps[item])
        mask_path = os.path.join(os.path.join(self.base_path, 'masks'), self.masks[item])
        img, mask = Image.open(img_path), Image.open(mask_path)
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        img = self.transform(img)
        mask = self.transform(mask)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img, mask


class Piccolo_test(Dataset):
    def __init__(self, test_path):
        super().__init__()
        self.base_path = test_path
        self.polyps = sorted(os.listdir(os.path.join(self.base_path, 'polyps')))
        self.masks = sorted(os.listdir(os.path.join(self.base_path, 'masks')))
        self.transform = torchvision.transforms.Compose([
            transforms.Resize((288, 288)),
            transforms.ToTensor(),

        ])

    def __len__(self):
        return self.polyps.__len__()

    def __getitem__(self, item):
        img_path = os.path.join(os.path.join(self.base_path, 'polyps'), self.polyps[item])
        mask_path = os.path.join(os.path.join(self.base_path, 'masks'), self.masks[item])
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        img, mask = Image.open(img_path), Image.open(mask_path)
        img = self.transform(img)
        mask = self.transform(mask)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img, mask

class Synapse_train(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = os.listdir(path)
        self.transform = torchvision.transforms.Compose(
            [
                # transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(45),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        path = os.path.join(self.path, self.data[item])
        f = np.load(path)
        img_arr, mask_arr = f['image'], f['label']
        img = Image.fromarray(np.array(img_arr*255).astype(np.uint8)).convert('RGB')
        mask = Image.fromarray(np.array(mask_arr).astype(np.uint8))
        seed = torch.random.seed()
        if torch.rand(1) >= 0.5:
            torch.random.manual_seed(seed)
            img = transforms.Resize((256, 256))(img)
            torch.random.manual_seed(seed)
            mask = transforms.Resize((256, 256))(mask)
        else:
            torch.random.manual_seed(seed)
            img = transforms.RandomCrop((256, 256))(img)
            torch.random.manual_seed(seed)
            mask = transforms.RandomCrop((256, 256))(mask)
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        img = self.transform(img)
        torch.random.manual_seed(seed)
        mask = self.transform(mask)
        return img, mask*255

class Synapse_test(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = os.listdir(path)
        self.transform = torchvision.transforms.Compose(
            [
                # transforms.Resize((256, 256)),
                # transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        path = os.path.join(self.path, self.data[item])
        f = np.load(path)
        img_arr, mask_arr = f['image'], f['label']
        img = Image.fromarray(np.array(img_arr*255).astype(np.uint8)).convert('RGB')
        img0 = img
        mask = Image.fromarray(np.array(mask_arr).astype(np.uint8))
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        img = self.transform(img)
        torch.random.manual_seed(seed)
        mask = self.transform(mask)
        img = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(img)
        return img, mask*255, img0

class Kvasir_train(Dataset):
    def __init__(self,base_path):
        self.base_path = base_path
        self.img_path = os.path.join(base_path, 'images')
        self.mask_path = os.path.join(base_path, 'masks')
        self.img_name = []
        with open(os.path.join(base_path, 'train.txt')) as f:
            for i in f.readlines():
                self.img_name.append(i[:-1])


        self.transform = torchvision.transforms.Compose(
            [
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(30),
                transforms.ToTensor(),

            ]
        )

    def __len__(self):
        return self.img_name.__len__()

    def __getitem__(self, item):
        img_path = os.path.join(self.img_path, self.img_name[item])
        mask_path = os.path.join(self.mask_path, self.img_name[item])
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img = transforms.ColorJitter(brightness=0.6, contrast=0.7, saturation=0.5, hue=0.1)(img)
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        img = self.transform(img)
        torch.random.manual_seed(seed)
        mask = self.transform(mask)
        return img, mask

class Kvasir_test(Dataset):
    def __init__(self, base_path):
        self.base_path = base_path
        self.img_path = os.path.join(base_path, 'images')
        self.mask_path = os.path.join(base_path, 'masks')
        self.img_name = []
        with open(os.path.join(base_path, 'test.txt')) as f:
            for i in f.readlines():
                self.img_name.append(i[:-1])
        self.transform = torchvision.transforms.Compose([
            transforms.ToTensor(),
        ]
        )

    def __len__(self):
        return self.img_name.__len__()

    def __getitem__(self, item):
        img_path = os.path.join(self.img_path, self.img_name[item])
        mask_path = os.path.join(self.mask_path, self.img_name[item])
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img = self.transform(img)
        mask = self.transform(mask)
        return img, mask


if __name__ == "__main__":
    dataset = ACDC("D:/学习/serch/models/model/data/ACDC-2D-All/train/Img", "D:/学习/serch/models/model/data/ACDC-2D-All/train/GT")
    print(dataset[0][0].shape, dataset[0][1].shape, dataset[0][1].max())


