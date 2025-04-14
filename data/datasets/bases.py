from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import numpy as np
import math
from torchvision import transforms
import torchvision.transforms.functional as TF
ImageFile.LOAD_TRUNCATED_IMAGES = True



def read_image(img_list):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    if type(img_list) == type("This is a str"):
        img_path = img_list
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                RGB = img.crop((0, 0, 256, 128))
                NI = img.crop((256, 0, 512, 128))
                TI = img.crop((512, 0, 768, 128))
                img3 = [RGB, NI, TI]
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
    else:
        img3 = []
        for i in img_list:
            img_path = i
            got_img = False
            if not osp.exists(img_path):
                raise IOError("{} does not exist".format(img_path))
            while not got_img:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img3.append(img)
                    got_img = True
                except IOError:
                    print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                    pass
    return img3


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None,flag=1,datasets=None):
        self.dataset = dataset
        self.transform = transform
        self.flag=flag
        self.mixing_erasing= mixing_erasing()
        self.data=datasets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img3 = read_image(img_path)

        if self.transform is not None:
            img = [self.transform(img) for img in img3]
            img1,img2,img3=img[0],img[1],img[2]
            # print(img)

            if self.flag:
               img1,img2,img3 = self.mixing_erasing.trans(img1,img2,img3)
               enhancer = ImageEnhancer(img1,self.data)
               enhancer.to_hsv_and_enhance()
               img1=enhancer.visible
           
            img = [img1,img2,img3]



        return img, pid, camid, trackid, img_path[0].split('/')[-1]




class mixing_erasing(object):
    def __init__(self,
                 probability=0.5,
                 sl=0.01,
                 sh=0.1,
                 r1=0.3,
                 mean=(0.4914, 0.4822, 0.4465),
                 mode='pixel',
                 device='cpu',
                 mixing_coeff=[1.0, 1.0]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.rand_color = False
        self.per_pixel = False
        self.mode = mode
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device
        self.type = type
        self.mixing_coeff = mixing_coeff

        self.to_grayscale = transforms.Grayscale(num_output_channels=3)

    def trans(self, img1, img2,img3):
        if random.uniform(0, 1) >= self.probability:
            return img1,img2,img3
        

        for attempt in range(100):
            area = img1.size()[1] * img1.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img1.size()[2] and h < img1.size()[1]:
                # Application zone
                x1 = random.randint(0, img1.size()[1] - h)
                y1 = random.randint(0, img1.size()[2] - w)
                if self.type == 'normal':
                    m = 1.0
                else: # soft - soft_IR - self
                    m = np.float32(
                        np.random.beta(self.mixing_coeff[0],
                                       self.mixing_coeff[1]))

                store_patch1 = img1[:, x1:x1 + h, y1:y1 + w].clone()  # RGB patch
                store_patch2 = img2[:, x1:x1 + h, y1:y1 + w].clone()
                store_patch3 = img3[:, x1:x1 + h, y1:y1 + w].clone()  # IR patch
                img1[:, x1:x1 + h,
                y1:y1 + w] = (1 - m) * img1[:, x1:x1 + h, y1:y1 +
                                                            w] + m * store_patch2
                img2[:, x1:x1 + h,
                y1:y1 + w] = (1 - m) * img2[:, x1:x1 + h, y1:y1 +
                                                                w] + m * store_patch3
                img3[:, x1:x1 + h,
                y1:y1 + w] = (1 - m) * img3[:, x1:x1 + h, y1:y1 +
                                                                w] + m * store_patch1


                return img1, img2,img3


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def _get_pixels(per_pixel,
                rand_color,
                patch_size,
                dtype=torch.float32,
                device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype,
                           device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


def adjust_range_to_01(tensor):
    return (tensor + 1) / 2

def adjust_range_to_m11(tensor):
    return tensor * 2 - 1

def rgb_to_hsv(image):
    image = adjust_range_to_01(image)  #
    r, g, b = image[0], image[1], image[2]
    max_val, _ = torch.max(image, dim=0)
    min_val, _ = torch.min(image, dim=0)
    diff = max_val - min_val
    

    h = torch.zeros_like(max_val)

    non_zero = diff > 0
    h[non_zero & (max_val == r)] = ((g - b) / diff)[non_zero & (max_val == r)] % 6
    h[non_zero & (max_val == g)] = ((b - r) / diff + 2)[non_zero & (max_val == g)]
    h[non_zero & (max_val == b)] = ((r - g) / diff + 4)[non_zero & (max_val == b)]
    h = h / 6  

    s = torch.zeros_like(max_val)
    s[non_zero] = diff[non_zero] / max_val[non_zero]


    v = max_val
    hsv = torch.stack((h, s, v), dim=0)


    return hsv

def hsv_to_rgb(image):
    h, s, v = image[0], image[1], image[2]
    i = torch.floor(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i.int() % 6
    
    conditions = [
        (i == 0, torch.stack((v, t, p), dim=0)),
        (i == 1, torch.stack((q, v, p), dim=0)),
        (i == 2, torch.stack((p, v, t), dim=0)),
        (i == 3, torch.stack((p, q, v), dim=0)),
        (i == 4, torch.stack((t, p, v), dim=0)),
        (i == 5, torch.stack((v, p, q), dim=0)),
    ]
    
    r, g, b = torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h)
    for condition, result in conditions:
        r[condition], g[condition], b[condition] = result[0][condition], result[1][condition], result[2][condition]
    rgb = torch.stack((r, g, b), dim=0)

    rgb = adjust_range_to_m11(rgb)
    
    return rgb

class ImageEnhancer:
    def __init__(self, visible_tensor,data):
        self.visible = visible_tensor
        self.data = data

    def to_hsv_and_enhance(self):

        self.visible = self.enhance_visible(rgb_to_hsv(self.visible),self.data)

        self.visible = hsv_to_rgb(self.visible)

    def enhance_visible(self, tensor,data):
       
        saturation_increase = 0.1
        value_increase = 0.1
        tensor[1, :, :] = torch.clamp(tensor[1, :, :] + saturation_increase, 0, 1)
        tensor[2, :, :] = torch.clamp(tensor[2, :, :] + value_increase, 0, 1) # 亮度
        return tensor

    def enhance_thermal(self, tensor):
        value_increase = 0.1 
        tensor[2, :, :] = torch.clamp(tensor[2, :, :] + value_increase, 0, 1) 
        return tensor

