from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import pandas as pd


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    # def get_imagedata_info(self, data):
    #     pids, cams, tracks = [], [], []

    #     for _, pid, camid, trackid in data:
    #         pids += [pid]
    #         cams += [camid]
    #         tracks += [trackid]
    #     pids = set(pids)
    #     cams = set(cams)
    #     tracks = set(tracks)
    #     num_pids = len(pids)
    #     num_cams = len(cams)
    #     num_imgs = len(data)
    #     num_views = len(tracks)
    #     return num_pids, num_imgs, num_cams, num_views

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        total_images = 0
        total_captions = 0
        with_caption = False

        for item in data:
            if len(item) == 5:
                img_paths, captions, pid, camid, trackid = item
                with_caption = True
            else:
                img_paths, pid, camid, trackid = item
                captions = []
            pids.append(pid)
            cams.append(camid)
            tracks.append(trackid)
            total_images += len(img_paths)
            total_captions += len(captions)

        num_pids = len(set(pids))
        num_cams = len(set(cams))
        num_items = len(data)
        num_views = len(set(tracks))

        if with_caption:
            return num_pids, num_items, num_cams, num_views, total_images, total_captions
        else:
            return num_pids, num_items, num_cams, num_views, total_images, None

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    # def print_dataset_statistics(self, train, query, gallery):
    #     num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
    #     num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
    #     num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

    #     print("Dataset statistics:")
    #     print("  ----------------------------------------")
    #     print("  subset   | # ids | # images | # cameras")
    #     print("  ----------------------------------------")
    #     print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
    #     print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
    #     print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
    #     print("  ----------------------------------------")

    def print_dataset_statistics(self, train, query, gallery, c_modality=None):
        train_info = self.get_imagedata_info(train)
        query_info = self.get_imagedata_info(query)
        gallery_info = self.get_imagedata_info(gallery)

        def format_info(split_name, info):
            num_pids, num_items, num_cams, num_views, total_imgs, total_caps = info
            row = {
                'Subset': split_name,
                '# IDs': num_pids,
                '# Data Items': num_items,
                '# Cameras': num_cams,
                '# Views': num_views,
                '# Total Images': total_imgs,
            }
            if c_modality:
                row['# Captions'] = total_caps
            return row

        stats = [
            format_info('Train', train_info),
            format_info('Query', query_info),
            format_info('Gallery', gallery_info)
        ]

        df = pd.DataFrame(stats)
        print("=> Dataset statistics:")
        print(df.to_markdown(index=False))


# Three modalities (original)
# class ImageDataset(Dataset):
#     def __init__(self, dataset, transform=None):
#         self.dataset = dataset
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         img_path, pid, camid, trackid = self.dataset[index]
#         if isinstance(img_path,list):
#             # print(img_path)
#             img_1 = read_image(img_path[0])
#             if self.transform is not None:
#                 img_1 = self.transform(img_1)

#             a,b,c=img_1.shape
#             tmp=torch.zeros(a,b,c)

#             if not os.path.exists(img_path[1]):
#                 img_2 = tmp
#             else:
#                 img_2 = read_image(img_path[1])
#                 if self.transform is not None:
#                     img_2 = self.transform(img_2)

#             if not os.path.exists(img_path[2]):
#                 img_3 = tmp
#             else:
#                 img_3 = read_image(img_path[2])
#                 if self.transform is not None:
#                     img_3 = self.transform(img_3)

#             return img_1, img_2, img_3, pid, camid, trackid, img_path[0].split('/')[-1]
#         else:
#             # print(img_path)
#             img = read_image(img_path)
#             if img.size == (768, 128):
#                 # print(1)
#                 #img1 = img
#                 img_1 = img.crop((0, 0, 256, 128))
#                 img_2 = img.crop((256, 0, 512, 128))
#                 img_3 = img.crop((256, 0, 768, 128))
#             if img.size == (512, 128):
#                 # print(2)
#                 #img1 = img
#                 img_1 = img.crop((0, 0, 256, 128))
#                 img_2 = img.crop((256, 0, 512, 128))
#                 img_3 = img.crop((256, 0, 512, 128))
#             if self.transform is not None:
#                 img_1 = self.transform(img_1)
#                 img_2 = self.transform(img_2)
#                 img_3 = self.transform(img_3)

#             return img_1, img_2, img_3, pid, camid, trackid, img_path[0].split('/')[-1]


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        if len(item) == 5:
            img_paths, captions, pid, camid, trackid = item
        else:
            img_paths, pid, camid, trackid = item
            captions = None

        imgs = []
        valid_filenames = []

        for path in img_paths:
            if os.path.exists(path):
                img = read_image(path)
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
                valid_filenames.append(os.path.basename(path))
            else:
                # Use a blank tensor if the image is missing (same shape as transformed images)
                if len(imgs) > 0:
                    a, b, c = imgs[0].shape
                else:
                    # Use dummy shape if no valid image yet
                    a, b, c = 3, 256, 128  # default image shape
                imgs.append(torch.zeros(a, b, c))
                valid_filenames.append(os.path.basename(path))

        # Pad with None if fewer than 3 modalities
        while len(imgs) < 3:
            imgs.append(torch.zeros_like(imgs[0]))
            valid_filenames.append("none.jpg")

        # Always return 3 images to preserve collate function expectations
        if captions:
            return imgs[0], imgs[1], imgs[2], captions, pid, camid, trackid, valid_filenames[0]
        else:
            return imgs[0], imgs[1], imgs[2], pid, camid, trackid, valid_filenames[0]


# Two modalities
class ImageDataset_dual(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        if isinstance(img_path, list):
            # Load first modality (e.g., visible)
            img_1 = read_image(img_path[0])
            if self.transform is not None:
                img_1 = self.transform(img_1)

            # Get shape for placeholder tensor
            a, b, c = img_1.shape
            tmp = torch.zeros(a, b, c)

            # Load second modality (e.g., IR or TI)
            if not os.path.exists(img_path[1]):
                img_2 = tmp
            else:
                img_2 = read_image(img_path[1])
                if self.transform is not None:
                    img_2 = self.transform(img_2)

            return img_1, img_2, pid, camid, trackid, img_path[0].split('/')[-1]
        else:
            # Handle single image case (not used for multi-modality)
            img = read_image(img_path)
            img_1 = img.crop((0, 0, 256, 128))  # First segment
            img_2 = img.crop((256, 0, 512, 128))  # Second segment
            if self.transform is not None:
                img_1 = self.transform(img_1)
                img_2 = self.transform(img_2)
            return img_1, img_2, pid, camid, trackid, img_path.split('/')[-1]