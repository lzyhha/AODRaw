import os
from PIL import Image, ExifTags
import rawpy
import numpy as np
import cv2
import torch
from debayer import Debayer3x3
import torch.nn.functional as F
from tqdm import tqdm
import argparse


def downsampling(args):
    root = args.root
    target = args.target

    if args.cuda:
        debayer = Debayer3x3().cuda()
    else:
        debayer = Debayer3x3()

    # image name list
    images = sorted(os.listdir(root))
    images = sorted([name.split('.')[0] for name in images])

    if not os.path.exists(target):
        os.makedirs(target)

    for name in tqdm(images):
        # load images
        image_srgb = cv2.imread(os.path.join(root, name + '.JPG'))
        image_srgb_pil = Image.open(os.path.join(root, name + '.JPG'))
        image_raw = rawpy.imread(os.path.join(root, name + '.ARW'))
        image_raw = image_raw.raw_image_visible.astype(np.float32)

        # flags for orientation
        # A part of images need to be rotated and we complete this part in preprocessing. 
        exif = image_srgb_pil._getexif()
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        
        # a part of RAW images need to be rotated
        if (image_srgb.shape[0] - image_srgb.shape[1]) * (image_raw.shape[0] - image_raw.shape[1]) < 0:
            if orientation in exif:
                assert exif[orientation] == 8
            else:
                exif[orientation] = 8

        image_raw = torch.from_numpy(image_raw).unsqueeze(0).unsqueeze(0)
        if args.cuda:
            image_raw = image_raw.cuda()

        # orientation
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        if orientation in exif:
            if exif[orientation] == 1:
                pass
            elif exif[orientation] == 8:
                image_raw = torch.flip(image_raw, dims=[3])
                image_raw = image_raw.permute(0, 1, 3, 2)
            else:
                raise NotImplementedError()
        else:
            assert (image_srgb.shape[0] - image_srgb.shape[1]) * (image_raw.shape[2] - image_raw.shape[3]) > 0

        image_raw = debayer(image_raw)

        '''
        A part of sRGB images have a lower resoluation than the corresponding RAW images 
        because the borders of the RAW images are lost when coverting to sRGB images, 
        we crop the borders of these RAW images in preprocessing.
        '''
        height, width = image_srgb.shape[0], image_srgb.shape[1]
        h_border = image_raw.shape[2] - height
        w_border = image_raw.shape[3] - width
        assert h_border % 2 == 0
        assert w_border % 2 == 0
        h_border = int(h_border / 2)
        w_border = int(w_border / 2)

        if h_border > 0:
            image_raw = image_raw[:, :, h_border:-h_border, :]
        if w_border > 0:
            image_raw = image_raw[:, :, :, w_border:-w_border]
        
        image_raw = F.interpolate(image_raw, size=(int(height / 3), int(width / 3)), mode='bilinear')
        assert image_raw.max() <= (2 ** 14 - 1)
        image_raw = image_raw / ((2 ** 14 - 1) / 255.)    

        image_raw = image_raw.squeeze(0).cpu().numpy() # shape of 3 x H x W
        np.save(os.path.join(target, name + '.npy'), image_raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--target", required=True, type=str)
    parser.add_argument("--cuda", action='store_true')
    args = parser.parse_args()

    downsampling(args)
