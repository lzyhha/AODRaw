import os
import cv2
from tqdm import tqdm
import argparse


def downsampling(args):
    root = args.root
    target = args.target

    if not os.path.exists(target):
        os.makedirs(target)

    images = sorted(os.listdir(root))
    images = sorted([n.split('.')[0] for n in images])

    for name in tqdm(images):
        image_srgb = cv2.imread(os.path.join(root, name + '.JPG'))
        height, width = image_srgb.shape[0], image_srgb.shape[1]
        
        image_srgb = cv2.resize(image_srgb, (int(width / 3), int(height / 3)), interpolation = cv2.INTER_LINEAR)

        cv2.imwrite(os.path.join(target, name + '.JPG'), image_srgb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--target", required=True, type=str)
    args = parser.parse_args()

    downsampling(args)
