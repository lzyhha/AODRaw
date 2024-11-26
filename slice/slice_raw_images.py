# Copyright (c) OpenMMLab. All rights reserved.
# Written by jbwang1997
# Reference: https://github.com/jbwang1997/BboxToolkit

import argparse
import datetime
import itertools
import json
import logging
import os
import os.path as osp
import time
from functools import partial
from math import ceil
from multiprocessing import Manager, Pool
from pycocotools.coco import COCO
import cv2
import numpy as np

from PIL import Image, ExifTags
import rawpy
import torch

try:
    import shapely.geometry as shgeo
except ImportError:
    shgeo = None


def add_parser(parser):
    """Add arguments."""
    parser.add_argument(
        '--base-json',
        type=str,
        default=None,
        help='json config file for split images')
    parser.add_argument(
        '--nproc', type=int, default=10, help='the procession number')

    # argument for loading data
    parser.add_argument(
        '--img-root',
        type=str,
        default=None,
        help='images root')
    parser.add_argument(
        '--ann-file',
        type=str,
        default=None,
        help='annotations')

    parser.add_argument(
        '--save-ann-file',
        type=str,
        default=None,
        help='annotations')

    # argument for splitting image
    parser.add_argument(
        '--sizes',
        nargs='+',
        type=int,
        default=[1024],
        help='the sizes of sliding windows')
    parser.add_argument(
        '--gaps',
        nargs='+',
        type=int,
        default=[512],
        help='the steps of sliding widnows')
    parser.add_argument(
        '--rates',
        nargs='+',
        type=float,
        default=[1.],
        help='same as DOTA devkit rate, but only change windows size')
    parser.add_argument(
        '--img-rate-thr',
        type=float,
        default=0.6,
        help='the minimal rate of image in window and window')
    parser.add_argument(
        '--iof-thr',
        type=float,
        default=0.7,
        help='the minimal iof between a object and a window')
    parser.add_argument(
        '--no-padding',
        action='store_true',
        help='not padding patches in regular size')
    parser.add_argument(
        '--padding-value',
        nargs='+',
        type=int,
        default=[0],
        help='padding value, 1 or channel number')

    # argument for saving
    parser.add_argument(
        '--save-dir',
        type=str,
        default='.',
        help='to save pkl and split images')
    parser.add_argument(
        '--save-ext',
        type=str,
        default='.png',
        help='the extension of saving images')


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='Splitting images')
    add_parser(parser)
    args = parser.parse_args()

    if args.base_json is not None:
        with open(args.base_json, 'r') as f:
            prior_config = json.load(f)

        for action in parser._actions:
            if action.dest not in prior_config or \
                    not hasattr(action, 'default'):
                continue
            action.default = prior_config[action.dest]
        args = parser.parse_args()

    # assert arguments
    assert args.img_root is not None, "argument img_root can't be None"
    assert args.ann_file is not None
    assert len(args.sizes) == len(args.gaps)
    assert len(args.sizes) == 1 or len(args.rates) == 1
    assert args.save_ext in ['.npy', '.JPG']
    assert args.iof_thr >= 0 and args.iof_thr < 1
    assert args.iof_thr >= 0 and args.iof_thr <= 1
    # assert not osp.exists(args.save_dir), \
    #     f'{osp.join(args.save_dir)} already exists'
    return args


def get_sliding_window(info, sizes, gaps, img_rate_thr):
    """Get sliding windows.

    Args:
        info (dict): Dict of image's width and height.
        sizes (list): List of window's sizes.
        gaps (list): List of window's gaps.
        img_rate_thr (float): Threshold of window area divided by image area.

    Returns:
        list[np.array]: Information of valid windows.
    """
    eps = 0.01
    windows = []
    width, height = info['width'], info['height']
    for size, gap in zip(sizes, gaps):
        assert size > gap, f'invaild size gap pair [{size} {gap}]'
        step = size - gap

        x_num = 1 if width <= size else ceil((width - size) / step + 1)
        x_start = [step * i for i in range(x_num)]
        if len(x_start) > 1 and x_start[-1] + size > width:
            x_start[-1] = width - size

        y_num = 1 if height <= size else ceil((height - size) / step + 1)
        y_start = [step * i for i in range(y_num)]
        if len(y_start) > 1 and y_start[-1] + size > height:
            y_start[-1] = height - size
        
        x_start = [int(x / 2) * 2 for x in x_start]
        y_start = [int(x / 2) * 2 for x in y_start]
        assert size % 2 == 0

        start = np.array(
            list(itertools.product(x_start, y_start)), dtype=np.int64)
        stop = start + size
        windows.append(np.concatenate([start, stop], axis=1))
    windows = np.concatenate(windows, axis=0)

    img_in_wins = windows.copy()
    img_in_wins[:, 0::2] = np.clip(img_in_wins[:, 0::2], 0, width)
    img_in_wins[:, 1::2] = np.clip(img_in_wins[:, 1::2], 0, height)
    img_areas = (img_in_wins[:, 2] - img_in_wins[:, 0]) * \
                (img_in_wins[:, 3] - img_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * \
                (windows[:, 3] - windows[:, 1])
    img_rates = img_areas / win_areas
    if not (img_rates > img_rate_thr).any():
        assert False
        max_rate = img_rates.max()
        img_rates[abs(img_rates - max_rate) < eps] = 1
    return windows[img_rates > img_rate_thr]


def bbox_overlaps_iof(bboxes1, bboxes2, eps=1e-6):
    """Compute bbox overlaps (iof).

    Args:
        bboxes1 (np.array): Horizontal bboxes1.
        bboxes2 (np.array): Horizontal bboxes2.
        eps (float, optional): Defaults to 1e-6.

    Returns:
        np.array: Overlaps.
    """
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]

    if rows * cols == 0:
        return np.zeros((rows, cols), dtype=np.float32)

    hbboxes1 = bboxes1
    hbboxes2 = bboxes2
    hbboxes1 = hbboxes1[:, None, :]
    lt = np.maximum(hbboxes1[..., :2], hbboxes2[..., :2])
    rb = np.minimum(hbboxes1[..., 2:], hbboxes2[..., 2:])
    wh = np.clip(rb - lt, 0, np.inf)
    h_overlaps = wh[..., 0] * wh[..., 1]
    
    l, t, r, b = [bboxes1[..., i] for i in range(4)]
    polys1 = np.stack([l, t, r, t, r, b, l, b], axis=-1)

    l, t, r, b = [bboxes2[..., i] for i in range(4)]
    polys2 = np.stack([l, t, r, t, r, b, l, b], axis=-1)
    if shgeo is None:
        raise ImportError('Please run "pip install shapely" '
                          'to install shapely first.')
    sg_polys1 = [shgeo.Polygon(p) for p in polys1.reshape(rows, -1, 2)]
    sg_polys2 = [shgeo.Polygon(p) for p in polys2.reshape(cols, -1, 2)]
    overlaps = np.zeros(h_overlaps.shape)
    for p in zip(*np.nonzero(h_overlaps)):
        overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area
    unions = np.array([p.area for p in sg_polys1], dtype=np.float32)
    unions = unions[..., None]

    unions = np.clip(unions, eps, np.inf)
    outputs = overlaps / unions
    if outputs.ndim == 1:
        outputs = outputs[..., None]
    return outputs


def get_window_obj(info, coco: COCO, windows, iof_thr):
    """

    Args:
        info (dict): Dict of bbox annotations.
        windows (np.array): information of sliding windows.
        iof_thr (float): Threshold of overlaps between bbox and window.

    Returns:
        list[dict]: List of bbox annotations of every window.
    """
    # bboxes = info['ann']['bboxes']
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[info['id']]))

    bboxes = [ann['bbox'] for ann in anns]
    bboxes = np.array([[bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]] for bbox in bboxes])
    iscrowds = np.array([ann['iscrowd'] for ann in anns])
    category_ids = np.array([ann['category_id'] for ann in anns])
    ann_info = dict(
        bboxes = bboxes, 
        iscrowds = iscrowds, 
        category_ids = category_ids, 
    )
    iofs = bbox_overlaps_iof(bboxes, windows)
    
    window_anns = []
    for i in range(windows.shape[0]):
        win_iofs = iofs[:, i]
        pos_inds = np.nonzero(win_iofs >= iof_thr)[0].tolist()
        ign_inds = np.nonzero((win_iofs < iof_thr) * (win_iofs > 0))[0].tolist()

        win_ann = {}

        win_ann['bboxes'] = ann_info['bboxes'][pos_inds + ign_inds]
        win_ann['category_ids'] = ann_info['category_ids'][pos_inds + ign_inds]
        win_ann['iscrowds'] = np.array((ann_info['iscrowds'][pos_inds]).tolist() + [1 for _ in range(len(ann_info['iscrowds'][ign_inds]))])

        window_anns.append(win_ann)
    return window_anns


def load_raw(root, name):
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

    if (image_srgb.shape[0] - image_srgb.shape[1]) * (image_raw.shape[0] - image_raw.shape[1]) < 0:
        print(image_srgb.shape, image_raw.shape, name, 'need orientation')
        if orientation in exif:
            assert exif[orientation] == 8
        else:
            print('no exif info')
            exif[orientation] = 8

    image_raw = torch.from_numpy(image_raw).unsqueeze(0).unsqueeze(0)

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

    # RGBG format
    image_bayer = torch.concat((
        image_raw[:, :, 0:image_raw.shape[2]:2, 0:image_raw.shape[3]:2],
        image_raw[:, :, 0:image_raw.shape[2]:2, 1:image_raw.shape[3]:2],
        image_raw[:, :, 1:image_raw.shape[2]:2, 1:image_raw.shape[3]:2],
        image_raw[:, :, 1:image_raw.shape[2]:2, 0:image_raw.shape[3]:2]), axis=1)
    
    '''
    A part of sRGB images have a lower resoluation than the corresponding RAW images 
    because the borders of the RAW images are lost when coverting to sRGB images, 
    we crop the borders of these RAW images in preprocessing.
    '''
    height, width = image_srgb.shape[0], image_srgb.shape[1]
    assert height % 2 == 0
    assert width % 2 == 0
    h_border = image_bayer.shape[2] - int(height // 2)
    w_border = image_bayer.shape[3] - int(width // 2)
    assert h_border % 2 == 0
    assert w_border % 2 == 0

    h_border = int(h_border / 2)
    w_border = int(w_border / 2)

    if h_border > 0:
        image_bayer = image_bayer[:, :, h_border:-h_border, :]
    if w_border > 0:
        image_bayer = image_bayer[:, :, :, w_border:-w_border]
    
    assert 0 < image_bayer.max() <= (2 ** 14 - 1)

    image_bayer = image_bayer.permute(0, 2, 3, 1)
    image_bayer = image_bayer.squeeze(0).cpu().numpy()

    return image_bayer


def crop_and_save_img(info, windows, window_anns, no_padding, img_root,
                      padding_value, save_dir, img_ext):
    """

    Args:
        info (dict): Image's information.
        windows (np.array): information of sliding windows.
        window_anns (list[dict]): List of bbox annotations of every window.
        img_dir (str): Path of images.
        no_padding (bool): If True, no padding.
        padding_value (tuple[int|float]): Padding value.
        save_dir (str): Save filename.
        anno_dir (str): Annotation filename.
        img_ext (str): Picture suffix.

    Returns:
        list[dict]: Information of paths.
    """
    if img_ext == '.npy':
        img = load_raw(img_root, info['file_name'].split('.')[0])
    else:
        img = cv2.imread(osp.join(img_root, info['file_name']))
    patch_infos = []
    for i in range(windows.shape[0]):
        if window_anns[i]['bboxes'].shape[0] == 0:
            continue
        assert window_anns[i]['bboxes'].shape[0] == window_anns[i]['iscrowds'].shape[0]
        assert window_anns[i]['category_ids'].shape[0] == window_anns[i]['iscrowds'].shape[0]
        assert window_anns[i]['bboxes'].shape[0] > 0

        if np.sum(window_anns[i]['iscrowds']) == window_anns[i]['bboxes'].shape[0]:
            continue
        assert np.sum(window_anns[i]['iscrowds']) < window_anns[i]['bboxes'].shape[0]

        patch_info = dict()
        for k, v in info.items():
            if k not in ['id', 'file_name', 'width', 'height', 'ann']:
                patch_info[k] = v

        window = windows[i]
        x_start, y_start, x_stop, y_stop = window.tolist()
        patch_info['x_start'] = x_start
        patch_info['y_start'] = y_start
        patch_info['id'] = \
            str(info['id']) + '__' + str(x_stop - x_start) + \
            '__' + str(x_start) + '___' + str(y_start)
        patch_info['ori_id'] = info['id']

        patch_info['filename'] = \
            info['file_name'].split('.')[0] + '__' + str(x_stop - x_start) + \
            '__' + str(x_start) + '___' + str(y_start) + img_ext

        ann = window_anns[i]
        ann['bboxes'] = translate(ann['bboxes'], -x_start, -y_start)
        patch_info['ann'] = ann

        assert y_start % 2 == 0 and y_stop % 2 == 0 and x_start % 2 == 0 and x_stop % 2 == 0
        patch = img[y_start // 2:y_stop // 2, x_start // 2:x_stop // 2]
        if not no_padding:
            height = (y_stop - y_start) // 2
            width = (x_stop - x_start) // 2
            assert (y_stop - y_start) % 2 == 0 and (x_stop - x_start) % 2 == 0
            if height > patch.shape[0] or width > patch.shape[1]:
                padding_patch = np.empty((height, width, patch.shape[-1]),
                                         dtype=patch.dtype)
                if not isinstance(padding_value, (int, float)):
                    assert len(padding_value) == patch.shape[-1]
                padding_patch[...] = padding_value
                padding_patch[:patch.shape[0], :patch.shape[1], ...] = patch
                patch = padding_patch
        patch_info['height'] = patch.shape[0] * 2
        patch_info['width'] = patch.shape[1] * 2
        if img_ext == '.npy':
            patch = patch.transpose(2, 0, 1)
            np.save(osp.join(save_dir, patch_info['filename']), patch)
        else:
            cv2.imwrite(osp.join(save_dir, patch_info['filename']), patch)
        patch_infos.append(patch_info)

    return patch_infos


def single_split(arguments, coco, sizes, gaps, img_rate_thr, iof_thr, no_padding,
                 padding_value, save_dir, img_root, img_ext, lock, prog, total,
                 logger):
    """

    Args:
        arguments (object): Parameters.
        sizes (list): List of window's sizes.
        gaps (list): List of window's gaps.
        img_rate_thr (float): Threshold of window area divided by image area.
        iof_thr (float): Threshold of overlaps between bbox and window.
        no_padding (bool): If True, no padding.
        padding_value (tuple[int|float]): Padding value.
        save_dir (str): Save filename.
        anno_dir (str): Annotation filename.
        img_ext (str): Picture suffix.
        lock (object): Lock of Manager.
        prog (object): Progress of Manager.
        total (object): Length of infos.
        logger (object): Logger.

    Returns:
        list[dict]: Information of paths.
    """
    img_info = arguments
    windows = get_sliding_window(img_info, sizes, gaps, img_rate_thr)
    window_anns = get_window_obj(img_info, coco, windows, iof_thr)
    patch_infos = crop_and_save_img(img_info, windows, window_anns,
                                    no_padding, img_root, padding_value, save_dir,
                                    img_ext)
    # assert patch_infos

    lock.acquire()
    prog.value += 1
    msg = f'({prog.value / total:3.1%} {prog.value}:{total})'
    msg += ' - ' + f"Filename: {img_info['file_name']}"
    msg += ' - ' + f"width: {img_info['width']:<5d}"
    msg += ' - ' + f"height: {img_info['height']:<5d}"
    msg += ' - ' + f'Patches: {len(patch_infos)}'
    logger.info(msg)
    lock.release()

    return patch_infos


def setup_logger(log_path):
    """Setup logger.

    Args:
        log_path (str): Path of log.

    Returns:
        object: Logger.
    """
    logger = logging.getLogger('img split')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    handlers = [logging.StreamHandler()]

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def translate(bboxes, x, y):
    """Map bboxes from window coordinate back to original coordinate.

    Args:
        bboxes (np.array): bboxes with window coordinate.
        x (float): Deviation value of x-axis.
        y (float): Deviation value of y-axis

    Returns:
        np.array: bboxes with original coordinate.
    """
    dim = bboxes.shape[-1]
    translated = bboxes + np.array([x, y] * int(dim / 2), dtype=np.float32)
    return translated


def cvt_to_coco_json(annotations, ori_ann_file):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    with open(ori_ann_file, 'r') as f:
        ori_ann = json.load(f)

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag, image_item):
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x1,y1,x2,y2
        # left_top
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))
        # left_bottom
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))
        # right_bottom
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))
        # right_top
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))

        annotation_item['segmentation'].append(seg)

        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(image_item['width'] - 1, bbox[2])
        bbox[3] = min(image_item['height'] - 1, bbox[3])

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    coco['categories'] = ori_ann['categories']

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['ori_id'] = ann_dict['ori_id']
        image_item['x_start'] = ann_dict['x_start']
        image_item['y_start'] = ann_dict['y_start']
        image_item['slice_id'] = ann_dict['slice_id']
        image_item['file_name'] = file_name
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        image_item['tag'] = ann_dict['tag']
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        iscrowds = ann['iscrowds']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=iscrowds[bbox_id], image_item=image_item)

        image_id += 1

    return coco


def main():
    """Main function of image split."""
    args = parse_args()

    padding_value = args.padding_value[0] \
        if len(args.padding_value) == 1 else args.padding_value
    sizes, gaps = [], []
    for rate in args.rates:
        sizes += [int(size / rate) for size in args.sizes]
        gaps += [int(gap / rate) for gap in args.gaps]
    save_imgs = args.save_dir
    if not os.path.exists(save_imgs):
        os.makedirs(save_imgs)
    logger = setup_logger(args.save_dir)

    print('Loading original data!!!')
    coco = COCO(args.ann_file)

    print('Start splitting images!!!')
    start = time.time()
    manager = Manager()
    worker = partial(
        single_split,
        coco=coco, 
        sizes=sizes,
        gaps=gaps,
        img_rate_thr=args.img_rate_thr,
        iof_thr=args.iof_thr,
        no_padding=args.no_padding,
        padding_value=padding_value,
        save_dir=save_imgs,
        img_root=args.img_root, 
        img_ext=args.save_ext,
        lock=manager.Lock(),
        prog=manager.Value('i', 0),
        total=len(coco.imgs),
        logger=logger)

    if args.nproc > 1:
        pool = Pool(args.nproc)
        patch_infos = pool.map(worker, list(coco.imgs.values()))
        pool.close()
    else:
        patch_infos = list(map(worker, list(coco.imgs.values())))
    
    if args.save_ann_file != "":
        annotations = []
        n = 0
        for patch_info in patch_infos:
            for patch in patch_info:
                n += 1
                annotation = {}
                annotation['slice_id'] = patch['id']
                for k in ['tag', 'x_start', 'y_start', 'ori_id', 'filename', 'height', 'width']:
                    annotation[k] = patch[k]
                annotation['ann'] = dict()
                annotation['ann']['bboxes'] = patch['ann']['bboxes']
                annotation['ann']['labels'] = patch['ann']['category_ids']
                annotation['ann']['iscrowds'] = patch['ann']['iscrowds']

                annotations.append(annotation)
        
        coco_annotations = cvt_to_coco_json(annotations, args.ann_file)

        with open(args.save_ann_file, 'w') as f:
            json.dump(coco_annotations, f)


if __name__ == '__main__':
    main()