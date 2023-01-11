import glob
import os
import numpy as np
import cv2
import albumentations as albu
from PIL import Image
import shutil
from torchvision.transforms import (Pad, ColorJitter, Resize, FiveCrop, RandomCrop,
                                    RandomHorizontalFlip, RandomRotation, RandomVerticalFlip)


train_data = ["top_mosaic_09cm_area1",
             "top_mosaic_09cm_area3",
             "top_mosaic_09cm_area5",
             "top_mosaic_09cm_area7",
             "top_mosaic_09cm_area11",
             "top_mosaic_09cm_area13",
             "top_mosaic_09cm_area15",
             "top_mosaic_09cm_area17",
             "top_mosaic_09cm_area21",
             "top_mosaic_09cm_area23",
             "top_mosaic_09cm_area26",
             "top_mosaic_09cm_area28",
             "top_mosaic_09cm_area30",
             "top_mosaic_09cm_area32",
             "top_mosaic_09cm_area34",
             "top_mosaic_09cm_area37"]
train_imgs_output_dir = "./data/train/images"
train_masks_output_dir = "./data/train/masks"
val_imgs_output_dir = "./data/val/images"
val_masks_output_dir = "./data/val/masks"
test_imgs_output_dir = "./data/test/images"
test_masks_output_dir = "./data/test/masks"
if not os.path.exists(train_imgs_output_dir):
    os.makedirs(train_imgs_output_dir)
if not os.path.exists(train_masks_output_dir):
    os.makedirs(train_masks_output_dir)
if not os.path.exists(val_imgs_output_dir):
    os.makedirs(val_imgs_output_dir)
if not os.path.exists(val_masks_output_dir):
    os.makedirs(val_masks_output_dir)
if not os.path.exists(test_imgs_output_dir):
    os.makedirs(test_imgs_output_dir)
if not os.path.exists(test_masks_output_dir):
    os.makedirs(test_masks_output_dir)


imgs_dir="./data/images"
masks_dir="./data/masks"
boundarys_dir="./data/boundarys"
img_paths = glob.glob(os.path.join(imgs_dir, "*.tif"))
mask_paths = glob.glob(os.path.join(masks_dir, "*.tif"))
boundary_paths = glob.glob(os.path.join(boundarys_dir, "*.tif"))
print(img_paths)
print(mask_paths)
print(boundary_paths)

ImSurf = np.array([255, 255, 255])  # label 0
Building = np.array([255, 0, 0]) # label 1
LowVeg = np.array([255, 255, 0]) # label 2
Tree = np.array([0, 255, 0]) # label 3
Car = np.array([0, 255, 255]) # label 4
Clutter = np.array([0, 0, 255]) # label 5
Boundary = np.array([0, 0, 0]) # label 6
num_classes = 6

def get_img_mask_padded(image, mask, patch_size, mode):
    img, mask = np.array(image), np.array(mask)
    oh, ow = img.shape[0], img.shape[1]
    rh, rw = oh % patch_size, ow % patch_size
    width_pad = 0 if rw == 0 else patch_size - rw
    height_pad = 0 if rh == 0 else patch_size - rh

    h, w = oh + height_pad, ow + width_pad
    pad_img = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right',
                               border_mode=0, value=[0, 0, 0])(image=img)
    if mode == 'train':
        pad_img = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right')(image=img)

    pad_mask = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right',
                                border_mode=0, value=[0, 0, 0])(image=mask)
    img_pad, mask_pad = pad_img['image'], pad_mask['image']
    img_pad = cv2.cvtColor(np.array(img_pad), cv2.COLOR_RGB2BGR)
    mask_pad = cv2.cvtColor(np.array(mask_pad), cv2.COLOR_RGB2BGR)
    return img_pad, mask_pad


def pv2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def car_color_replace(mask):
    mask = cv2.cvtColor(np.array(mask.copy()), cv2.COLOR_RGB2BGR)
    mask[np.all(mask == [0, 255, 255], axis=-1)] = [0, 204, 255]

    return mask


def rgb_to_2D_label(_label):
    print(_label.size)
    _label = _label.transpose(2, 0, 1)
    label_seg = np.zeros(_label.shape[1:], dtype=np.uint8)
    label_seg[np.all(_label.transpose([1, 2, 0]) == ImSurf, axis=-1)] = 0
    label_seg[np.all(_label.transpose([1, 2, 0]) == Building, axis=-1)] = 1
    label_seg[np.all(_label.transpose([1, 2, 0]) == LowVeg, axis=-1)] = 2
    label_seg[np.all(_label.transpose([1, 2, 0]) == Tree, axis=-1)] = 3
    label_seg[np.all(_label.transpose([1, 2, 0]) == Car, axis=-1)] = 4
    label_seg[np.all(_label.transpose([1, 2, 0]) == Clutter, axis=-1)] = 5
    # label_seg[np.all(_label.transpose([1, 2, 0]) == Boundary, axis=-1)] = 6
    return label_seg


def image_augment(image, mask, patch_size, mode='train', val_scale=1.0):
    image_list = []
    mask_list = []
    image_width, image_height = image.size[1], image.size[0]
    mask_width, mask_height = mask.size[1], mask.size[0]
    assert image_height == mask_height and image_width == mask_width
    if mode == 'train':
        # resize_0 = Resize(size=(int(image_width * 0.25), int(image_height * 0.25)))
        # resize_1 = Resize(size=(int(image_width * 0.5), int(image_height * 0.5)))
        # resize_2 = Resize(size=(int(image_width * 0.75), int(image_height * 0.75)))
        # resize_3 = Resize(size=(int(image_width * 1.25), int(image_height * 1.25)))
        # resize_4 = Resize(size=(int(image_width * 1.5), int(image_height * 1.5)))
        # resize_5 = Resize(size=(int(image_width * 1.75), int(image_height * 1.75)))
        # resize_6 = Resize(size=(int(image_width * 2.0), int(image_height * 2.0)))
        # image_resize_0, mask_resize_0 = resize_0(image.copy()), resize_0(mask.copy())
        # image_resize_1, mask_resize_1 = resize_1(image.copy()), resize_1(mask.copy())
        # image_resize_2, mask_resize_2 = resize_2(image.copy()), resize_2(mask.copy())
        # image_resize_3, mask_resize_3 = resize_3(image.copy()), resize_3(mask.copy())
        # image_resize_4, mask_resize_4 = resize_4(image.copy()), resize_4(mask.copy())
        # image_resize_5, mask_resize_5 = resize_5(image.copy()), resize_5(mask.copy())
        # image_resize_6, mask_resize_6 = resize_6(image.copy()), resize_6(mask.copy())
        h_vlip = RandomHorizontalFlip(p=1.0)
        v_vlip = RandomVerticalFlip(p=1.0)
        # crop_1 = RandomCrop(size=(int(image_width*0.75), int(image_height*0.75)))
        # crop_2 = RandomCrop(size=(int(image_width * 0.5), int(image_height * 0.5)))
        # color = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        image_h_vlip, mask_h_vlip = h_vlip(image.copy()), h_vlip(mask.copy())
        image_v_vlip, mask_v_vlip = v_vlip(image.copy()), v_vlip(mask.copy())
        # image_crop_1, mask_crop_1 = crop_1(image.copy()), crop_1(mask.copy())
        # image_crop_2, mask_crop_2 = crop_2(image.copy()), crop_2(mask.copy())
        # image_color = color(image.copy())

        image_list_train = [image, image_h_vlip, image_v_vlip]
        mask_list_train = [mask, mask_h_vlip, mask_v_vlip]
        # image_list_train = [image]
        # mask_list_train = [mask]
        for i in range(len(image_list_train)):
            image_tmp, mask_tmp = get_img_mask_padded(image_list_train[i], mask_list_train[i], patch_size, mode)
            mask_tmp = rgb_to_2D_label(mask_tmp.copy())
            image_list.append(image_tmp)
            mask_list.append(mask_tmp)
    else:
        rescale = Resize(size=(int(image_width * val_scale), int(image_height * val_scale)))
        image, mask = rescale(image.copy()), rescale(mask.copy())
        image, mask = get_img_mask_padded(image.copy(), mask.copy(), patch_size, mode)
        mask = rgb_to_2D_label(mask.copy())
        image_list.append(image)
        mask_list.append(mask)
    return image_list, mask_list


def randomsizedcrop(image, mask):
    # assert image.shape[:2] == mask.shape
    h, w = image.shape[0], image.shape[1]
    crop = albu.RandomSizedCrop(min_max_height=(int(3*h//8), int(h//2)), width=h, height=w)(image=image.copy(), mask=mask.copy())
    img_crop, mask_crop = crop['image'], crop['mask']
    return img_crop, mask_crop


def car_aug(image, mask):
    assert image.shape[:2] == mask.shape
    v_flip = albu.VerticalFlip(p=1.0)(image=image.copy(), mask=mask.copy())
    h_flip = albu.HorizontalFlip(p=1.0)(image=image.copy(), mask=mask.copy())
    rotate_90 = albu.RandomRotate90(p=1.0)(image=image.copy(), mask=mask.copy())
    # blur = albu.GaussianBlur(p=1.0)(image=image.copy())
    image_vflip, mask_vflip = v_flip['image'], v_flip['mask']
    image_hflip, mask_hflip = h_flip['image'], h_flip['mask']
    image_rotate, mask_rotate = rotate_90['image'], rotate_90['mask']
    # blur_image = blur['image']
    image_list = [image, image_vflip, image_hflip, image_rotate]
    mask_list = [mask, mask_vflip, mask_hflip, mask_rotate]

    return image_list, mask_list

def vaihingen_format(inp):
    (img_path, mask_path, imgs_output_dir, masks_output_dir, eroded, gt, mode, val_scale, split_size, stride) = inp
    img_filename = os.path.splitext(os.path.basename(img_path))[0]
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    if eroded:
        mask_path = mask_path[:-4] + '_noBoundary.tif'
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')
    
    image_pad, mask_pad = get_img_mask_padded(img.copy(), mask.copy(), 16, "train")
    mask_tmp = rgb_to_2D_label(mask_pad.copy())
    out_mask_path = os.path.join(test_masks_output_dir,
                                 "{}.png".format(mask_filename))
    cv2.imwrite(out_mask_path, mask_tmp)
    out_img_path = os.path.join(test_imgs_output_dir,
                                "{}.tif".format(img_filename))

    cv2.imwrite(out_img_path, image_pad)
    if gt:
        mask_ = car_color_replace(mask)
        out_origin_mask_path = os.path.join(masks_output_dir + '/origin/', "{}.tif".format(mask_filename))
        cv2.imwrite(out_origin_mask_path, mask_)
    # print(img_path)
    # print(img.size, mask.size)
    # img and mask shape: WxHxC
    image_list, mask_list = image_augment(image=img.copy(), mask=mask.copy(), patch_size=split_size,
                                          mode=mode, val_scale=val_scale)
    assert img_filename == mask_filename and len(image_list) == len(mask_list)
    for m in range(len(image_list)):
        k = 0
        img = image_list[m]
        mask = mask_list[m]
        assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]
        if gt:
            mask = pv2rgb(mask)

        for y in range(0, img.shape[0], stride):
            for x in range(0, img.shape[1], stride):
                img_tile = img[y:y + split_size, x:x + split_size]
                mask_tile = mask[y:y + split_size, x:x + split_size]

                if img_tile.shape[0] == split_size and img_tile.shape[1] == split_size \
                        and mask_tile.shape[0] == split_size and mask_tile.shape[1] == split_size:
                    image_crop, mask_crop = randomsizedcrop(img_tile, mask_tile)
                    bins = np.array(range(num_classes + 1))
                    class_pixel_counts, _ = np.histogram(mask_crop, bins=bins)
                    cf = class_pixel_counts / (mask_crop.shape[0] * mask_crop.shape[1])
                    if cf[4] > 0.1 and mode == 'train':
                        car_imgs, car_masks = car_aug(image_crop, mask_crop)
                        for i in range(len(car_imgs)):
                            out_img_path = os.path.join(imgs_output_dir,
                                                        "{}_{}_{}_{}.tif".format(img_filename, m, k, i))
                            
                            cv2.imwrite(out_img_path, car_imgs[i])

                            out_mask_path = os.path.join(masks_output_dir,
                                                         "{}_{}_{}_{}.png".format(mask_filename, m, k, i))
                            cv2.imwrite(out_mask_path, car_masks[i])
                    else:
                        out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.tif".format(img_filename, m, k))
                        
                        cv2.imwrite(out_img_path, img_tile)

                        out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.png".format(mask_filename, m, k))
                        cv2.imwrite(out_mask_path, mask_tile)

                k += 1



for img_path, mask_path in zip(img_paths, mask_paths):
    print(img_path, mask_path)
    img_filename = os.path.splitext(os.path.basename(img_path))[0]
    print(img_filename)
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    # mask_path = mask_path[:-4] + '_noBoundary.tif'
    # img = Image.open(img_path).convert('RGB').resize((1900,2500))
    # mask = Image.open(mask_path).convert('RGB').resize((1900,2500))
    # print(img.size)
    # img.show()
    # mask = rgb_to_2D_label(mask.copy())
    if img_filename in train_data:
        inp = (img_path, mask_path, train_imgs_output_dir, train_masks_output_dir, False, False, "train", 1.0, 1024, 512)
        vaihingen_format(inp)
    else:
        inp = (img_path, mask_path, val_imgs_output_dir, val_masks_output_dir, False, False, "val", 1.0, 1024, 1024)
        vaihingen_format(inp)
        
    # else:
    #     inp = (img_path, mask_path, test_imgs_output_dir, test_masks_output_dir, False, False, "test", 1.0, 1024, 1024)
    #     vaihingen_format(inp)