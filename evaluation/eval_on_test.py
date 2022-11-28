# camera-ready

import sys
import os

sys.path.append(os.getcwd())
print(os.getcwd())
from datasets import DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

# sys.path.append("/root/deeplabv3/model")
from model.deeplabv3 import DeepLabV3

# sys.path.append("/root/deeplabv3/utils")
from utils.utils import label_img_to_color, pv2rgb
from utils.metric import Evaluator

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

batch_size = 1
num_classes = 6
CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
network = DeepLabV3("eval_test", project_dir="./").cuda()
network.load_state_dict(torch.load("./training_logs/model_2/checkpoints/model_2_epoch_841.pth"))

val_dataset = DatasetVal(data_path="./data/val/images/",
                         mask_path="./data/val/masks/")

num_val_batches = int(len(val_dataset)/batch_size)
print ("num_val_batches:", num_val_batches)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=1)

# with open("/root/deeplabv3/data/cityscapes/meta/class_weights.pkl", "rb") as file: # (needed for python3)
#     class_weights = np.array(pickle.load(file))
# class_weights = torch.from_numpy(class_weights)
# class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

# # loss function
# loss_fn = nn.CrossEntropyLoss(weight=class_weights)
loss_fn = nn.CrossEntropyLoss()
metrics = Evaluator(num_class=num_classes)
network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
batch_losses = []
for step, (imgs, label_imgs, img_names) in enumerate(val_loader):
    with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
        label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))

        outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

        # compute the loss:
        loss = loss_fn(outputs, label_imgs)
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)

        ########################################################################
        # save data for visualization:
        ########################################################################
        outputs = outputs.data.cpu().numpy() # (shape: (batch_size, num_classes, img_h, img_w))
        pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, img_h, img_w))
        pred_label_imgs = pred_label_imgs.astype(np.uint8)
        for i in range(pred_label_imgs.shape[0]):
            metrics.add_batch(label_imgs[i], pred_label_imgs[i])
        for i in range(pred_label_imgs.shape[0]):
            if i == 0:
                pred_label_img = pred_label_imgs[i] # (shape: (img_h, img_w))
                img_name = img_names[i]
                img = imgs[i] # (shape: (3, img_h, img_w))

                img = img.data.cpu().numpy()
                img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
                img = img*np.array([0.229, 0.224, 0.225])
                img = img + np.array([0.485, 0.456, 0.406])
                img = img*255.0
                img = img.astype(np.uint8)

                # pred_label_img_color = label_img_to_color(pred_label_img)
                pred_label_img_color = pv2rgb(pred_label_img)
                overlayed_img = 0.35*img + 0.65*pred_label_img_color
                overlayed_img = overlayed_img.astype(np.uint8)

                cv2.imwrite(network.model_dir + "/" + img_name + "_overlayed.png", overlayed_img)

mIoU = np.nanmean(metrics.Intersection_over_Union()[:-1])
F1 = np.nanmean(metrics.F1()[:-1])
OA = np.nanmean(metrics.OA())
iou_per_class = metrics.Intersection_over_Union()
eval_value = {'mIoU': mIoU,
                'F1': F1,
                'OA': OA}
print('train:', eval_value)

iou_value = {}
for class_name, iou in zip(CLASSES, iou_per_class):
    iou_value[class_name] = iou
print(iou_value)
val_loss = np.mean(batch_losses)
print ("val loss: %g" % val_loss)
