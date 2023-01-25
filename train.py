# camera-ready

import sys
import time

from datasets import DatasetTrain, DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

from model.deeplabv3 import DeepLabV3
from model.deeplabv3MutilDecoder import DeepLabV3MutilDecoder
from model.deeplabv3Boundary import DeepLabV3Boundary
from model.deeplabv3AddBoundary import DeepLabV3AddBoundary
from model.unet_model import UNet

from utils.utils import add_weight_decay, num_classes

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

import time
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--mode",
        type=str,
        help="model change",
        default="DeepLabV3AddBoundary")
    
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size",
        default="16")
    
    opt = parser.parse_args()
    mode = opt.mode
    batch_size = opt.batch_size
    # NOTE! NOTE! change this to not overwrite all log data when you train the model:
    model_id = "_2"

    num_epochs = 1000
    
    learning_rate = 0.00001

    device = None
    network = None
    if mode == "Unet":
        device = "cuda:0"
        network = UNet(mode+model_id, project_dir="./", n_channels=3, n_classes=num_classes).to(device)
    elif mode == "DeepLabV3":
        device = "cuda:0"
        network = DeepLabV3(mode+model_id, project_dir="./").to(device)
    elif mode == "DeepLabV3MutilDecoder":
        device = "cuda:0"
        network = DeepLabV3MutilDecoder(mode+model_id, project_dir="./").to(device)
    elif mode == "DeepLabV3Boundary":
        device = "cuda:0"
        network = DeepLabV3Boundary(mode+model_id, project_dir="./").to(device)
    elif mode == "DeepLabV3AddBoundary":
        device = "cuda:1"
        network = DeepLabV3AddBoundary(mode+model_id, project_dir="./").to(device)
    else:
        print("mode input error!")
        exit()

    train_dataset = DatasetTrain(data_path="./data/train/images/",
                                mask_path="./data/train/masks/",
                                boundary_path="./data/train/boundarys/")
    val_dataset = DatasetVal(data_path="./data/val/images/",
                            mask_path="./data/val/masks/",
                            boundary_path="./data/val/boundarys/")

    num_train_batches = int(len(train_dataset)/batch_size)
    num_val_batches = int(len(val_dataset)/batch_size)
    print ("num_train_batches:", num_train_batches)
    print ("num_val_batches:", num_val_batches)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=1)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size, shuffle=False,
                                            num_workers=1)

    params = add_weight_decay(network, l2_value=0.0001)
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # with open("/root/deeplabv3/data/cityscapes/meta/class_weights.pkl", "rb") as file: # (needed for python3)
    #     class_weights = np.array(pickle.load(file))
    # class_weights = torch.from_numpy(class_weights)
    # class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

    # loss function
    # loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    loss_fn = nn.CrossEntropyLoss()

    epoch_losses_train = []
    epoch_losses_val = []
    min_loss = 100
    for epoch in range(num_epochs):
        print ("###########################")
        print ("######## NEW EPOCH ########")
        print ("###########################")
        print ("epoch: %d/%d" % (epoch+1, num_epochs))
        start = time.time()

        ############################################################################
        # train:
        ############################################################################
        network.train() # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (imgs, label_imgs, label_boundarys) in enumerate(train_loader):
            #current_time = time.time()

            imgs = Variable(imgs).to(device) # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device) # (shape: (batch_size, img_h, img_w))
            label_boundarys = Variable(label_boundarys.type(torch.LongTensor)).to(device) # (shape: (batch_size, img_h, img_w))

            output_mask, output_boundary = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

            # compute the loss:
            # print(outputs.shape)
            # print(label_imgs.shape)
            loss_mask = loss_fn(output_mask, label_imgs)
            # loss_mask_value = loss_mask.data.cpu().numpy()

            loss_boundary = loss_fn(output_boundary, label_boundarys)
            # loss_boundary_value = loss_boundary.data.cpu().numpy()
            loss = loss_mask + loss_boundary
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            # optimization step:
            optimizer.zero_grad() # (reset gradients)
            loss.backward() # (compute gradients)
            optimizer.step() # (perform optimization step)

            #print (time.time() - current_time)

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)
        with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
            pickle.dump(epoch_losses_train, file)
        print ("train loss: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_train, "k^")
        plt.plot(epoch_losses_train, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("train loss per epoch")
        plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
        plt.close(1)

        print ("####")

        ############################################################################
        # val:
        ############################################################################
        network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (imgs, label_imgs, label_boundarys, _) in enumerate(val_loader):
            with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
                imgs = Variable(imgs).to(device) # (shape: (batch_size, 3, img_h, img_w))
                label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device) # (shape: (batch_size, img_h, img_w))
                label_boundarys = Variable(label_boundarys.type(torch.LongTensor)).to(device) # (shape: (batch_size, img_h, img_w))

                output_mask, output_boundary = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

                # compute the loss:
                # print(outputs.shape)
                # print(label_imgs.shape)
                loss_mask = loss_fn(output_mask, label_imgs)
                # loss_mask_value = loss_mask.data.cpu().numpy()

                loss_boundary = loss_fn(output_boundary, label_boundarys)
                # loss_boundary_value = loss_boundary.data.cpu().numpy()
                loss = loss_mask + loss_boundary
                loss_value = loss.data.cpu().numpy()
                batch_losses.append(loss_value)

        epoch_loss = np.mean(batch_losses)
        epoch_losses_val.append(epoch_loss)
        with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
            pickle.dump(epoch_losses_val, file)
        print ("val loss: %g" % epoch_loss)
        cost = time.time()-start
        print ("epoch cost:" , cost, "time left:", (num_epochs-epoch)*cost)
        plt.figure(1)
        plt.plot(epoch_losses_val, "k^")
        plt.plot(epoch_losses_val, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("val loss per epoch")
        plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
        plt.close(1)

        # save the model weights to disk:
        if epoch_loss < min_loss:
            print("save model epoch_loss:", epoch_loss)
            min_loss = epoch_loss
            checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
            torch.save(network.state_dict(), checkpoint_path)
