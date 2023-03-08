# camera-ready
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from functools import partial


nonlinearity = partial(F.relu, inplace=True)

from model.resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from model.aspp import ASPP, ASPP_Bottleneck

class DeeplabV3MergeBoundary(nn.Module):
    def __init__(self, model_id, project_dir):
        super(DeeplabV3MergeBoundary, self).__init__()

        self.num_classes = 6
        self.boundary_classes = 2

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        # self.resnet = ResNet18_OS8() # NOTE! specify the type of ResNet here
        # self.aspp = ASPP(num_classes=512) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead
        self.resnet = ResNet50_OS16() # NOTE! specify the type of ResNet here
        self.aspp = ASPP_Bottleneck(num_classes=1024) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

        self.decoder4 = DecoderBlock(1024, 512)
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, self.num_classes)
        self.merge = MergeBlock([512, 256, 128], self.num_classes)

        self.decoder41 = DecoderBlock(1024, 512)
        self.decoder31 = DecoderBlock(512, 256)
        self.decoder21 = DecoderBlock(256, 128)
        self.decoder11 = DecoderBlock(128, self.boundary_classes)


    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        f1 = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        # output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))
        output1 = self.decoder41(f1)
        output1_1 = F.upsample(output1, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))
        output2 = self.decoder31(output1)
        output2_1 = F.upsample(output2, size=(h, w), mode="bilinear")
        output3 = self.decoder21(output2)
        output3_1 = F.upsample(output3, size=(h, w), mode="bilinear")
        output4 = self.decoder11(output3)
        
        # mask
        outpu5 = self.decoder4(f1)
        output5_1 = F.upsample(outpu5, size=(h, w), mode="bilinear")
        output6 = self.decoder3(outpu5)
        output6_1 = F.upsample(output6, size=(h, w), mode="bilinear")
        output7 = self.decoder2(output6)
        output7_1 = F.upsample(output7, size=(h, w), mode="bilinear")
        output8 = self.decoder1(output7)

        output8 = self.merge([output1_1+output5_1, output2_1+output6_1, output3_1+output7_1], output8)
        return output8, output4

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)


class MergeBlock(nn.Module):
    def __init__(self, mask_channels, n_filters):
        super(MergeBlock, self).__init__()
        self.conv0 = nn.Conv2d(mask_channels[0], n_filters, 1)
        self.conv1 = nn.Conv2d(mask_channels[1], n_filters, 1)
        self.conv2 = nn.Conv2d(mask_channels[2], n_filters, 1)
        self.weight = torch.nn.Parameter(torch.Tensor(np.ones([1,len(mask_channels)])))

    def forward(self, masks, input):
        self.mask0 = self.conv0(masks[0])
        self.mask1 = self.conv1(masks[1])
        self.mask2 = self.conv2(masks[2])
        self.mask = self.mask0*self.weight[0][0]+self.mask1*self.weight[0][1]+self.mask2*self.weight[0][2]
        return input.mul(self.mask)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x