# This implementation is based on the ResNet implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets, models, transforms
import cv2
from torch2trt import torch2trt
from torch2trt import TRTModule


class TRTModel:

    dictionary = {
        "Contempt": 7,
        "Anger": 6,
        "Disgust": 5,
        "Fear": 4,
        "Surprise": 3,
        "Sadness": 2,
        "Happiness": 1,
        "Neutral": 0,
    }

    def __init__(self, size=224):
        self.model_trt = TRTModule()
        PATH = '../models/resnet50.224.trt.pth'
        self.model_trt.load_state_dict(torch.load(PATH))
        self.model_trt.eval().cuda()

        self.label_map = dict((v, k) for k, v in self.dictionary.items())
        self.size = size

    def image_loader(self, image):
        loader = transforms.Compose([transforms.ToTensor()])
        image = loader(image).float()
        image = image.unsqueeze(0)
        return image

    def resize_image(self, image):
        image = cv2.resize(image, (self.size, self.size))
        return image

    def face_expression(self, image):
        resized_image = self.resize_image(image)
        tensor_image = self.image_loader(resized_image)
        tensor_image = tensor_image.cuda().contiguous()
        with torch.no_grad():
            outputs = self.model_trt(tensor_image)
            _, predicted = torch.max(outputs, 1)
            idx = predicted.item()
            face_expression = self.label_map[idx]
            return face_expression
