import torch
import torchvision
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import cv2
import numpy as np
import sys



# converts resnet18 model to TRT model
# usage: python convert2trt.py <mode> <source> <target>
# currently implemented modes: 'resnet18', 'resnet50'

__all__ = ['ResNet']

def convert():
    if len(sys.argv) < 4:
        print("usage: python convert2trt.py <mode> <source> <target>")
        sys.exit(-1)

    MODE = sys.argv[1]
    resnet18_model = ResNet(MODE)
    PATH = sys.argv[2]
    TARGET = sys.argv[3]
    checkpoint = torch.load(PATH)
    resnet18_model.load_state_dict(checkpoint['model_state_dict'])
    resnet18_model.cuda().eval()

    #create dummy data
    data = torch.zeros((1, 3, 224, 224)).cuda()

    # convert to TensorRT feeding sample data as input
    model_trt = torch2trt(resnet18_model, [data])

    print("{}".format(model_trt(data)))
    print("{}".format(resnet18_model(data)))

    torch.save(model_trt.state_dict(), TARGET)


class ResNet(nn.Module):

    def __init__(
        self, arch, pretrained=False, progress=False, num_cls=8, **kwargs
    ):
        super().__init__()

        if(arch == 'resnet18'):
            self.model = torchvision.models.resnet18(num_classes=num_cls)
        elif(arch == 'resnet50'):
            self.model = torchvision.models.resnet50(num_classes=num_cls)
        else:
            raise NotImplementedError("Resnet to be implemented:", arch)

    def forward(self, input_image):
        cls = self.model(input_image)
        return cls

if __name__ == "__main__":
    convert()
