#分类推理文件
from torchvision import models
import torch
from torch import nn
from PIL import Image
import numpy as np
class Resnet():
    def __init__(self,numClass,path):
        self.device = torch.device("cuda:0")
        self.model = models.resnet50()

        finFeatureNum = self.model.fc.in_features  # 得到最后全连接层的特征个数
        self.model.fc = nn.Sequential(nn.Linear(finFeatureNum, numClass), nn.LogSoftmax(dim=1))  # 设置最后一层需要分类的种类
        checkPoint = torch.load(path)
        self.model.load_state_dict(checkPoint['stateDict'])
        self.model.to(self.device)

        print("resnet模型初始化完毕！最佳精度：",checkPoint['bestAcc'])
    def getRes(self,file):
        img = Image.open(file)
        img = img.resize((224, 224))
        img = img.convert("RGB")
        img = np.array(img) / 255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 0).astype(np.float32)
        self.model.eval()
        outputs = self.model(torch.from_numpy(img).to(self.device))
        _, preds_tensor = torch.max(outputs, 1)
        index = preds_tensor.item()
        outputs = outputs.cpu().detach().numpy().squeeze()

        return index,np.exp(outputs)[index]