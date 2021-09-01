##微服务端口
import os
from flask import *
from findWheat.detect import Detector#yolo模型接口
import datetime
from flask_cors import CORS,cross_origin
import time
from torchvision import models
import torch
from torch import nn
from PIL import Image
import numpy as np
from classifyDisease.getResnet import Resnet
from findWheat.models.experimental import attempt_load




app = Flask(__name__,static_folder=r'findWheat\runs\detect\exp')
cors = CORS(app)


def getRes(fileName,imgUrl):#读取麦穗检测结果文件函数

    root = r'findWheat/runs/detect/exp/labels/' + fileName + '.txt'
    data = 0
    try:
        with open(root, 'r') as f:
            data = f.readlines()
    except:
        print("无检测目标")
    num = 0
    ID = 0
    res = {}
    tableData = []
    oneRes = {}
    oneRes['id'] = ID
    oneRes['name'] = '序号'
    tableData.append(oneRes)
    ID = ID + 1

    oneRes = {}
    oneRes['id'] = ID
    oneRes['name'] = 'x中心坐标'
    tableData.append(oneRes)
    ID = ID + 1

    oneRes = {}
    oneRes['id'] = ID
    oneRes['name'] = 'y中心坐标'
    tableData.append(oneRes)
    ID = ID + 1

    oneRes = {}
    oneRes['id'] = ID
    oneRes['name'] = '宽度'
    tableData.append(oneRes)
    ID = ID + 1

    oneRes = {}
    oneRes['id'] = ID
    oneRes['name'] = '高度'
    tableData.append(oneRes)
    ID = ID + 1

    if data != 0:
        for t in data:
            # ID = ID + 1
            oneRes = {}
            oneRes['id'] = ID
            oneRes['name'] = num + 1
            tableData.append(oneRes)
            ID = ID + 1

            tem = t.split()
            x_center = '%.2f' % (float(tem[1]) * 1024)
            y_center = '%.2f' % (float(tem[2]) * 1024)
            width = '%.2f' % (float(tem[3]) * 1024)
            height = '%.2f' % (float(tem[4]) * 1024)

            oneRes = {}
            oneRes['id'] = ID
            oneRes['name'] = x_center
            tableData.append(oneRes)
            ID = ID + 1

            oneRes = {}
            oneRes['id'] = ID
            oneRes['name'] = y_center
            tableData.append(oneRes)
            ID = ID + 1

            oneRes = {}
            oneRes['id'] = ID
            oneRes['name'] = width
            tableData.append(oneRes)
            ID = ID + 1

            oneRes = {}
            oneRes['id'] = ID
            oneRes['name'] = height
            tableData.append(oneRes)
            ID = ID + 1

            num = num + 1
            res['tableData'] = tableData
            res['num'] = num
            res['imgUrl'] = imgUrl
    else:
        res['tableData'] = tableData
        res['num'] = num
        res['imgUrl'] = imgUrl
    return jsonify(res)



@cross_origin()
@app.route('/uploadYolo', methods=['GET', 'POST'])
def uploadYolo():#麦穗识别接收接口
    file = request.files['file']#获取文件对象
    print(datetime.datetime.now(), file.filename)
    UPLOAD_FOLDER = r'temPhotoYolo'
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    if Detector(file_path,yolov5) == 1:
        print("成功提交")
        return getRes(file.filename[0:-4],'exp/'+file.filename)#

@cross_origin()
@app.route('/uploadResnet', methods=['GET', 'POST'])
def uoloadResnet():#小麦病害分类接收接口
    file = request.files['file']#获取文件对象
    print(datetime.datetime.now(), file.filename)
    UPLOAD_FOLDER = r'temPhotoResnet'
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    value = resnet.getRes(file_path)
    res = {
        'index':value[0],
        'acc':value[1].item()
    }
    print(res)
    return jsonify(res)

resnet = Resnet(7,r'classifyDisease\weights\best.pth')#加载resnet模型
yolov5 = attempt_load(r'findWheat\weights\best.pt', map_location="cuda:0") #加载yolov5模型，第一个参数 模型权重位置 第二个参数 是否调用GPU
app.run(host='127.0.0.1', port=5003)#需要外网访问得自己做穿透
