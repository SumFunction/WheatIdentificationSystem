# 项目名称
小麦识别系统

# 使用说明

该版本只是部署上线版本，模型训练版本请下载另外一个分支版本。

请下载对应的权重文件放在对应的weights文件夹下才能进行使用。

将对应的前端界面导入到微信开发工具，后端启动app.py文件即可。

# 模型权重下载地址

目前阿里云盘暂不支持分享

# 项目结构说明

## **app.py**

后端监听程序，负责接收请求并调用模型识别

## classifyDisease文件夹

存放resnet检测模型脚本。

### weights文件夹

存放该模型的权重文件

### getResnet.py

模型推理文件

## findWheat文件夹

存放yolov5检测模型脚本

### weights文件夹

存放该模型的权重文件

### detect.py

模型推理文件

### runs文件夹

保存检测结果文件

其余一些文件均为辅助推理的文件或者文件夹

## tempPhotoResnet文件夹

暂存病害识别功能用户上传的图片

## tempPhotoYolo文件夹

暂存麦穗检测功能用户上传的图片

# 记录模型上线部署的一些坑

## 1.yolov5的load模型问题。

yolov5模型要求，起始的根目录与weights文件夹和models文件夹相同，否则会报“no module”的错误。

具体解决方法为：

我在app.py中调用推理文件，所以起始根目录为‘WheatProject\After-end’

但是，实际上需要将根目录放在'‘WheatProject\After-end\findWheat'下，与上面两个文件夹同级，故在模型加载前加入下面一行命令即可：

```python
import sys
sys.path.insert(0,'./findWheat')
```

产生上述问题的主要原因是yolov5保存模型的机制，他造成了我们加载模型需要严格的路径限制。

像我们平常写的模型保存，只保存状态字典，则输入对应的相对路径即可加载完成，没有这么多问题。

<a href="https://github.com/pytorch/pytorch/issues/3678">问题解决参考链接1</a>

<a href="https://github.com/ultralytics/yolov5/issues/353">问题解决参考链接2</a>

