# 项目

利用tensorflow+LeNet实现MNIST手写体数字识别

# 说明

## 环境配置

```html
电脑环境：windows10

python3.6.5

第三方库：requirements.txt
```

## 数据集

共有7万张图片。其中6万张用于训练神经网络，1万张用于测试神经网络。

每张图片是一个28\*28像素点的0~9的手写数字图片。

黑底白字。黑底用0表示，白字用0~1之间的浮点数表示，越接近1，颜色越白。

[MNIST下载](http://yann.lecun.com/exdb/mnist/)

> 提示：项目中已经包含数据集，不需要你再下载！

## 仓库

本仓库包括以下：

- `data`：里面包含下载号的MNIST手写体数据集；
- `requirements.txt`：第三方库；
- `main.py`：利用tensorflow+LeNet实现手写体识别主程序;

# 使用

请先配置`python`运行环境：

```html
pip install -r requirements.txt
```

运行程序：
```html
python main.py
```

# 查看

在运行代码文件夹下运行：
```html
tensorboard --logdir=log
```

即可打开`http://localhost:6006/` 查看运行的情况：

<center><img src="https://s1.ax1x.com/2020/08/10/abcPds.png" alt="abcPds.png" border="0" /></center>


