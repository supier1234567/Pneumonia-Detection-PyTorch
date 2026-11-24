# Chest X-ray Pneumonia Classification using PyTorch

## 项目介绍

本项目基于 PyTorch 深度学习框架，构建卷积神经网络（CNN），实现胸部 X 光医学影像的肺炎自动分类。

使用 PneumoniaMNIST 数据集，通过图像预处理、神经网络训练、模型评估和预测，实现从医学影像数据到 AI 分类结果的完整流程。

## 技术栈

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- MedMNIST

## 项目主要内容

**1. 医学影像数据处理**

- 使用 MedMNIST 加载 PneumoniaMNIST 数据集
- 使用 OpenCV 进行图像读取与处理
- 将图像转换为 PyTorch Tensor，供神经网络训练使用

**2. CNN 模型构建**

- 利用 PyTorch 构建卷积神经网络
- 通过卷积层提取医学影像特征
- 完成 Normal 和 Pneumonia 二分类任务

**3. 模型训练与评估**

- 实现模型训练及参数优化
- 使用验证集评估分类准确率
- 通过 Softmax 将模型输出转换为类别概率

**4. 医学影像预测**

- 输入胸部 X 光图像
- 输出预测类别：Normal / Pneumonia
- 显示模型预测概率

## 数据集

PneumoniaMNIST — MedMNIST

https://medmnist.com/

## 项目意义

本项目用于学习医学影像 AI 分析的基本流程，为进一步研究医学图像分割（U-Net）、医学影像深度学习及 AI 辅助诊断技术奠定实践基础。

**说明：本项目仅用于学习与研究，不用于临床诊断。**