import medmnist
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,accuracy_score,
    precision_score,recall_score,f1_score)

transforms = transforms.ToTensor()


# 下载胸部x光数据集
train_dataset = medmnist.PneumoniaMNIST(
    split = "train",
    transform=transforms,
    download = True
)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
#images, labels = next(iter(train_loader))

# image, label = dataset[0] # 读取第一张照片
# image_array = np.array(image) # PIL图片转换成numpy数组

# plt.imshow(image_array, cmap="gray")
# plt.axis("off")
# plt.show()
# image_cv = cv2.resize(image_array,(28,28)) #opencv 
# x = torch.from_numpy(image_cv).float()/255 #opencv 变成tensor数组
# x = x.unsqueeze(0).unsqueeze(0)
# print("original shape : ", image_array.shape)
# print("tensor shape : ", x.shape)
# print("label : ",label)
# plt.imshow(x[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
# plt.title("Preprocessed Chest X-ray")
# plt.axis("off")
# plt.show()

# 进入CNN
conv = nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
    stride=1,
    padding=1
)
#y = conv(x)
# print("Before CNN:", x.shape)
# print("After CNN:", y.shape)
# # 转换为 NumPy，方便 Matplotlib 显示
# feature_maps = y.detach().numpy()

# plt.figure(figsize=(12, 6))

# for i in range(8):
#     plt.subplot(2, 4, i + 1)
#     plt.imshow(feature_maps[0, i], cmap="gray")
#     plt.title(f"Feature Map {i+1}")
#     plt.axis("off")

# plt.tight_layout()
# plt.show()

#创建模型
model = nn.Sequential(
    nn.Conv2d(1,8,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2,2),

    nn.Conv2d(8,16,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2,2),

    nn.Flatten(),
    nn.Linear(16*7*7,2)
)
#y = model(x)
#print(y.shape)
# probabilite = torch.softmax(output,dim = 1)
# prediction = torch.argmax(output,dim = 1)
# print("input : " , x.shape)
# print("output : ", output.shape)
# print("logits : ", output)
# print("probablite  : ",probabilite)
# print("prediction : ", prediction)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.001
)
num_epochs = 5 # 让cnn 完整学习整个训练集5遍
loss_history = []
for epoch in range(num_epochs): 
    model.train() # 现在把模型切换到训练模式
    running_loss = 0
    for images , labels in train_loader : # 负责每个batch的x光图片和标签
        labels = labels.squeeze(1).long()
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*images.size(0)
    epoch_loss = running_loss/len(train_dataset)
    loss_history.append(epoch_loss)
    print(
        f"Epoch {epoch+1}/{num_epochs}, "
        f"Loss: {epoch_loss:.4f}"
    )

val_dataset = medmnist.PneumoniaMNIST(
    split = "val",
    transform= transforms,
    download=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_dataset = medmnist.PneumoniaMNIST(
    split="test",
    transform=transforms,
    download=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
model.eval()

all_predictions = [] #CNN预测的标签
all_labels = [] #真实标签

with torch.no_grad():
    for images,labels in test_loader:
        labels = labels.squeeze(1).long()
        logits = model(images)
        prediction = torch.argmax(logits, dim = 1)
        all_predictions.extend(prediction.numpy())
        all_labels.extend(labels.numpy())
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0,1]
)
accuracy = accuracy_score(all_labels,all_predictions)
precision = precision_score(
    all_labels,
    all_predictions,
    zero_division= 0
)
recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)
f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)
print(cm)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["normal", "pneumonia"]
)
disp.plot(cmap="Blues")
plt.title("CNN Confusion Matrix")
plt.show()
# plt.plot(range(1, num_epochs + 1), loss_history, marker="o")
# plt.xlabel("Epoch")
# plt.ylabel("Training Loss")
# plt.title("CNN Training Loss")
# plt.grid(True)
# plt.show()

torch.save(
    model.state_dict(),
    "pneumonia_cnn.pth"
)

