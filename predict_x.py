import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

image = cv2.imread(
    "/Users/celia/Desktop/python/Pneumoia-Detection-CNN/xray_test.jpg",
    cv2.IMREAD_GRAYSCALE
)
if image is None:
    raise FileNotFoundError("没有找到图片，请检查路径")
print("Original shape:", image.shape)

plt.imshow(image, cmap="gray")
plt.title("Original X-ray")
plt.axis("off")
plt.show()

image_resized = cv2.resize(image,(28,28))
x = torch.from_numpy(image_resized).float()/255
x = x.unsqueeze(0).unsqueeze(0)
print("tensor shape : " ,x.shape)

model = nn.Sequential(
    nn.Conv2d(1,8,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(8,16,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(16*7*7,2)
)
model.load_state_dict(
    torch.load(
        "pneumonia_cnn.pth",
        map_location="cpu",
        weights_only=True
    )
)
model.eval()
with torch.no_grad():
    logits = model(x)
    probabilite = torch.softmax(logits, dim =1)
    prediction = torch.argmax(logits,dim = 1)

print("logits :",logits)
print("probabilite : ", probabilite)
print("prediction : ", prediction.item())

class_names = ["Normal", "Pneumonia"]

predicted_class = class_names[prediction.item()]

score = probabilite[0, prediction.item()].item()

plt.figure(figsize=(6, 6))

plt.imshow(image, cmap="gray")

plt.title(
    f"Prediction: {predicted_class}\n"
    f"Softmax Score: {score:.2%}"
)

plt.axis("off")
plt.show()