import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "dataset/img_10.jpg"
img = Image.open(image_path)
print(img)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
img = transform(img)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("mymodel_40.pth")
img = torch.reshape(img, (1, 3, 32, 32))
img = img.cuda()
with torch.no_grad():
    output = model(img)
print(output)
target = output.argmax(1).item()

if target == 0:
    print(f"{image_path} is 'airplane' ")
if target == 1:
    print(f"{image_path} is 'car' ")
if target == 2:
    print(f"{image_path} is 'bird' ")
if target == 3:
    print(f"{image_path} is 'cat' ")
if target == 4:
    print(f"{image_path} is 'deer' ")
if target == 5:
    print(f"{image_path} is 'dog' ")
if target == 6:
    print(f"{image_path} is 'frog' ")
if target == 7:
    print(f"{image_path} is 'horse' ")
if target == 8:
    print(f"{image_path} is 'ship' ")
if target == 9:
    print(f"{image_path} is 'truck' ")