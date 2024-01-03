import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "dataset/img_9.jpg"
img = Image.open(image_path)
print(img)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
img = transform(img)

class AlexNet(torch.nn.Module):
    def __init__(self, num_classes, init_weights=False):
        super(AlexNet, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=2),
                                          torch.nn.BatchNorm2d(64),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0))

        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(64, 192, kernel_size=4, stride=1, padding=1),
                                          torch.nn.BatchNorm2d(192),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0))

        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(384),
                                          torch.nn.ReLU(inplace=True))

        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(256),
                                          torch.nn.ReLU(inplace=True))

        self.layer5 = torch.nn.Sequential(torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(256),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.avgpool = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(output_size=(3, 3)))

        self.fc1 = torch.nn.Sequential(torch.nn.Dropout(p=0.5, inplace=False),
                                       torch.nn.Linear(256 * 3 * 3, 1024),
                                       torch.nn.ReLU(inplace=True))

        self.fc2 = torch.nn.Sequential(torch.nn.Dropout(p=0.5, inplace=False),
                                       torch.nn.Linear(1024, 1024),
                                       torch.nn.ReLU(inplace=True))

        self.fc3 = torch.nn.Sequential(torch.nn.Dropout(p=0.5, inplace=False),
                                       torch.nn.Linear(1024, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

model = torch.load("AN_model58.pth")
img = torch.reshape(img, (1, 3, 32, 32))
img = img.cuda()
with torch.no_grad():
    output = model(img)
print(output)
target = output.argmax(1).item()

if target == 0:
    print(f"This image is 'airplane' ")
if target == 1:
    print(f"This image is 'car' ")
if target == 2:
    print(f"This image is 'bird' ")
if target == 3:
    print(f"This image is 'cat' ")
if target == 4:
    print(f"This image is 'deer' ")
if target == 5:
    print(f"This image is 'dog' ")
if target == 6:
    print(f"This image is 'frog' ")
if target == 7:
    print(f"This image is 'horse' ")
if target == 8:
    print(f"This image is 'ship' ")
if target == 9:
    print(f"This image is 'truck' ")