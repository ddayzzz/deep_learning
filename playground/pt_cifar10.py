# 定义一个神经网络
# based on https://zhuanlan.zhihu.com/p/25572330
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch
from torch import optim
from torch import autograd


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)  # 第一层卷积
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)  # 进行 Wx + b
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        定义前向传播的过程
        :param x:
        :return:
        """
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # 2x2
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 如果大小的正方形, 直接使用数值也可以表示边长
        # x = x.view(-1, self.num_flat_features(x))  # 类似于 reshape 在 flatten 的作用, https://pytorch.org/docs/stable/tensors.html?highlight=view#torch.Tensor.view
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        """
        在各种卷积层输出之后, 进行扁平化操作
        :param x:
        :return:
        """
        size = x.size()[1:]  # 去掉 batch 的 shape
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def show_image(loader):
    import matplotlib.pyplot as plt
    import numpy as np

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # show some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    plt.show()
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net()
criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':
    # show_image(testloader)
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 输入图像和标签
            # 变成变量输入到网络中
            inputs, labels = autograd.Variable(data[0]), autograd.Variable(data[1])
            # 优化器清空
            optimizer.zero_grad()
            # forward + backward + optimizer
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 统计数据
            running_loss += loss.data
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("Finished!")