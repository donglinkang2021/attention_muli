from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def load_data_train(args):
    mnist_train = datasets.MNIST(
        "mnist-data", 
        train=True, 
        download=True, 
        transform=transform
    )
    train_loader = DataLoader(
        mnist_train, 
        batch_size= args.batch_size,
        shuffle=True
    )
    return train_loader

def load_data_test(args):
    mnist_test = datasets.MNIST(
        "mnist-data", 
        train=False, 
        download=True, 
        transform=transform
    )
    test_loader = DataLoader(
        mnist_test, 
        batch_size= args.batch_size,
        shuffle=True
    )
    return test_loader

class ConvNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=16, 
            kernel_size=5,
            stride=1,
        ) # 1*28*28 -> 16*24*24
        self.pool1 = nn.MaxPool2d(
            kernel_size=2
        ) # 16*24*24 -> 16*12*12
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        ) # 16*12*12 -> 32*12*12
        self.pool2 = nn.MaxPool2d(
            kernel_size=2
        ) # 32*12*12 -> 32*6*6
        self.flatten = nn.Flatten() # 32*6*6 -> 1152
        self.fc1 = nn.Linear(32*6*6, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)   # 1*28*28 -> 16*24*24
        x = self.relu(x)    # 16*24*24
        x = self.pool1(x)   # 16*24*24 -> 16*12*12
        x = self.conv2(x)   # 16*12*12 -> 32*12*12
        x = self.relu(x)    # 32*12*12
        x = self.pool2(x)   # 32*12*12 -> 32*6*6
        x = self.flatten(x) # 32*6*6 -> 1152
        x = self.fc1(x)     # 1152 -> 512
        x = self.relu(x)    # 512
        x = self.fc2(x)     # 512 -> 10
        return x

def train(args):
    epoches = args.epoches
    net = ConvNN().to(args.device)
    net.train()
    loss_fn = nn.CrossEntropyLoss()
    opitimizer = optim.Adam(net.parameters(), lr=args.lr)
    train_loader = load_data_train(args)
    loss = 0 
    for epoch in range(epoches):
        for i, (inputs, labels) in enumerate(train_loader):
            opitimizer.zero_grad()
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opitimizer.step()
            if i % args.print_every == 0:
                print(f"epoch: {epoch}/{epoches}, step: {i}, loss: {loss.item()}")
    torch.save(net.state_dict(), "conv.pth")

def test(args):
    net = ConvNN().to(args.device)
    net.load_state_dict(torch.load("conv.pth"))
    net.eval()
    test_loader = load_data_test(args)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = net(inputs)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {correct/total}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--print_args", action="store_true", default=False)
    parser.add_argument("--epoches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--print_every", type=int, default=100)
    args = parser.parse_args()
    return args

def print_args(args):
    args = {**vars(args)}
    ## 打印超参数
    print(''.join(['=']*80))
    tplt = "{:^20}\t{:^20}\t{:^20}"
    print(tplt.format("Name", "Value", "Type"))
    for k, v in args.items():
        v_type = type(v).__name__  # 获取参数的类型名称
        print(tplt.format(k, v, v_type))
    print(''.join(['=']*80))

if __name__ == "__main__":
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    if args.print_args:
        print_args(args)
    if args.train:
        train(args)
    if args.test:
        test(args)