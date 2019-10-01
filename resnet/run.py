import torch
import torch.nn as nn
import torch.optim as optim
import argparse

import torchvision
import torchvision.transforms as transforms

from resnet.model import resnet18, resnet34, resnet50, resnet101, resnet152

# Parse hyperparameters from args
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)

# Transform train and test dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Build train and test dataset
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Build data loader of train and test
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Build a resnet models
if args.model == 'resnet18':
    model = resnet18()
elif args.model == 'resnet34':
    model = resnet34()
elif args.mode == 'resnet50':
    model = resnet50()
elif args.mode == 'resnet101':
    model = resnet101()
elif args.mode == 'resnet152':
    model = resnet152()
else:
    raise Exception('Put \'resent18\' or \'resnet34\' for model argument')
model = model.to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    print('=============================================')
    print('Epoch: %d' % epoch)
    print('=============================================')
    print('[Train Started]')
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print('[%d] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            batch_idx + 1, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


best_acc = 0
def test():
    print('[Test Started]')
    model.eval()
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
    cur_acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 70 == 0:
                cur_acc = 100. * correct / total
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), cur_acc, correct, total))
        best_acc = max(best_acc, cur_acc)
        print('best_acc: %.3f' % (best_acc))


if __name__ == '__main__':
    for epoch in range(0, 200):
        train(epoch)
        test()
