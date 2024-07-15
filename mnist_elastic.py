import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

def train(args, rank, local_rank):
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    model = Net().to(local_rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(1, args.epochs + 1):
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(local_rank), target.cuda(local_rank)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--world-size', type=int, default=int(os.environ.get('WORLD_SIZE', 1)), help='number of distributed processes')
    parser.add_argument('--rank', type=int, default=int(os.environ.get('RANK', 0)), help='rank of this process')
    parser.add_argument('--local-rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)), help='local rank of this process')
    args = parser.parse_args()

    train(args, args.rank, args.local_rank)
