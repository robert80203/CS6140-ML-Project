import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import sklearn
import sklearn.metrics as metrics
import pprint


##################
#11112022
# Conclusions
# 1. Replace the ReLU function with sigmoid and then we do not need the softmax function
##################

class gsp(nn.Module):
    def __init__(self, prunning_set=None):
        super(gsp, self).__init__()
        self.prunning_set = prunning_set
        self.main = nn.ParameterList()
        in_out_list = [[784, 20],
                       [20, 20],
                       [20, 10]]
        for i in range(len(in_out_list)):
            self.main.append(nn.Parameter(torch.randn(in_out_list[i][0], in_out_list[i][1]).normal_(mean=0, std=0.01), requires_grad=True))
        self.num_layers = len(in_out_list)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, x.size(2) * x.size(3))
        for i in range(self.num_layers - 1):
            x = self.activation(torch.matmul(x, self.main[i]))
        output = torch.matmul(x, self.main[self.num_layers - 1])
        if self.prunning_set is not None:
            prunning_list = []
            for i in self.prunning_set:
                prunning_list.append(torch.softmax(self.main[i], dim=0))
            return output, prunning_list
        else:
            return output

class gsp2(nn.Module):
    def __init__(self, prunning_set=None):
        super(gsp2, self).__init__()
        self.prunning_set = prunning_set
        self.main = nn.ParameterList()
        in_out_list = [[784, 20],
                       [20, 20],
                       [20, 10]]
        for i in range(len(in_out_list)):
            self.main.append(nn.Parameter(torch.randn(in_out_list[i][0], in_out_list[i][1]).normal_(mean=0, std=0.01), requires_grad=True))
        self.num_layers = len(in_out_list)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, x.size(2) * x.size(3))
        for i in range(self.num_layers - 1):
            x = self.activation(torch.matmul(x, self.main[i]))
        output = torch.matmul(x, self.main[self.num_layers - 1])

        if self.prunning_set is not None:
            prunning_list = []
            for i in self.prunning_set:
                prunning_list.append(self.main[i])
            return output, prunning_list
        else:
            return output

def step_one():
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=16)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=16)

    device = torch.device("cuda")
    model = gsp(prunning_set=None).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        score = 0
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            if idx % 1000 == 0:
                print('epoch %d, iter %d, loss %.3f'%(epoch, idx, loss.item()))
            
        for idx, batch in enumerate(test_loader):
            x, y = batch
            x = x.to(device)
            output = model(x)
            argmax_output = torch.argmax(output, dim=1)
            score += metrics.precision_score(y, argmax_output.cpu(), average='micro')
        print(score/len(test_loader))
        torch.save(model.state_dict(), 'checkpoint.pt')

def step_two():
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=16)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=16)

    device = torch.device("cuda")
    model = gsp(prunning_set=[1]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load('checkpoint.pt'))

    for epoch in range(10):
        score = 0
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            output, w = model(x)
            ce_loss = criterion(output, y)
            entropy_loss = 0
            for i in range(len(w)):
                entropy_loss += - 100 * (w[i] * torch.log(w[i])).sum() / w[i].size(1)
            loss =  ce_loss + entropy_loss 
            loss.backward()
            optimizer.step()
            if idx % 1000 == 0:
                print('epoch %d, iter %d, CEloss %.3f, Eloss %.3f'%(epoch, idx, ce_loss.item(), entropy_loss.item()))

        for idx, batch in enumerate(test_loader):
            x, y = batch
            x = x.to(device)
            output, _ = model(x)
            argmax_output = torch.argmax(output, dim=1)
            score += metrics.precision_score(y, argmax_output.cpu(), average='micro')
        print(score/len(test_loader))
        torch.save(model.state_dict(), 'checkpoint_step2.pt')

def evaluation():
    model = gsp(prunning_set=[1])
    model.load_state_dict(torch.load('checkpoint.pt'))
    pprint.pprint(torch.softmax(model.main[1][:, 0], dim=0))
    model.load_state_dict(torch.load('checkpoint_step2.pt'))
    pprint.pprint(torch.softmax(model.main[1][:, 0], dim=0))
    pprint.pprint(model.main[1][:, 0])


if __name__ == '__main__':
    step_one()
    step_two()
    evaluation()