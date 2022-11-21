import torch
import pprint
import sklearn
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms


##################
#Issues
#1. Add convolutional layers
#2. 3 fc layers reach the maximum
#3. step1, step2, 1 layer works
#4. spep1, step2, 1, 2 layer is still running
#5. step1, 0 layer failed

##################
#Issues
#1. Number of layers
#2. Removing those zero-like neurons
#3. 
##################
#11112022
# Conclusions
# 1. Entropy loss with softmax cannot control the real values of weights
# 11142022
# 1. Replace some ReLU with sigmoid and process and regularize them by softmax and Entropy loss
##################



torch.set_printoptions(precision=3)

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
            #self.main.append(nn.Parameter(torch.zeros(in_out_list[i][0], in_out_list[i][1]) + 0.01, requires_grad=True))
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
                #prunning_list.append(torch.softmax(self.activation(torch.matmul(x, self.main[i])), dim=1))
                prunning_list.append(torch.softmax(self.main[i], dim=0))
            return output, prunning_list
        else:
            return output

class gsp2(nn.Module):
    def __init__(self, prunning_set=None):
        super(gsp2, self).__init__()
        self.prunning_set = prunning_set
        self.prunning_act_set = [1]
        self.main = nn.ParameterList()
        in_out_list = [[784, 20],
                       [20, 20],
                       [20, 10]]
        for i in range(len(in_out_list)):
            self.main.append(nn.Parameter(torch.randn(in_out_list[i][0], in_out_list[i][1]).normal_(mean=0, std=0.01), requires_grad=True))
        self.num_layers = len(in_out_list)
        self.activation = nn.ReLU()
        self.w_act = nn.Sigmoid()

    def forward(self, x, threshold=False):
        x = x.view(-1, x.size(2) * x.size(3))
        for i in range(self.num_layers - 1):
            if i in self.prunning_act_set: # specifically for layer 1
                w = self.w_act(self.main[i])
                if threshold:
                    w[w > 0.9] = 1.
                    w[w <= 0.9] = 0.
                x = self.activation(torch.matmul(x, w))
            else:
                x = self.activation(torch.matmul(x, self.main[i]))
        output = torch.matmul(x, self.main[self.num_layers - 1])

        if self.prunning_set is not None:
            prunning_list = []
            for i in self.prunning_set:
                prunning_list.append(torch.softmax(self.w_act(self.main[i]), dim=0))
            return output, prunning_list
        else:
            return output

class gsp3(nn.Module): # https://github.com/pytorch/examples/blob/main/mnist/main.py
    def __init__(self, prunning_set=None):
        super(gsp3, self).__init__()
        self.prunning_set = prunning_set
        self.prunning_act_set = [1, 2]
        self.main = nn.ParameterList()
        # in_out_list = [[9216, 128],
        #                [128, 64],
        #                [64, 32],
        #                [32, 16],
        #                [16, 10]]
        in_out_list = [[9216, 128],
                       [128, 20],
                       [20, 10]]
        for i in range(len(in_out_list)):
            self.main.append(nn.Parameter(torch.randn(in_out_list[i][0], in_out_list[i][1]).normal_(mean=0, std=0.01), requires_grad=True))
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.num_layers = len(in_out_list)
        self.activation = nn.ReLU()
        self.w_act = nn.Sigmoid()

    def forward(self, x, threshold=False):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)

        for i in range(self.num_layers - 1):
            if i in self.prunning_act_set: # specifically for layer 1
                w = self.w_act(self.main[i])
                if threshold:
                    w[w > 0.9] = 1.
                    w[w <= 0.9] = 0.
                x = self.activation(torch.matmul(x, w))
            else:
                x = self.activation(torch.matmul(x, self.main[i]))
        output = torch.matmul(x, self.main[self.num_layers - 1])

        if self.prunning_set is not None:
            prunning_list = []
            for i in self.prunning_set:
                prunning_list.append(torch.softmax(self.w_act(self.main[i]), dim=0))
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
    #model = gsp(prunning_set=None).to(device)
    model = gsp3(prunning_set=None).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
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
        scheduler.step()
            
        for idx, batch in enumerate(test_loader):
            x, y = batch
            x = x.to(device)
            output = model(x)
            argmax_output = torch.argmax(output, dim=1)
            score += metrics.precision_score(y, argmax_output.cpu(), average='micro')
        print(score/len(test_loader))
        torch.save(model.state_dict(), 'weights/gsp3/step1_pru_12.pt')

def step_two():
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=16)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=16, shuffle=False)

    device = torch.device("cuda")
    #model = gsp(prunning_set=[1]).to(device)
    model = gsp3(prunning_set=[1]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load('weights/gsp3/step1_pru_12.pt'))
    #model.load_state_dict(torch.load('weights/gsp2/checkpoint_step2.pt'))

    for epoch in range(40):
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
            loss =  0.1 * ce_loss + entropy_loss 
            loss.backward()
            optimizer.step()
            if idx % 1000 == 0:
                print('epoch %d, iter %d, CEloss %.3f, Eloss %.3f'%(epoch, idx, ce_loss.item(), entropy_loss.item()))
        model.eval()
        for idx, batch in enumerate(test_loader):
            with torch.no_grad():
                x, y = batch
                x = x.to(device)
                output, _ = model(x)
                argmax_output = torch.argmax(output, dim=1)
                score += metrics.precision_score(y, argmax_output.cpu(), average='micro')
        print(score/len(test_loader))
        torch.save(model.state_dict(), 'weights/gsp3/step2_pru_12.pt')
        model.train()

def evaluation():
    model = gsp(prunning_set=[1])
    model.load_state_dict(torch.load('weights/gsp2/checkpoint.pt'))
    print(torch.softmax(model.main[1][:, 0], dim=0))
    model.load_state_dict(torch.load('weights/gsp2/checkpoint_step2.pt'))
    print(torch.softmax(model.main[1][:, 0], dim=0))
    print(model.main[1][:, 0])

def evaluation2():
    device = torch.device("cuda")
    model = gsp3(prunning_set=[1])
    model = model.to(device)
    model.load_state_dict(torch.load('weights/gsp3/step1_pru_12.pt'))
    print('step1, weights after sigmoid')
    print(torch.sigmoid(model.main[1][:, 0]))
    print('='*20)
    model.load_state_dict(torch.load('weights/gsp3/step2_pru_12.pt'))
    print('step2, weights after sigmoid')
    print(torch.sigmoid(model.main[1][:, 0]))
    print('='*20)
    print('step2, weights after sigmoid and softmax')
    print(torch.softmax(torch.sigmoid(model.main[1][:, 0]), dim=0))
    print('='*20)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset2 = datasets.MNIST('./', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=16, shuffle=False)
    score = 0
    model.eval()
    for idx, batch in enumerate(test_loader):
        x, y = batch
        x = x.to(device)
        output, _ = model(x)
        argmax_output = torch.argmax(output, dim=1)
        score += metrics.precision_score(y, argmax_output.cpu(), average='micro')
    print(score/len(test_loader))

if __name__ == '__main__':
    #step_one()
    step_two()
    #evaluation2()