# There are random function from 8 inputs and X random inputs added.
# We split data in 2 parts, on first part you will train and on second
# part we will test
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('./../utils')
import solutionmanager as sm
from gridsearch import GridSearch
        
class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.solution = solution
        self.hidden_size = self.solution.hidden_size
        self.linear_size = self.hidden_size *144
        RunningStats = True
        self.convol1 = nn.Conv2d(1,self.hidden_size, 5)
        self.batchnorm_conv = nn.BatchNorm2d(self.hidden_size)
        #self.conv1_drop = nn.Dropout2d()
        self.linear1 = nn.Linear(self.linear_size, 10) 
        self.batchnorm_ln1 = nn.BatchNorm1d(10)

    def forward(self, x):
        y = x
        x = self.convol1(x)
        x = F.relu(F.max_pool2d(x, 2))
        #x = self.conv1_drop(x)
        if self.solution.do_batch_norm:
            x = self.batchnorm_conv(x)
        x = x.view(-1, self.linear_size)
        x = self.linear1(x)
        if self.solution.do_batch_norm:
            x = self.batchnorm_ln1(x)
        x = F.log_softmax(x, dim=1)
        return x
        
class Solution():
    def __init__(self):
        # best so far: hs 40, lr 0.7, momentum 0.5
        # best so far2: hs 7, lr 0.07, momentum 0.5
        # 0.08/8, momentum 0.5, 144 neurons in linear layer (5 pixel window)
        self = self
        self.lr = 0.08
        self.hidden_size = 8
        self.momentum = 0.5
        self.do_batch_norm = False
        self.batch_size = 8
        #self.lr_grid = [0.005,0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 1,5]
        self.lr_grid = [0.05, 0.06, 0.08, 0.09, 0.1, 1.1, 1.2]
        self.hidden_size_grid = [4,5,6,7,8,9,10]
        #self.momentum_grid = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.grid_search = GridSearch(self).set_enabled(DoGridSearch)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        batches = train_data.size(0)//self.batch_size
        goodCount = 0
        goodLimit = batches
        # Put model in train mode
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum = self.momentum, weight_decay = 0.000)
        if DoGridSearch:
            print("LR = {}, HS = {}, Momentum = {}".format(self.lr, self.hidden_size, self.momentum))
        while True:
            ind = step%batches
            start_ind = self.batch_size * ind
            end_ind = start_ind + self.batch_size
            data = train_data[start_ind:end_ind]
            target = train_target[start_ind:end_ind]         
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1:
                break
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(data)
            # if x < 0.5 predict 0 else predict 1
            predict = output.max(1, keepdim=True)[1]
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = target.view(-1).size(0)
            # calculate loss
            loss = F.nll_loss(output, target)
            #loss = nn.BCELoss()(output,target)
    
            if loss.item() < 0.05: #correct / total>0.97:  # float(diff) < 0.3: 
                goodCount += 1
                if goodCount >= 10: # goodLimit:
                    break
            else:
                goodCount = 0
            # Number of correct predictions
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
            # self.print_stats(step, loss, correct, total)
            #if DoGridSearch and step % 50 == 0:
            #   self.print_stats(step, loss, correct, total, goodCount)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
            
        if DoGridSearch:
            data, target = context.get_case_data().test_data
            output = model(data)
            predict = output.max(1, keepdim=True)[1]
            correct = predict.eq(target.view_as(predict)).long().sum().item()      
            total = target.view(-1).size(0)
            loss = F.nll_loss(output, target)
            loss_record = round(loss.item(),3)
            predict_ratio = round(correct/total,3)
            time_solved = round(2-time_left,3)
            if BestSol["ratio"] < predict_ratio:
                BestSol["ratio"] = predict_ratio
                BestSol["time"] = time_solved
                BestSol["steps"] = step
                BestSol["LR"] = self.lr
                BestSol["HS"] = self.hidden_size
            if predict_ratio>0.945:
                self.print_stats(step, loss_record, correct, total)
            else:
                print("No good {}".format(predict_ratio))
        return step

    def print_stats(self, step, loss, correct, total):
        print("Step = {} Prediction = {}/{} Error = {}".format(step, correct, total, loss))

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2
        self.size_limit = 1000000
        self.test_limit = 0.95

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10
        print("Start data loading...")
        train_dataset = torchvision.datasets.MNIST(
            './../data/data_mnist', train=True, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
        test_dataset = torchvision.datasets.MNIST(
            './../data/data_mnist', train=False, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
        self.train_data = next(iter(trainLoader))
        self.test_data = next(iter(test_loader))
        print("Data loaded")

    def select_data(self, data, digits):
        data, target = data
        mask = target == -1
        for digit in digits:
            mask |= target == digit
        indices = torch.arange(0,mask.size(0))[mask].long()
        return (torch.index_select(data, dim=0, index=indices), target[mask])

    def create_case_data(self, case):
        if case == 1:
            digits = [0,1]
        elif case == 2:
            digits = [8, 9]
        else:
            digits = [i for i in range(10)]

        description = "Digits: "
        for ind, i in enumerate(digits):
            if ind > 0:
                description += ","
            description += str(i)
        train_data = self.select_data(self.train_data, digits)
        test_data = self.select_data(self.test_data, digits)
        return sm.CaseData(case, Limits(), train_data, test_data).set_description(description).set_output_size(10)

class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()


# If you want to run specific case, put number here
BestSol = { "time"  :10000,
            "steps" :10000,
           "LR"     :0,
           "HS"     :0,
           "ratio"  :0
           }
DoGridSearch = False
if DoGridSearch:
    MyCaseNumber = 7
else:
    MyCaseNumber = -1
sm.SolutionManager(Config()).run(MyCaseNumber)
if DoGridSearch:
    print("Best parameters: LR = {}, HS = {}, solves in {} s and with {} ratio".format(BestSol["LR"],BestSol["HS"],BestSol["time"],BestSol["ratio"]))
