# You need to learn a function with n inputs.
# For given number of inputs, we will generate random function.
# Your task is to learn it
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
    def __init__(self, input_size, output_size, hidden_size):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        #sm.SolutionManager.print_hint("Hint[1]: Explore more deep neural networks")
        #self.hidden_size = 10
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.linear1bn = nn.BatchNorm1d(self.hidden_size,track_running_stats=False)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear2bn = nn.BatchNorm1d(self.hidden_size,track_running_stats=False)
        self.linear3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3bn = nn.BatchNorm1d(self.hidden_size,track_running_stats=False)
        self.linear4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear4bn = nn.BatchNorm1d(self.hidden_size,track_running_stats=False)
        self.linear5 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear5bn = nn.BatchNorm1d(self.hidden_size,track_running_stats=False)
        self.linear6 = nn.Linear(self.hidden_size, output_size)
        self.linear6bn = nn.BatchNorm1d(output_size,track_running_stats=False)


    def forward(self, x):
        x = self.linear1(x)
        x = nn.ReLU6()(self.linear1bn(x))
        x = self.linear2(x)
        x = nn.ReLU6()(self.linear2bn(x))
        x = self.linear3(x)
        x = nn.ReLU6()(self.linear3bn(x))
        x = self.linear4(x)
        x = nn.ReLU6()(self.linear4bn(x))
        x = self.linear5(x)
        x = nn.ReLU6()(self.linear5bn(x))
        x = self.linear6(x)
        x = F.sigmoid(self.linear6bn(x))
        return x

class Solution():
    def __init__(self):
        self.lr = 0.08
        #self.lr_grid = [0.6,0.65,0.7,0.75,1.0,2.0, 5, 10]
        self.lr_grid = [0.05, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14]
        self.hidden_size = 39
        #self.hidden_size_grid = [40,50,55,58,60,61,62,63,64,65]
        self.hidden_size_grid = [28, 32, 34, 35, 36, 37, 38, 39, 45, 50]
        self.grid_search = GridSearch(self).set_enabled(False)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self.hidden_size)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # Put model in train mode
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum = 0.9, nesterov = True)
        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1:
                break
            data = train_data
            target = train_target
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            #sm.SolutionManager.print_hint("Hint[2]: Explore other activation functions", step)
            output = model(data)
            # if x < 0.5 predict 0 else predict 1
            predict = output.round()
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = target.view(-1).size(0)
            if correct == total:
                print("PASSED - STEP {}".format(step))
                if BestSol["steps"]>step:
                    BestSol["steps"] = step
                    BestSol["LR"] = self.lr
                    BestSol["HS"] = self.hidden_size
                break
            if step > 30 :
                print("FAILED")
                break
            # calculate loss
            loss = nn.BCELoss()(output,target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
            # self.print_stats(step, loss, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
        print("LR = {} HS = {}".format(self.lr,self.hidden_size))
        return step
    
    def print_stats(self, step, loss, correct, total):
        if step % 10 == 0:
            print("Step = {} Prediction = {}/{} Error = {}".format(step, correct, total, loss.item()))

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 10000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, input_size, seed):
        random.seed(seed)
        data_size = 1 << input_size
        data = torch.FloatTensor(data_size, input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):
            for j in range(input_size):
                input_bit = (i>>j)&1
                data[i,j] = float(input_bit)
            target[i] = float(random.randint(0, 1))
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        input_size = min(3+case, 7)
        data, target = self.create_data(input_size, case)
        return sm.CaseData(case, Limits(), (data, target), (data, target)).set_description("{} inputs".format(input_size))


class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
BestSol = {"steps"  :10000,
           "LR"     :0,
           "HS"     :0
           }
sm.SolutionManager(Config()).run(case_number=-1)
#print("Best parameters: LR = {}, HS = {}, solves in {} steps".format(BestSol["LR"],BestSol["HS"],BestSol["steps"]))
