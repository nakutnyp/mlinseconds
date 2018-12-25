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
        RunningStats = False
        self.linears = nn.ModuleList([nn.Linear(self.input_size if i == 0 else self.hidden_size, self.hidden_size if i != self.solution.layers_number -1 else self.output_size) for i in range(self.solution.layers_number)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(self.hidden_size if i != self.solution.layers_number-1 else self.output_size, track_running_stats=RunningStats) for i in range(self.solution.layers_number)])
        for i in range(self.solution.layers_number):
            nn.init.xavier_uniform_(self.linears[i].weight)

    def forward(self, x):
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            if self.solution.do_batch_norm:
                x = self.batch_norms[i](x)
            act_function = self.solution.activation_output if i == len(self.linears)-1 else self.solution.activation_hidden
            x = self.solution.activations[act_function](x)
        return(x)
        
class Solution():
    def __init__(self):
        # best so far - lr 1.5, hs 15, layers 3
        self = self
        self.layers_number = 3
        self.lr = 1.5
        self.hidden_size = 15
        self.momentum = 0.8
        self.activation_hidden = 'relu6'
        self.activation_output = 'sigmoid'
        self.do_batch_norm = True
        self.lr_grid = [1.25,1.27, 1.29, 1.3,1.31, 1.32, 1.34]
        self.hidden_size_grid = [12,13,14,15,16,17]
        #self.momentum_grid = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.activations = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'relu6': nn.ReLU6(),
            'rrelu0103': nn.RReLU(0.1, 0.3),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'leakyrelu01': nn.LeakyReLU(0.1)
            }
        #self.do_batch_norm_grid = [False, True]
        #self.activation_hidden_grid = self.activations.keys()
        #self.activation_output_grid = self.activations.keys()
        self.grid_search = GridSearch(self).set_enabled(DoGridSearch)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # Put model in train mode
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum = self.momentum, weight_decay = 0.000)
        if DoGridSearch:
            print("LR = {}, HS = {}, Layers = {}, HA = {}, OA = {}, Momentum = {}".format(self.lr, self.hidden_size, self.layers_number, self.activation_hidden, self.activation_output,self.momentum))
        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.01:
                print("Time run out")
                break
            data = train_data
            target = train_target
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(data)
            # if x < 0.5 predict 0 else predict 1
            predict = output.round()
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = target.view(-1).size(0)
            # calculate loss
            #loss = ((output-target)**2).sum()
            loss = nn.BCELoss()(output,target)
            if correct == total and loss.item() < 0.015:  # float(diff) < 0.3: 
                if DoGridSearch:
                    time_left = context.get_timer().get_time_left()
                    time_solved = 2-time_left
                    self.print_stats(step, loss, correct, total, diff)
                    print("PASSED - Solved in {} s".format(time_solved))
                    if BestSol["time"] > time_solved:
                        BestSol["time"] = time_solved
                        BestSol["steps"] = step
                        BestSol["LR"] = self.lr
                        BestSol["HS"] = self.hidden_size
                print("Diff: ",diff)
                break
            #if step > 1000 :
            #    print("Too many steps")
            #    break
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
            if DoGridSearch and step % 40 == 0:
                self.print_stats(step, loss, correct, total, diff)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
            
        if DoGridSearch:
            model.eval()
            data, target = context.get_case_data().train_data
            output = model(data)
            predict = output.round()
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            total = target.view(-1).size(0)
            loss = nn.BCELoss()(output,target)
            diff = (output.data-target.data).abs().max()
            print("TRAIN DATA - prediction: {}/{}, error: {}, diff: {}".format(correct,total, loss.item(), diff))
            data, target = context.get_case_data().test_data
            output = model(data)
            predict = output.round()
            correct = predict.eq(target.view_as(predict)).long().sum().item()      
            total = target.view(-1).size(0)
            loss = nn.BCELoss()(output,target)
            diff = (output.data-target.data).abs().max()
            print("TEST DATA - prediction: {}/{}, error: {}, diff: {}".format(correct,total, loss.item(), diff))
            print("------------------------")
   
        return step

    def print_stats(self, step, loss, correct, total, diff):
        print("Step = {} Prediction = {}/{} Error = {} Diff = {}".format(step, correct, total, round(float(loss.item()),4), round(float(diff),3)))

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 1000000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, data_size, input_size, random_input_size, seed):
        random.seed(seed)
        function_size = 1 << input_size
        function_table = [random.randint(0, 1) for _ in range(function_size)]
        total_input_size = input_size+random_input_size
        input_bit_indexes = {x.item():(1<<i) for i,x in enumerate(torch.randperm(total_input_size)[:input_size])}
        data = torch.FloatTensor(data_size, total_input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):
            fun_ind = i%len(function_table)
            fun_value = function_table[fun_ind]
            for j in range(total_input_size):
                input_bit = random.randint(0, 1)
                if j in input_bit_indexes:
                    input_bit = fun_ind&1
                    fun_ind = fun_ind >> 1
                data[i,j] = float(input_bit)
            target[i] = float(fun_value)
        perm = torch.randperm(data.size(0))
        data = data[perm]
        target = target[perm]
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        data_size = 256*32
        input_size = 8
        random_input_size = min(32, (case-1)*4)

        data, target = self.create_data(2*data_size, input_size, random_input_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs and {} random inputs".format(input_size, random_input_size))

class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
BestSol = { "time"  :10000,
            "steps" :10000,
           "LR"     :0,
           "HS"     :0
           }
DoGridSearch = False
if DoGridSearch:
    MyCaseNumber = 10
else:
    MyCaseNumber = -1
sm.SolutionManager(Config()).run(MyCaseNumber)
if DoGridSearch:
    print("Best parameters: LR = {}, HS = {}, solves in {} s and {} steps".format(BestSol["LR"],BestSol["HS"],BestSol["time"],BestSol["steps"]))
