# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# TODO to solve harder paths
'''Add another hidden layer to the DQN also using RELU.
Once the car reaches the goal it earns a reward of 2
Decreased punishment from going further away from the goal
Set temperature parameter to 75.
Added a timer for how long it takes for the agent to reach the destination. 
If the agent does not find the destination after 10 seconds it gets a punishment (reward -= 0.3), 
after 20 seconds more punishment (reward -= 0.5) and so on. 
The more time it takes to find the destination the more punishment it gets. 
Added this timer to the list of signals passed to the DQN so it can learn from it:
last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation, self.last_time]'''

# Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__() # use the Pytorch Module inheritance to use its functions later
        self.input_size = input_size # the size of the input vector
        self.nb_action = nb_action # the size of the output action
        # we only use one hidden layer, so we need two full connections for our nn
        # the Linear function just create a vanilla full connection between two layers
        self.fc1 = nn.Linear(input_size, 30) # full connections between the input layer and the hidden layer
        self.fc2 = nn.Linear(30, nb_action) # full connections between the hidden layer and the output layer
    
    def forward(self, state):
    	# state is our inputs
        hidden_neurons_outputs = F.relu(self.fc1(state))
        # q_values are our outputs 
        q_values = self.fc2(hidden_neurons_outputs)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
    	# the capacity is the number of last events the agent memorizes to take actions later
        self.capacity = capacity
        # the memory is the list containing those last events
        # an event is composed of four parts: last state, new state, last action, last reward
        self.memory = []
    
    def push(self, event):
        self.memory.append(event) # add an event in the memory
        if len(self.memory) > self.capacity:
            del self.memory[0] # ensure that we don't have more events in memory than we want
    
    def sample(self, batch_size): # returns a batch of pseudo random events from the memory
    	# to ensure better learning, the events should be as diverse as possible (hence pseudo random)
    	# zip function reshape a list this way: 
    	# list = ((1,2,3),(4,5,6)) => zip(*list) => ((1,4),(2,3),(5,6))
    	# this allow use to create one batch for each of the state, action and reward
    	# from ((stat1,act1,r1),(stat2,act2,r2),(stat3,act3,r3)) to ((states), (actions), (rewards))
    	# because we will later need to put each of these batches with this format in a Pytorch variable
    	# so that each one will get a gradient, and we will be able to differentiate each of them
        samples = zip(*random.sample(self.memory, batch_size))
        # we return the samples into a Pytorch variable with the map() function
        # the lambda function take the samples, concatenate them with respect to the first dimension (which corresponds to the states)
        # this is required so that each row (state, action, reward) corresponds to the same time t, and we know which state it belongs to
        # so each sample (states), (actions), (rewards) has now its state represented in it, this way our samples are ordered
        # so we will return something like this (state), (state), (action), (reward)
        # and then, we convert these tensors into some Torch variables that contains both a tensor and a gradient
        # so that later when we will apply stochastic gradient descent, we will be able to differentiate to update the weights
        return map(lambda sample: Variable(torch.cat(sample, 0)), samples)

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma # the delay coefficient
        self.reward_window = [] # the slide window of the evolving mean of the last rewards
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000) # we give the memory capacity to ReplayMemory()
        '''Adam is a replacement optimization algorithm for stochastic gradient descent for training 
        deep learning models. Adam combines the best properties of the AdaGrad and RMSProp algorithms to 
        provide an optimization algorithm that can handle sparse gradients on noisy problems'''
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        # last_state is a vector of 5 demensions
        # but for Pytorch, it needs to be a Torch Tensor with one more dimension corresponding to the batch
        # this dimension should be the first one in the tensor
        # unsqueeze(0) creates the new dimension at the first index 0 for the batch
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
    	# softmax chooses the action to do by considering the probabilities of each actions
    	# the likelihood of selecting one action is directly linked to its probability
    	# we need to convert state to a Torch Tensor with Variable()
    	# volatile = True inform Torch that we don't want the gradient associated with the variable
    	# because we don't need it, so removing it will improve performance
    	# and we multiply it by the temperature parameter
    	# it allows us to modulate how the nn will be sure of which action it will play
    	# the higher the number is, the more sure the nn is to choose the action
    	# the higher the temperature is, the higher the probability of selecting the best q value is
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        # the multinomial() will give us a random draw from the probs distribution
        action = probs.multinomial()
        # as action is a Pytorch variable with its first dimension corresponding to the batch,
        # we need to access the second dimension to get the action 
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
    	# retrieve only the chosen action from the outputs with .gather(1, batch_action.unsqueeze(1)).squeeze(1)
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # we choose the maximum of all the Qvalues with .detach().max(1)[0] as the formula wants
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        # temporal difference loss => smooth_l1_loss(prediction, target)
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad() # with Pytorch we must reinitialize the optimizer for each time
        td_loss.backward(retain_variables = True) # backpropagation, retain_variables = True frees memory
        self.optimizer.step() # update the weights according to the gradient
    
    def update(self, reward, new_signal):
    	# convert the signal to a Torch Tensor with the new dimension for the batch
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # add this event to the memory
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        # if we have enough memory, we can start to learn
        if len(self.memory.memory) > 100:
        	# we get a random sample batch of 100 samples
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        # update
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        # limit the size of the event window
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
    	# return the mean of all the rewards in the event window
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")