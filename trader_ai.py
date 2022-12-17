import numpy as np
import torch
from agent import Agent
import random

# This class represents an deep reinforcement trained stock trader.
# It fixes the stop-loss and take-profit and learns when to enter the market and when to leave
class DeepQTrader(Agent):

	def __init__(self, env, num_input_variables, num_hidden_variables, epsilon=0.25, batch_size=16, learning_rate=0.01, gamma=0.9, threshold=0.5, train_agent=True):
		# Stock state tracking parameters
		self.batch_size = batch_size
		self.num_input_variables = num_input_variables
		self.counter = 0
		self.threshold = threshold
		self.gamma = gamma
		self.learning_rate = learning_rate
		self.state_batch = torch.zeros(size=(batch_size, num_input_variables))
		self.rewards = torch.zeros(batch_size)
		self.trade_closed = torch.zeros(batch_size)
		self.tp = 0
		self.sl = 0
		self.train_agent = train_agent
		self.state = env.stocks
		self.env = env
		self.q_values = torch.zeros(batch_size)
		self.future_rewards = torch.zeros(batch_size)
		self.in_trade = False
		self.total_reward = 0
		self.softmax = torch.nn.Softmax(dim=1)
		self.other_softmax = torch.nn.Softmax(dim=0)
		self.epsilon = epsilon

		# Qlearning DNN
		self.model = self.create_model(num_input_variables, num_hidden_variables)
		self.loss_function = torch.nn.MSELoss()
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

	# Use the gathered batch samples to perform a Qlearning update
	# Loss = MSE(F(state_batch), r + y*Q_max_a(s',a')) 
	def update_model_parameters(self):
		pred_y = self.softmax(self.model(self.state_batch))
		loss = self.loss_function(torch.max(pred_y, dim=1).values, self.rewards + self.future_rewards)
		self.model.zero_grad()
		loss.backward()
		self.optimizer.step()
		
	# Reset the batch parameters
	def reset(self):
		self.state_batch *= 0
		self.counter = 0

	# Buffer (last 10 samples of) the state, and other input variables
	def create_training_sample(self):
		self.state_batch[self.counter] = torch.cat((self.state[-(self.num_input_variables - 1):],torch.tensor(int(self.in_trade)).unsqueeze(0)))

	# Actions [Buy:0; Do_nothing:1; Sell:2]
	# Check if TP/SL is hit, otherwise choose action based on max Q value of possible actions with probabilty 1-epsilon
	# We cannot buy if in a trade, and cannot sell if not in a trade
	def policy(self):
		if self.in_trade and (self.state[self.state.shape[0] - 1] > self.tp or self.state[self.state.shape[0] - 1] < self.sl):
			action = 2
		else:
			with torch.no_grad():
				q_values = self.other_softmax(self.model(self.state_batch[self.counter]))
			if self.in_trade:
				q_values[0] = 0
			elif not self.in_trade:
				q_values[2] = 0

			# take the action with max q value with probability 1-epsilon
			sorted_q_values, sorted_q_indices = q_values.sort(descending=True)
			if random.random() > self.epsilon:
				action = sorted_q_indices[0].item()
			else:
				action = sorted_q_indices[1].item()

		# When buying set TP/SL for trade
		if action == 0:
			self.calculate_tp()
			self.calculate_sl()
			self.in_trade = True

		# When selling, were no longer in a trade
		if action == 2:
			self.in_trade = False

		return action

	# calculate action -> 
	# pass action to environment -> 
	# store rewards and expected future rewards in batch -> 
	# train when buffer is full
	def act(self):
		self.create_training_sample()
		action = self.policy()

		# Perform action and receive reward and new state
		(reward, new_state) = self.env.next(self, action)
		self.state = new_state
		self.rewards[self.counter] = reward
		self.total_reward += reward

		# Store expected future reward
		with torch.no_grad():
			next_maxq = torch.max(self.other_softmax(self.model(self.state_batch[self.counter])))
		self.future_rewards[self.counter] = self.gamma * next_maxq

		# Log everything
		#self.log(self.state, action, reward, next_maxq)

		# Increment counter
		self.counter += 1		

		# If buffer is full, perform learning update and clean buffer
		if self.counter == self.batch_size and self.train_agent:
			self.update_model_parameters()
			self.reset()

		return action

	# Calculate take-profit, i.e. the price we consider high enough to exit the trade
	def calculate_tp(self):
		#return self.current_value(self.state) + (torch.mean(self.state) - torch.max(self.state))
		self.tp = self.state[self.state.shape[0] - 1] + 0.3

	# Calculate stop-loss, i.e. the price we consider low enough to exit the trade
	def calculate_sl(self):
		#return self.current_value(self.state) - (torch.mean(self.state) - torch.max(self.state))
		self.sl = self.state[self.state.shape[0] - 1] - 0.3

	# maintain current prediction
	def current_value(self):
		return self.state[self.state.shape[0] - 1]

	# Construct DNN
	def create_model(self, num_input_variables, num_hidden_variables):
		model = torch.nn.Sequential(torch.nn.Linear(num_input_variables, num_hidden_variables),
                      torch.nn.ReLU(),
                      torch.nn.Linear(num_hidden_variables, 3),
                      torch.nn.Sigmoid())
		return model

	def log(self, state, action, reward, next_maxq):
		print("state: {0}\naction: {1}\nreward: {2}\n qmax(s',a'): {3}\n\n".format(state, action, reward, next_maxq))

	def load_model(self, path):
		self.model.load_state_dict(torch.load(path))

	def save_model(self, path):
		torch.save(self.model.state_dict(), path)

