import numpy as np
import torch
from agent import Agent

# This class represents an deep reinforcement trained stock trader.
# It fixes the stop-loss and take-profit and learns when to enter the market
class TraderAI(Agent):

	def __init__(self, num_input_variables, num_hidden_variables, batch_size, learning_rate, threshold=0.5, train_agent=True):
		# Stock state tracking parameters
		self.batch_size = batch_size
		self.num_input_variables = num_input_variables
		self.counter = 0
		self.threshold = threshold
		self.learning_rate = learning_rate
		self.state_batch = torch.zeros(size=(batch_size, num_input_variables))
		self.trade_results = torch.zeros(batch_size)
		self.trade_closed = torch.zeros(batch_size)
		self.take_profits = torch.zeros(batch_size)
		self.stop_losses = torch.zeros(batch_size)
		self.train_agent = train_agent
		self.current_predition = None

		# DNN policy
		self.model = self.create_model(num_input_variables, num_hidden_variables)
		self.loss_function = torch.nn.CrossEntropyLoss()
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

	# inference with dnn
	def predict(self, state):
		return self.model(state) > self.threshold

	# Use the gathered batch samples to perform a learning update
	def update_model_parameters(self):
		pred_y = self.model(self.state_batch)
		loss = self.loss_function(pred_y, self.trade_results.long())
		self.model.zero_grad()
		loss.backward()
		self.optimizer.step()

	# Reset the batch parameters
	def reset(self):
		self.state_batch *= 0
		self.trade_results *= 0
		self.trade_closed *= 0
		self.take_profits *= 0
		self.stop_losses *= 0
		self.counter = 0

	# Fill a batch with stock data and update parameters, update trade statusses 
	# When all trades have been closed perform learning update
	def train(self, env_state):
		if self.counter < self.batch_size:
			self.state_batch[self.counter] = env_state
			self.take_profits[self.counter] = self.calculate_tp(env_state)
			self.stop_losses[self.counter] = self.calculate_sl(env_state)
		self.counter += 1
		self.update_trade_results(env_state)
		if torch.sum(self.trade_closed) == self.trade_closed.shape[0]:
			self.update_model_parameters()
			self.reset()

	# Check whether the stock hit the tp/sl and closes trade if so
	def update_trade_results(self, env_state):
		# set trade result if tp or sl was hit, only if trade was still open
		for i in range(min(self.batch_size,self.counter)):
			if self.trade_closed[i] == 0:
				if self.take_profits[i] < env_state[env_state.shape[0]-1]:
					self.trade_results[i] = 1
					self.trade_closed[i] = 1
				if self.stop_losses[i] > env_state[env_state.shape[0]-1]:
					self.trade_results[i] = 0
					self.trade_closed[i] = 1

	# Calculate take-profit, i.e. the price we consider high enough to exit the trade
	def calculate_tp(self, state):
		return self.current_value(state) + (torch.mean(state) - torch.max(state))

	# Calculate stop-loss, i.e. the price we consider low enough to exit the trade
	def calculate_sl(self, state):
		return self.current_value(state) - (torch.mean(state) - torch.max(state))

	# receive the state information from the environment
	def update_state(self, env_state):
		print(env_state)
		self.current_prediction = self.predict(env_state)
		if self.train_agent:
			self.train(env_state)

	# maintain current prediction
	def current_value(self, state):
		return state[state.shape[0] - 1]

	# Construct DNN
	def create_model(self, num_input_variables, num_hidden_variables):
		model = torch.nn.Sequential(torch.nn.Linear(num_input_variables, num_hidden_variables),
                      torch.nn.ReLU(),
                      torch.nn.Linear(num_hidden_variables, 1),
                      torch.nn.Sigmoid())
		return model

	def load_model(self, path):
		self.model.load_state_dict(torch.load(path))

	def save_model(self, path):
		torch.save(self.model.state_dict(), path)

