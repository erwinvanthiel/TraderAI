import numpy as np
import torch
from agent import Agent
import random

# This class represents an deep reinforcement trained stock trader.
# It fixes the stop-loss and take-profit and learns when to enter the market and when to leave
class DeepQTrader(Agent):

	def __init__(self, env, num_input_variables, epsilon=1, batch_size=64, learning_rate=0.005, gamma=0.95, train_agent=True):
		# Stock state tracking parameters
		self.batch_size = batch_size
		self.num_input_variables = num_input_variables
		self.counter = 0
		self.gamma = gamma
		self.learning_rate = learning_rate
		self.state_batch = torch.zeros(size=(batch_size, num_input_variables))
		self.next_state_batch = torch.zeros(size=(batch_size, num_input_variables))
		self.action_batch = torch.zeros(batch_size)
		self.rewards = torch.zeros(batch_size)
		self.trade_closed = torch.zeros(batch_size)
		self.tp = 0
		self.sl = 0
		self.train_agent = train_agent
		self.state = env.stocks
		self.env = env
		self.q_next_values = torch.zeros(batch_size)
		self.in_trade = False
		self.total_reward = 0
		self.epsilon = epsilon
		self.trade_state = torch.zeros(batch_size)

		# Qlearning DNN
		self.model = self.create_model(num_input_variables)
		self.loss_function = torch.nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

	# Use the gathered batch samples to perform a Qlearning update
	# Loss = MSE(F(state_batch), r + y*Q_max_a(s',a')) 
	def update_model_parameters(self):
		# DEBUG
		# with torch.no_grad(): 
		# 	print(self.model(torch.tensor([0,1,2,3,4,1]).float()))
		self.model.zero_grad()
		q_pred = self.model(self.state_batch.unsqueeze(1)).squeeze()[torch.arange(self.batch_size), self.action_batch.long()]
		self.q_next_values[torch.where(self.action_batch == 2)] = 0
		q_target = self.rewards + self.gamma * self.q_next_values

		loss = self.loss_function(q_pred, q_target)
		print(loss)
		loss.backward()
		self.optimizer.step()
		self.epsilon = self.epsilon * 0.999
		
	# Reset the batch clock
	def reset(self):
		self.counter = 0

	# Actions [Buy:0; Do_nothing:1; Sell:2]
	# Check if TP/SL is hit, otherwise choose action based on max Q value of possible actions with probabilty 1-epsilon
	# We cannot buy if in a trade, and cannot sell if not in a trade
	def policy(self):
		# Manage exit conditions
		if self.in_trade and (self.current_value() > self.tp or self.current_value() < self.sl):
			action = 2

		with torch.no_grad():
			q_values = self.model(self.state_batch[self.counter].unsqueeze(0).unsqueeze(1)).squeeze()

		# Ensure we only buy when not in a trade and sell when in a trade
		if self.in_trade:
			q_values[0] = -1000
		elif not self.in_trade:
			q_values[2] = -1000

		# take the action with max q value with probability 1-epsilon
		_, sorted_q_indices = q_values.sort(descending=True)
		if random.random() > self.epsilon:
			action = sorted_q_indices[0].item()
		else:
			action = sorted_q_indices[random.randint(0,1)].item()
		
		# When buying set TP/SL for trade
		if action == 0:
			self.calculate_tp()
			self.calculate_sl()
			self.in_trade = True
			self.trade_state[self.counter] = 1

		# When selling, were no longer in a trade
		if action == 2:
			self.in_trade = False

		return action

	# calculate action -> 
	# pass action to environment -> 
	# store rewards and expected future rewards in batch -> 
	# train when buffer is full
	def act(self):
		# Buffer (last n samples of) the state
		self.state_batch[self.counter] = torch.cat((self.state.clone()[1-self.num_input_variables:], torch.tensor(int(self.in_trade)).unsqueeze(0)))
		action = self.policy()
		self.action_batch[self.counter] = action

		# Perform action and receive reward and new state
		(reward, new_state) = self.env.next(action)
		self.state = new_state
		self.next_state_batch[self.counter] = torch.cat((self.state.clone()[1-self.num_input_variables:], torch.tensor(int(self.in_trade)).unsqueeze(0)))
		self.rewards[self.counter] = reward

		# Store next q values
		with torch.no_grad():
			next_q_values = self.model(self.next_state_batch[self.counter].unsqueeze(0).unsqueeze(1)).squeeze()
			next_q_values[2-int(self.in_trade)*2] = -1000 # prevent buying when in trade
			self.q_next_values[self.counter] = torch.max(next_q_values)

		self.total_reward += reward.item()

		# Log everything
		# self.log(action, reward)

		# Increment counter
		self.counter += 1		

		# If buffer is full, perform learning update and clean buffer
		if self.counter == self.batch_size:
			if self.train_agent:
				self.update_model_parameters()
			self.reset()

		return action, reward

	# Calculate take-profit, i.e. the price we consider high enough to exit the trade
	def calculate_tp(self):
		self.tp = self.current_value() + 2

	# Calculate stop-loss, i.e. the price we consider low enough to exit the trade
	def calculate_sl(self):
		#return self.current_value(self.state) - (torch.mean(self.state) - torch.max(self.state))
		self.sl = self.current_value() - 1.5

	# maintain current prediction
	def current_value(self):
		return self.state[self.state.shape[0] - 1].clone()

	# Construct DNN
	def create_model(self, number_of_inputs):
		model = torch.nn.Sequential(
					  torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
					  torch.nn.ReLU(),
					  torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
					  torch.nn.ReLU(),
					  torch.nn.Flatten(),
					  torch.nn.Linear(64*(number_of_inputs-4), 128),
				      torch.nn.ReLU(),
			          torch.nn.Linear(128,3)
					  )
		return model

	def log(self, action, reward):
		print("action: {0}\nreward: {1}\n\n".format(action, reward))

	def load_model(self, path):
		self.model.load_state_dict(torch.load(path))

	def save_model(self, path):
		torch.save(self.model.state_dict(), path)

