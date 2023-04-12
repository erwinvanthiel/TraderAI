from environment import Environment
import numpy as np
import torch

class StockMarketSimulationEnvironment(Environment):

	def __init__(self, memory_size, stock_simulation_function):
		self.clock = 0
		self.stocks = torch.zeros(memory_size)
		self.stock_simulation_function = stock_simulation_function
		self.trade_entry = None

	def next(self, action):
		reward = torch.zeros(1)
		if action == 0:
			self.trade_entry = self.get_current_price()
		if action == 2:
			reward = self.get_reward()

		# Increment clock
		self.clock = self.clock + 1

		# Update stock price
		for index, element in enumerate(self.stocks):
			if index == len(self.stocks)-1:
				continue
			self.stocks[index] = self.stocks[index+1]
		self.stocks[self.stocks.shape[0]-1] = self.stock_simulation_function(self.clock)

		# if action == 0:
		# 	reward += torch.tensor(0.1) if self.stocks[self.stocks.shape[0]-1].item() > self.stocks[self.stocks.shape[0]-2].item() else torch.tensor(-0.1)
		#
		# if action == 2:
		# 	reward += torch.tensor(-0.05) if self.stocks[self.stocks.shape[0]-1].item() > self.stocks[self.stocks.shape[0]-2].item() else torch.tensor(0.05)

		# give the agent its reward and the next state
		return (reward, self.stocks)

	# Buy: 1
	# Do nothing: 1 
	# Sell: 2
	# When agent sells, reward is the profit/loss + a bonus for exiting/entering at the right moment
	def get_reward(self):
		return (self.get_current_price() - self.trade_entry)
		
	def get_current_price(self):
		return self.stocks.clone()[self.stocks.shape[0]-1] 
		






