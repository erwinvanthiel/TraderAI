from environment import Environment
import numpy as np
import torch

class StockMarketSimulationEnvironment(Environment):

	def __init__(self, memory_size, stock_simulation_function):
		self.clock = 0
		self.stocks = torch.zeros(memory_size)
		self.stock_simulation_function = stock_simulation_function

	def next(self, agent, action):
		# Increment clock
		self.clock = self.clock + 1

		# Shift all stock values to the left and add latest value
		for index, element in enumerate(self.stocks):
			if index == len(self.stocks)-1:
				continue
			self.stocks[index] = self.stocks[index+1]
		self.stocks[self.stocks.shape[0]-1] = self.stock_simulation_function(self.clock)

		# give the agent its reward and the next state
		return (self.get_reward(agent, action), self.stocks)

	# Buy: 1 - 0 = 1
	# Do nothing: 1 - 1 = 0
	# Sell: 1 - 2 = - 1
	def get_reward(self, agent, action):
		return (self.stocks[self.stocks.shape[0]-1] - self.stocks[self.stocks.shape[0]-2]) * (1 - action)
		
		
		






