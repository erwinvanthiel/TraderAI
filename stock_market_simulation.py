from environment import Environment
import numpy as np
import torch

class StockMarketSimulationEnvironment(Environment):

	def __init__(self, memory_size, stock_simulation_function):
		super().__init__()
		self.clock = 0
		self.stocks = torch.zeros(memory_size)
		self.stock_simulation_function = stock_simulation_function

	def next(self):
		# Increment clock
		self.clock = self.clock + 1

		# Shift all stock values to the left and add latest value
		for index, element in enumerate(self.stocks):
			if index == len(self.stocks)-1:
				continue
			self.stocks[index] = self.stocks[index+1]
		self.stocks[self.stocks.shape[0]-1] = self.stock_simulation_function(self.clock)

	    # train agents
		for agent in self.agents:
			agent.update_state(self.stocks[-agent.num_input_variables:])



