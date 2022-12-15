class Environment():

		def __init__(self):
			self.agents = []

		def next(self, clock):
			raise NotImplementedError()

		def add_agent(self, agent):
			self.agents.append(agent)
