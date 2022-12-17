class Agent():

	def load_model(self, path):
		raise NotImplementedError()

	def policy(self):
		raise NotImplementedError()

	def act(self):
		raise NotImplementedError()

	def save_model(self, path):
		raise NotImplementedError()