class Agent():

	def __init__(self, input_variables):
		self.input_variables = input_variables

	def load_model(self, path):
		raise NotImplementedError()

	def predict(self, state):
		raise NotImplementedError()

	def update_state(self, env_state):
		raise NotImplementedError()

	def save_model(self, path):
		raise NotImplementedError()