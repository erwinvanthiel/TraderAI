import numpy as np
import torch
from agent import Agent


class DeepQTrader(Agent):
    def __init__(self, num_input_variables, num_hidden_variables, batch_size, learning_rate, threshold=0.5, train_agent=True):
        # Stock state tracking parameters
        self.batch_size = batch_size
        self.num_input_variables = num_input_variables
        self.counter = 0
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.state_batch = torch.zeros(size=(batch_size, num_input_variables))
        self.rewards = torch.zeros(batch_size)
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
        return torch.nn.Sigmoid(self.model(state)) > self.threshold

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
        self.update_model_parameters()
        self.reset()

    # receive the state information from the environment
    def update_state(self, env_state):
        self.current_prediction = self.predict(env_state)
        if self.train_agent:
            self.train(env_state)

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
