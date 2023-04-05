import numpy as np
from stock_market_simulation import StockMarketSimulationEnvironment
from trader_ai import DeepQTrader
import http.server
import json
import random

class RequestHandler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):

        # trigger agent
        action, reward = agent.act()

        # Send response status code
        self.send_response(200)

        # Send headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-type','application/json')
        self.end_headers()

        # Send data back to client
        message = json.dumps(DataView(agent.state.tolist(), action, (env.trade_entry.item() if env.trade_entry != None else -1), reward.item(), agent.total_reward).__dict__)
        print(action, env.trade_entry, reward)
        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))
        return

def sine(x):
    return 1 + np.sin(2 * np.pi * x / 24)

def composite_sine(x):
    val = (1 + np.sin(2 * np.pi * x / 24)) * (random.random() * 0.1 + 0.95)
    val += (1 + np.sin(2 * np.pi * x / (7*24))) * (random.random() * 0.1 + 0.95) * 5
    return val

def test(x):
    return [0,1,2,3,4,5,4,3,2,1][x % 10]

class DataView():
    def __init__(self, stocks, action, entry, reward, total_reward):
        self.stocks = stocks
        self.action = action
        self.reward = reward
        self.entry = entry
        self.total_reward = total_reward

env = StockMarketSimulationEnvironment(80, sine)
agent = DeepQTrader(env, 10)

# Training
for i in range(0, 100000):
    print(i)
    agent.act()

agent.save_model("test.pt")
agent.epsilon = 0
agent.total_reward = 0

print("Done training")

# Create server
server = http.server.HTTPServer(('localhost', 8000), RequestHandler)

# Start listening for requests
server.serve_forever()


