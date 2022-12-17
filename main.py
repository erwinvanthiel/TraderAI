import numpy as np
from stock_market_simulation import StockMarketSimulationEnvironment
from trader_ai import DeepQTrader
import http.server

def simulation(x):
    return np.sin(2 * np.pi * x / 24)

env = StockMarketSimulationEnvironment(100, simulation)
agent = DeepQTrader(env, 10, 4)

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):

        # trigger agent
        action = agent.act()

        # Send response status code
        self.send_response(200)

        # Send headers
        self.send_header('Content-type','application/json')
        self.end_headers()

        # Send message back to client
        message = str(action) + str(agent.in_trade)

        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))
        return

# Create server
server = http.server.HTTPServer(('localhost', 8000), RequestHandler)

# Start listening for requests
server.serve_forever()


