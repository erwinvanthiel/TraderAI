import numpy as np
from stock_market_simulation import StockMarketSimulationEnvironment
from trader_ai import TraderAI
import http.server


env = StockMarketSimulationEnvironment(100, np.sin)
agent = TraderAI(4, 2, 3, 0.01, True)
env.add_agent(agent)

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):

        # Update environemnt
        env.next()

        # Send response status code
        self.send_response(200)

        # Send headers
        self.send_header('Content-type','application/json')
        self.end_headers()
        # Send message back to client
        message = (str)(agent.state_batch)

        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))
        return

# Create server
server = http.server.HTTPServer(('localhost', 8000), RequestHandler)

# Start listening for requests
server.serve_forever()


