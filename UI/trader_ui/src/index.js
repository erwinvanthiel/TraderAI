import React, { useState } from 'react';
import ReactDOM from 'react-dom/client';
import fetch from 'isomorphic-fetch';
import Plot from 'react-plotly.js';
import './index.css'

function mapNumberToAction(number) {
  switch (number) {
    case 0:
      return 'Buy';
    case 1:
      return 'Wait';
    case 2:
      return 'Sell';
    default:
      return 'Invalid';
  }
}

function MyComponent() {
  const [data, setData] = useState([]);

  const handleClick = () => {
    fetch('http://localhost:8000')
      .then(response => response.json())
      .then(response => setData(response));
      console.log(data)
  }

  return (
    <div>
      { data.stocks != undefined ? 
        <Plot
  className="plotje"
  data={[
    {
      x: data.stocks.map((element, index) => index),
      y: data.stocks,
      type: 'bar',
      marker: {
        color: "black",
      },
    },
    {
      x: data.stocks.map((element, index) => index),
      y: data.stocks.map((element, index) => data.stocks.length - 2 === index ? data.stocks[index] : 0),
      type: 'bar',
      marker: {
        color: data.reward === 0 ? 'black' : (data.reward >= 0 ? 'green' : 'red'),
        opacity: 0.8,
      },
    }

  ]}
  layout={{
        barmode: 'overlay' // set barmode to 'overlay' for overlapping bars
      }}
/> : null
      }
      { data.reward != undefined ? <p>Entry: {data.entry}</p> : null }
      { data.exit != undefined ? <p>Exit: {data.exit}</p> : null }
      { data.action != undefined ? <p>Action: {mapNumberToAction(data.action)}</p> : null }
      { data.reward != undefined ? <p>Reward: {data.reward}</p> : null }
      { data.total_reward != undefined ? <p>Accumulated reward: {data.total_reward}</p> : null }
    <button onClick={handleClick}>Next</button>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <MyComponent />
  </React.StrictMode>
);


