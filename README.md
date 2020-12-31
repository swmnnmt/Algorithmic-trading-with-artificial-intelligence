# Algorithmic trading with artificial intelligence

## If a human investor can be successful, why can't a machine?

I would just like to add a disclaimer — this project is entirely intended for research purposes! I’m just a student
learning about deep learning, and the project is a work in progress, please don’t put any money into it!

Algorithmic trading has revolutionised the stock market and its surrounding industry. Over 70% of all trades happening
in the US right now are being handled by bots. Gone are the days of the packed stock exchange with suited people waving
sheets of paper shouting into telephones.

This got me thinking of how I could develop my own algorithm for trading stocks, or at least try to accurately predict
them.

I’ve learned a lot about neural networks and machine learning over the summer and one of the most recent and applicable
ML technologies I learnt about is the LSTM cell.

### The Dataset

The good thing about stock price history is that it’s basically a well labelled pre formed dataset. After some googling
I found a service called pytse client. They offered the daily price history of Tehran Stock Exchange for the past 19
years. This included the open, high, low, close, volume,etc of trades for each day, from today all the way back up to
1380s. Extract your CSV file in `/data` folder which has Open Price, High Price, Low Price, Volume, Close Price columns.

### The Algorithm
Armed with an okay-ish stock prediction algorithm I thought of a naive way of creating a bot to decide to buy/sell 
a stock today given the stock’s history. In essence, you just predict the opening value of the stock for the next day, 
and if it is beyond a threshold amount you buy the stock. If it is below another threshold amount, sell the stock.
This dead simple algorithm actually seemed to work quite well — visually at least.

### How to run it:

#### Anaconda:

`git clone https://github.com/swmnnmt/Algorithmic-trading-with-artificial-intelligence.git`

`cd Algorithmic-trading-with-artificial-intelligence`

`conda create --name venv python=3.8`

`conda activate venv`

`conda install --file requirements.txt`

`python BakeOff.py`

#### Pip:

`git clone https://github.com/swmnnmt/Algorithmic-trading-with-artificial-intelligence.git`

`cd Algorithmic-trading-with-artificial-intelligence`

`python3 -m venv venv`

`venv\Scripts\activate.bat` or `source venv/bin/activate`

`pip install -r requirements.txt`

`python BakeOff.py`

For more information about how I implement it,
see https://www.kaggle.com/alirezanematolahy/tehran-stock-prediction-using-lstm-model

Be Happy, that's all we can do :)