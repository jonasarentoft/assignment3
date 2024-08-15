import numpy as np
from fastapi import FastAPI, Query
import pickle

# This strategy utilizes the XGBoost Classifier (fitted in the other file) to try and predict if a 
#buy signal when the OSV exceeds a certain threshold will be profitable

app = FastAPI()

class TradingStrategy:
    def __init__(self):
        self.prev_dc_events = []
        self.prev_os_events = []
        self.prev_profits = []
        self.prev_open_events = []
        self.prev_take_profit_events = []
        self.prev_stop_loss_events = []
        self.prev_prices = []
        
        self.dc_events = []
        self.os_events = []
        self.profits = []
        self.open_events = []
        self.take_profit_events = []
        self.stop_loss_events = []
        self.prices = []
        
        self.opens = []
        self.closes = []
        
        self.first_trade_detected = False
        self.first_trade_outcome = None
        self.second_trade_detected = False
        self.second_trade_outcome = None
        
        self.threshold = -0.7
        self.theta = 0.01
        
        self.extreme_price = None
        self.signal = None
        self.trend = None
        self.theta = None
        self.last_low = None
        self.last_high = None 
        self.open_price = None
        self.has_open_position = False
        self.open_price = None
        
        self.counter = 0

    def detect_trend(self, tickprice):
        trend = None
        if (self.extreme_price * (1 + self.theta)) <= tickprice:
            trend = 'up'
        elif (self.extreme_price * (1 - self.theta)) >= tickprice:
            trend = 'down'
        return trend
    
    def increment_counter(self):
        self.counter += 1
        
# Initialize strategy
strategy = TradingStrategy()

# Load model (this is saves from the other file)
FILENAME = 'xgboost_classifier.pkl'
XGBModel = pickle.load(open(FILENAME, "rb"))



@app.get("/signal/{tickprice}/theta")
async def read_price(tickprice, theta = Query(...)):
    if not isinstance(tickprice, float):
        tickprice = float(tickprice.replace(',', '.'))
    
    if not isinstance(theta, float):
        theta = float(theta.replace(',', '.'))
    
    if not strategy.theta:
        strategy.theta = theta
    
    strategy.signal = 'hold'
    strategy.prices.append([strategy.counter, tickprice])

    if not strategy.trend:
        if not strategy.extreme_price:
            strategy.extreme_price = tickprice
            strategy.last_high = [0, tickprice]
            strategy.last_low = [0, tickprice]
            strategy.dc_events.append([0, tickprice])
            #print(f'First price is {tickprice}')
            
        strategy.trend = strategy.detect_trend(tickprice)
        
        if strategy.trend:
            #print(f'First trend is {strategy.trend} @ {tickprice}')
            strategy.dc_events.append([strategy.counter, tickprice])
    
    elif strategy.trend == 'down':
        if strategy.last_low[1] > tickprice:
            strategy.last_low = [strategy.counter, tickprice]
            
        p_ext = strategy.dc_events[-1][1]
        p_dcc = p_ext * (1 - strategy.theta)
    
        osv = ((tickprice - p_dcc) / p_dcc) / strategy.theta
        
        if ((osv <= strategy.threshold) and (not strategy.has_open_position)):
            strategy.has_open_position = True
            #print(f'Position opened @ {tickprice}')
            strategy.open_price = tickprice
            strategy.open_events.append([strategy.counter, strategy.open_price])    
            
            if strategy.second_trade_detected:
                SMA_30 = np.mean(strategy.prices[-60:], axis = 0)[1]
                SMA_5 = np.mean(strategy.prices[-5:], axis = 0)[1]
                
                time_since_dc = strategy.counter - strategy.dc_events[-1][0]
                
                time_between_os_and_dc = strategy.dc_events[-1][0] - strategy.os_events[-1][0]
                
                
                
                # Here, we 'create' the input vector to the XGBoost Classfier
                X = np.array([SMA_30, SMA_5, time_since_dc, time_between_os_and_dc, strategy.second_trade_outcome, strategy.first_trade_outcome])
                
                #print(X)
                
                # We make the XGBoost Clasffier predict whether a buy signal will result in profit
                # If 1, the strategy buys. If 0, nothing happens
                signal = int(XGBModel.predict(X.reshape(1, -1))[0])
                #print(signal)
                #print(type(signal))
                
                
                if signal == 1:
                    strategy.opens.append([strategy.counter, tickprice])
                    strategy.signal = 'buy'


        if tickprice >= strategy.last_low[1] * (1 + strategy.theta):
            #print(f'Trend changed to up')
            strategy.trend = 'up'
            strategy.dc_events.append([strategy.counter, tickprice])
            strategy.os_events.append(strategy.last_low)
            strategy.last_high = [strategy.counter, tickprice]
            if strategy.has_open_position:
                strategy.has_open_position = False
                
                if not strategy.second_trade_detected:
                    if not strategy.first_trade_detected:
                        strategy.first_trade_detected = True
                        
                        if tickprice > strategy.open_price:
                            strategy.first_trade_outcome = 1
                            
                        else:
                            strategy.first_trade_outcome = 0
                        
                    else:
                        strategy.second_trade_detected = True
                        
                        if tickprice > strategy.open_price:
                            strategy.second_trade_outcome = 1
                        
                        else:
                            strategy.second_trade_outcome = 0
                else:
                    strategy.second_trade_outcome = strategy.first_trade_outcome
                    strategy.signal = 'sell'
                    print(f'her')
                    strategy.has_open_position = False
                    strategy.closes.append([strategy.counter, tickprice])
                    if tickprice > strategy.open_price:
                        strategy.first_trade_outcome = 1
                            
                    else:
                        strategy.first_trade_outcome = 0
                    
                    
                if tickprice > strategy.open_price:
                    strategy.take_profit_events.append([strategy.counter, tickprice])
                    #print(f'Took profit @ {tickprice}')
                else:
                    strategy.stop_loss_events.append([strategy.counter, tickprice])
                    #print(f'Stopped Loss @ {tickprice}')
            
            
            
    
    elif strategy.trend == 'up':
        if strategy.last_high[1] < tickprice:
            strategy.last_high = [strategy.counter, tickprice]
            
        if tickprice <= strategy.last_high[1] * (1 - strategy.theta):
            strategy.dc_events.append([strategy.counter, tickprice])
            strategy.os_events.append(strategy.last_high)
            strategy.trend = 'down'
            strategy.last_low = [strategy.counter, tickprice]
            
            
    strategy.increment_counter()
        
    return {"tradeSignal": strategy.signal}


@app.get("/get_data")
async def read_data():
    return {
        "opens": strategy.opens,
        "closes": strategy.closes
    }


