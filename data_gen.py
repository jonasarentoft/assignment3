import numpy as np
from fastapi import FastAPI, Query

app = FastAPI()

class TradingStrategy:
    def __init__(self):
        
        self.dc_events = []
        self.os_events = []
        self.profits = []
        self.open_events = []
        self.take_profit_events = []
        self.stop_loss_events = []
        self.prices = []
        self.confirmation_time = None
        self.open_time = None
        self.time_diffs = []
        self.time_last_event = []
        
        self.sma_30 = []
        self.sma_5 = []
        self.osvs = []
        self.first_trade = True
        self.z = []
        self.data_prices = []
        self.open_osv = None

        
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
        

strategy = TradingStrategy()

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
            print(f'First price is {tickprice}')
            
        strategy.trend = strategy.detect_trend(tickprice)
        
        if strategy.trend:
            print(f'First trend is {strategy.trend} @ {tickprice}')
            strategy.dc_events.append([strategy.counter, tickprice])
    
    elif strategy.trend == 'down':
        if strategy.last_low[1] > tickprice:
            strategy.last_low = [strategy.counter, tickprice]
            
        p_ext = strategy.dc_events[-1][1]
        p_dcc = p_ext * (1 - strategy.theta)
    
        osv = ((tickprice - p_dcc) / p_dcc) / strategy.theta
        strategy.osvs.append(osv)
        
        if ((osv <= strategy.threshold) and (not strategy.has_open_position)):
            strategy.signal = 'buy'
            strategy.open_osv = osv
                        

            strategy.has_open_position = True
            print(f'Position opened @ {tickprice}')
            strategy.open_price = tickprice
            strategy.open_time = strategy.counter
            strategy.open_events.append([strategy.counter, strategy.open_price])

        if tickprice >= strategy.last_low[1] * (1 + strategy.theta):
            print(f'Trend changed to up')
            strategy.trend = 'up'
            strategy.dc_events.append([strategy.counter, tickprice])
            strategy.os_events.append(strategy.last_low)
            strategy.last_high = [strategy.counter, tickprice]

            
            if strategy.has_open_position:
                strategy.signal = 'sell'
                strategy.has_open_position = False
                
                if tickprice > strategy.open_price:
                    if strategy.first_trade:
                        strategy.first_trade = False
                        
                        
                    else:
                        strategy.z.append(1)
                        strategy.time_diffs.append(strategy.open_time - strategy.confirmation_time)
                        strategy.time_last_event.append(strategy.confirmation_time - strategy.os_events[-2][0])
                        strategy.osvs.append(strategy.open_osv)
                        strategy.data_prices.append(tickprice)
                        strategy.sma_30.append(np.mean(strategy.prices[-60:], axis = 0)[1])
                        strategy.sma_5.append(np.mean(strategy.prices[-5:], axis = 0)[1])
                        
                    print(f'Took profit @ {tickprice}')
                
                else:
                    if strategy.first_trade:
                        strategy.first_trade = False
                    else:
                        strategy.z.append(0)
                        strategy.time_diffs.append(strategy.open_time - strategy.confirmation_time)
                        strategy.time_last_event.append(strategy.confirmation_time - strategy.os_events[-2][0])
                        strategy.osvs.append(strategy.open_osv)
                        strategy.data_prices.append(tickprice)
                        strategy.sma_30.append(np.mean(strategy.prices[-60:], axis = 0)[1])
                        strategy.sma_5.append(np.mean(strategy.prices[-5:], axis = 0)[1])

                    print(f'Stopped Loss @ {tickprice}')
    
    elif strategy.trend == 'up':
        if strategy.last_high[1] < tickprice:
            strategy.last_high = [strategy.counter, tickprice]
            
        if tickprice <= strategy.last_high[1] * (1 - strategy.theta):
            strategy.dc_events.append([strategy.counter, tickprice])
            strategy.os_events.append(strategy.last_high)
            strategy.trend = 'down'
            strategy.confirmation_time = strategy.counter
            strategy.last_low = [strategy.counter, tickprice]
            
        
                
                
                
                
            
    strategy.increment_counter()
        
    return {"tradeSignal": strategy.signal}

@app.get("/get_data")
async def read_data():
    return {
        "prices" : strategy.data_prices,
        "osv" : strategy.open_osv,
        "sma_30" : strategy.sma_30,
        "sma_5" : strategy.sma_5,
        "z" : strategy.z,
        "time_diffs" : strategy.time_diffs,
        "time_last_event" : strategy.time_last_event,
    }
    