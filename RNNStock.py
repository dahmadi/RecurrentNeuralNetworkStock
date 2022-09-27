import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pandas_datareader as web 
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, LSTM

#load data

company = 'FB'

#data range 
start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

data = web.DataReader(company, 'yahoo', start, end)

#prepare data 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1)) #predicting closing price

#how many days in the past do I want to look into 
prediction_days = 60 

#defining empty lists for training data
x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
	x_train.append(scaled_data[x-prediction_days:x, 0]0
	y_train.append(scaled_data[x,0])
	
x_train, y_train = np.arrau(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Building neural network 

model = Sequential()

model.add(LSTM(units = 50, return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(unis=1)) #prediciton of closing price

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32) #sees same data 24 times, model sees 32 units at once 

#test for network accuracy 
#load test data

test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].value
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#make a prediciton for the test data

x_test = []

for x in range(prediction_days, len(model_inputs)):
	x_test.append(model_inputs[x-predicition_days:x, 0]0
	
x_test = np.array(x_test)
x_test) = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

predicted_prices - model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

#plot test predicitons 
plt.plot(actual_prices, color = "black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color = "green", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()t
plt.show() 

