import requests
import pandas as pd
from   dateutil import parser, rrule
from   datetime import datetime, time, date
import time
from   matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import os
import sys

city_url = {

"Weston"  : "https://www.wunderground.com/history/airport/KBED/%s/%s/%s/DailyHistory.html?req_city=Weston&req_state=MA&req_statename=Massachusetts&reqdb.zip=02493&reqdb.magic=1&reqdb.wmo=99999",
"Chicago" : "https://www.wunderground.com/history/airport/KORD/%s/%s/%s/DailyHistory.html?req_city=Chicago&req_statename=Illinois",
"NewYork" : "https://www.wunderground.com/history/airport/KNYC/%s/%s/%s/DailyHistory.html?req_city=New+York&req_state=NY&req_statename=New+York&reqdb.zip=10001&reqdb.magic=8&reqdb.wmo=99999"
}

def get_weather(year, month, day):
   
    url_list = []
    url_list.append(city_url["Weston"]%(year,month,day))
    url_list.append(city_url["Chicago"]%(year,month,day))
    url_list.append(city_url["NewYork"]%(year,month,day))
    
    weather_data_list = []
    
    for url in url_list:
        tables = pd.read_html(requests.get(url, headers={'User-agent': 'Mozilla/5.0'}).text)
        dataframe = tables[0]
        weather_data_list.append(dataframe.iloc[1,1])
        
    return weather_data_list
    
    
    
# Model architecture parameters
n_weather     = 4
n_filter_days = 6
n_neurons_1   = 8
n_neurons_2   = 2
n_target      = 1

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_filter_days* (n_weather-1) +1])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer() 

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_filter_days* (n_weather-1) +1, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_2, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))

# Hidden layer 2
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_2, W_out), bias_out))
 
# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))   
# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse) 

start_date = "2013/1/1"
end_date   = "2018/5/28"
start = parser.parse(start_date)
end = parser.parse(end_date)
dates = list(rrule.rrule(rrule.DAILY, dtstart=start, until=end))    

temps = []
tlist = []

for i in range(n_weather-1):
    temps.append([])
    tlist.append([])   

# read data from website and store
if sys.argv[1] == "w":
    with open('weather.txt', mode = 'w') as f:

        for date in dates:
            weather_data_list =get_weather(date.year, date.month, date.day)
            for counter, temperature in enumerate(weather_data_list): 
                print(date, temperature)
                f.write("%s "%temperature.split()[0]) 
                temps[counter].append(float(temperature.split()[0])/100.0)
            f.write(os.linesep)   
            print("")
 
# read data from file
else:
    with open('weather.txt', mode = 'r') as f:
        for line in f:
            if len(line.split()) >=3:
                temps[0].append(float(line.split()[0])/100.0)
                temps[1].append(float(line.split()[1])/100.0)
                temps[2].append(float(line.split()[2])/100.0)
            
for counter, temp in enumerate(temps):
    print(temp)
    tlist[counter] = np.array(temp)
    plt.plot(tlist[counter])
  
plt.show()

day_num = np.arange(len(temps[0]))
day_num = (np.modf(day_num/365.25))[0]

# Make Session
net = tf.Session()
# Run initializer
net.run(tf.global_variables_initializer())

# Number of epochs and batch size
epochs = 10000
batch_size = 128

y_train = tlist[0][n_filter_days::1].copy()

X_train =np.column_stack( [t[0:len(y_train)] for t in tlist ] )
for t in tlist:
    for i in range(n_filter_days-1):
        X_train =np.column_stack( (X_train, t[i+1:len(y_train)+i+1]) )
    
X_train =np.column_stack( (X_train,  day_num[:len(y_train)]) ) 

for e in range(epochs):
    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})
    var = net.run(mse, feed_dict={X: batch_x, Y: batch_y})
    print("**loss**",var)

        
           

