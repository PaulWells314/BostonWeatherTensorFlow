import requests
import pandas as pd
from dateutil import parser, rrule
from datetime import datetime, time, date
import time
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

def get_weather(year, month, day):
    url = "https://www.wunderground.com/history/airport/KBED/%s/%s/%s/DailyHistory.html?req_city=Weston&req_state=MA&req_statename=Massachusetts&reqdb.zip=02493&reqdb.magic=1&reqdb.wmo=99999"%(year,month,day)
    url2 = "https://www.wunderground.com/history/airport/KORD/%s/%s/%s/DailyHistory.html?req_city=Chicago&req_statename=Illinois"%(year,month,day)
    url3 = "https://www.wunderground.com/history/airport/KNYC/%s/%s/%s/DailyHistory.html?req_city=New+York&req_state=NY&req_statename=New+York&reqdb.zip=10001&reqdb.magic=8&reqdb.wmo=99999"%(year,month,day)
    
    tables = pd.read_html(requests.get(url,
                               headers={'User-agent': 'Mozilla/5.0'}).text)
    dataframe = tables[0]
    
    tables2 = pd.read_html(requests.get(url2,
                               headers={'User-agent': 'Mozilla/5.0'}).text)
    dataframe2 = tables2[0]
    
    tables3 = pd.read_html(requests.get(url3,
                               headers={'User-agent': 'Mozilla/5.0'}).text)
    dataframe3 = tables3[0]
    return (dataframe.iloc[1,1], dataframe2.iloc[1,1], dataframe3.iloc[1,1])
    
# Model architecture parameters
n_weather = 4
n_neurons_1 = 8
n_neurons_2 = 2
n_target = 1

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_weather])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer() 

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_weather, n_neurons_1]))
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

temps = []
temps2 = []
temps3 = []
start_date = "2013/1/1"
end_date = "2018/5/22"
start = parser.parse(start_date)
end = parser.parse(end_date)
dates = list(rrule.rrule(rrule.DAILY, dtstart=start, until=end))    
#print(dates)

for date in dates:
    (temperature, temperature2, temperature3) =get_weather(date.year, date.month, date.day)
    print(temperature, temperature2)
    temps.append(float(temperature.split()[0])/100.0)
    temps2.append(float(temperature2.split()[0])/100.0)
    temps3.append(float(temperature3.split()[0])/100.0)
    
print(temps)
print(temps2)
print(temps3)

t = np.array(temps)
t2 = np.array(temps2)
t3 = np.array(temps3)
plt.plot(t)
plt.plot(t2)
plt.plot(t3)
plt.show()

day_num = np.arange(len(temps))
day_num = (np.modf(day_num/365.25))[0]

# Make Session
net = tf.Session()
# Run initializer
net.run(tf.global_variables_initializer())

# Number of epochs and batch size
epochs = 10000
batch_size = 32

y_train = t[8::1].copy()
X_train =np.column_stack( (t[:len(y_train)], t2[:len(y_train)], t3[:len(y_train)],  day_num[:len(y_train)] ) )

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

        
           

