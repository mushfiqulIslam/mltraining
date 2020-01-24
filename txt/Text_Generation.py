# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
##from keras.callbacks import ModelCheckpoint
##from keras.utils import np_utils
# load ascii text and covert to lowercase

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

raw_text = open("wonderland.txt", encoding="utf-8").read()
#raw_text = raw_text.lower()
#allText = word_tokenize(raw_text)

tokenizer = RegexpTokenizer(r'\w+')
tokenized = tokenizer.tokenize(raw_text)
print("length ",len(tokenized))
romanNum = ['_i_', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii']
removewords = ['ll', 've', 're', 's', 't', 'the', 'A', 'a', 'Oh', 'And', 'or', 'Ma']

tokenized = [w for w in tokenized if w not in romanNum]
tokenized = [w for w in tokenized if w not in removewords]
print("After removing some words length ",len(tokenized))

le = preprocessing.LabelEncoder()
le.fit(tokenized)
array_list = le.transform(tokenized)
print("array_list shape is ", array_list.shape, "class size: ", len(list(le.classes_)))

###print(list(le.classes_))
###print('############',le.transform(["flavour", "golden", "Alice"]))

shapable = array_list[:-1].copy()
data = shapable.reshape(-1, 27348)
data = data.astype('float32')
print(data.shape)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data)
scaled = scaler.transform(data)
#print("scaled values",scaled)

# specify the number of lag hours
n_hours = 5
n_features = 5
n_ahead = 5
st = n_hours*n_features
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, n_ahead)
#reframed.to_csv("reframed.csv")
print("column number")
print(reframed.columns,len(reframed.columns), len(reframed.index))

# split into train and test sets
values = reframed.values
train_size = int(len(values) * 0.8)
test_size = len(values) - train_size
train, test = values[0:train_size,:], values[train_size:-1,:]

# split into input and outputs
n_obs = n_hours * n_features + n_hours
train_X, train_y = train[:, 0:n_obs], train[:, n_obs:]
test_X_n, test_y = test[:, 0:n_obs], test[:, n_obs:]
print("train_X Shape ", train_X.shape, "train_y shape ", train_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape(train_X.shape[0], n_hours, n_features+1)
test_X = test_X_n.reshape(test_X_n.shape[0], n_hours, n_features+1)
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

batchsize = 100
# design network
model = Sequential()
model.add(LSTM(256, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(train_y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=20, batch_size=batchsize, validation_data=(test_X, test_y), verbose=2, shuffle=False)

## list all data in history
#print(history.history.keys())

## save model to single file
#model.save('lstm_model_mutivariate_multistep_multiout.h5')
#

# make a prediction
yhat = model.predict(test_X)
print("prediction: ", yhat, yhat.shape)

reshaped_yhat = yhat.reshape(yhat.shape[0]*int(yhat.shape[1]/data.shape[1]),data.shape[1])

inv_yhat = scaler.inverse_transform(reshaped_yhat)
print("scaled prediction: ", inv_yhat)

round_value = np.around(inv_yhat)

a = round_value.flatten()
#a = int(round_value.reshape(-1,1))
a = a.astype('int')
decoded = le.inverse_transform(a)
print("Decoded prediction: ", decoded)

##test_X_plot = test_X.reshape((test_X.shape[0], n_obs))
##train_X_plot = train_X.reshape((train_X.shape[0], n_obs))
##print("yhat",yhat,yhat.shape,"test_X",test_X.shape,train_X.shape)
