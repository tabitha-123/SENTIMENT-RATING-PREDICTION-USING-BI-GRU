from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.shortcuts import redirect
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import numpy as np

import io
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import Dropout, LSTM
from keras.callbacks import ModelCheckpoint
import os
import pickle
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Convolution2D
from math import sqrt
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import math
from keras.models import Model
from keras.layers import GRU, Bidirectional, Input, Dense, Layer
import keras.backend as K





analyzer = SentimentIntensityAnalyzer()

scaler = MinMaxScaler(feature_range = (0, 1))
scaler1 = MinMaxScaler(feature_range = (0, 1))

dataset = pd.read_csv("Dataset/Dataset.csv")
dataset.fillna(0, inplace=True)#remove missing values
data = dataset.values
Y = data[:,3:4]
X = data[:,2:4]

X = scaler.fit_transform(X)
Y = scaler1.fit_transform(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

dataset = dataset.drop_duplicates(subset=['Review_id'])
dataset = dataset.values

#function to calculate accuracy and prediction sales graph
def calculateMetrics(algorithm, predict, test_labels):
    predict = predict.reshape(-1, 1)
    predict = scaler1.inverse_transform(predict)
    test_label = scaler1.inverse_transform(test_labels)
    predict = predict.ravel()
    test_label = test_label.ravel()
    mse_error = sqrt(mean_squared_error(test_label, predict))
    mse_error = mse_error
    return mse_error

X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

lstm_model = load_model("model/lstm_weights.hdf5")
predict = lstm_model.predict(X_test1[:,0:1])
lstm_accuracy = abs(1 - calculateMetrics("LSTM Model", predict, y_test))#call function to plot LSTM crop yield prediction

X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
#training BERT model
#training BERT-like CNN model
bert_model = Sequential()

# reduce number of filters to make model less powerful
bert_model.add(Convolution2D(16, (1 , 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
bert_model.add(MaxPooling2D(pool_size = (1, 1)))

# add dropout to prevent overfitting
bert_model.add(Convolution2D(16, (1, 1), activation = 'relu'))
bert_model.add(MaxPooling2D(pool_size = (1, 1)))
bert_model.add(Dropout(0.4))   # <-- new

bert_model.add(Flatten())
bert_model.add(Dense(units = 128, activation = 'relu'))  # reduced units
bert_model.add(Dropout(0.5))   # <-- new
bert_model.add(Dense(units = 1))

bert_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

if os.path.exists("model/bert_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/bert_weights.hdf5', verbose = 1, save_best_only = True)
    bert_model.fit(X_train1, y_train,
                   batch_size = 16,   # slightly bigger batch size
                   epochs = 20,       # fewer epochs (avoid overfitting)
                   validation_data=(X_test1, y_test),
                   callbacks=[model_check_point],
                   verbose=1)
else:
    bert_model.load_weights("model/bert_weights.hdf5")

predict = bert_model.predict(X_test1)
bert_accuracy = 1 - calculateMetrics("BERT Model", predict, y_test)  # call function to plot LSTM crop yield prediction


# Custom Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Prepare data for Extension Model
X_train_ext = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_ext = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define model
input_layer = Input(shape=(X_train_ext.shape[1], 1))
gru_out = Bidirectional(GRU(64, return_sequences=True))(input_layer)
attn_out = Attention()(gru_out)
dense1 = Dense(128, activation='relu')(attn_out)
output_layer = Dense(1)(dense1)

extension_model = Model(inputs=input_layer, outputs=output_layer)
extension_model.compile(optimizer='adam', loss='mean_squared_error')

# Train / Load Weights
if not os.path.exists("model/extension_weights.hdf5"):
    model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5',
                                        verbose=1, save_best_only=True)
    extension_model.fit(X_train_ext, y_train,
                        batch_size=8, epochs=1000,
                        validation_data=(X_test_ext, y_test),
                        callbacks=[model_check_point], verbose=1)
else:
    extension_model.load_weights("model/extension_weights.hdf5")

# Predict & Evaluate
predict_ext = extension_model.predict(X_test_ext)
extension_accuracy = 1 - calculateMetrics("Extension Bi-GRU model", predict_ext, y_test)





def FileComment(request):
    if request.method == 'GET':
       return render(request, 'FileComment.html', {})

def SingleComment(request):
    if request.method == 'GET':
       return render(request, 'SingleComment.html', {})    

def getSentiment(comment):
    sentiment = -1
    vs = analyzer.polarity_scores(comment)
    compound = vs['compound']
    if compound >= 0.5:
        sentiment =  5
    elif compound < 0.5 and compound >= 0.1:
        sentiment = 4
    elif compound < 0.1 and compound >= 0.05:
        sentiment = 3
    elif compound < 0.05 and compound > -0.05:
        sentiment = 2
    else:
        sentiment = 1
    return sentiment, compound

def sentimentValue(predict):
    sentiment = "None"
    if predict >= 4:
        sentiment = "Positive"
    elif predict == 3:
        sentiment = "Neutral"
    else:
        sentiment = "Negative";
    return sentiment

def SingleCommentAction(request):
    if request.method == 'POST':
        global cnn_model, scaler, scaler1
        bert_model = load_model("model/bert_weights.hdf5")
        comment = request.POST.get('t1', False)
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Test Review</th><th><font size="" color="black">Predicted Sentiment</th>'
        output +='<th><font size="" color="black">Rating</th></tr>'
        sentiment, hatred = getSentiment(comment)#finding hatred percentage
        data = []
        data.append([sentiment, sentiment])
        data = np.asarray(data)
        data = scaler.transform(data)
        data = np.reshape(data, (data.shape[0], data.shape[1], 1, 1))
        predict = bert_model.predict(data)
        predict = scaler1.inverse_transform(predict)
        predict = predict.ravel()
        predict = predict[0]
        predict = int(round(predict))
        sentiment = sentimentValue(predict)
        ratings = ""
        for i in range(0, predict):
            ratings += "*"
        output+='<td><font size="" color="black">'+comment+'</td><td><font size="" color="black">'+sentiment+'</td>'
        output +='<td><font size="" color="black">'+str(ratings)+'</td></tr>'
        output+= "</table></br>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def FileCommentAction(request):
    if request.method == 'POST':
        global dataset, scaler, scaler1, cnn_model
        bert_model = load_model("model/bert_weights.hdf5")
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("SentimentApp/static/"+fname):
            os.remove("SentimentApp/static/"+fname)
        with open("SentimentApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        testData = pd.read_csv("SentimentApp/static/"+fname)
        testData.fillna(0, inplace = True)
        testData = testData.values
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Test Review</th><th><font size="" color="black">Predicted Sentiment</th><th><font size="" color="black">Ratings</th></tr>'
        result = []
        for i in range(len(testData)):
            comment = testData[i,0]
            sentiment, hatred = getSentiment(comment)
            data = []
            data.append([sentiment, sentiment])
            data = np.asarray(data)
            data = scaler.transform(data)
            data = np.reshape(data, (data.shape[0], data.shape[1], 1, 1))
            predict = bert_model.predict(data)
            predict = scaler1.inverse_transform(predict)
            predict = predict.ravel()
            predict = predict[0]
            predict = int(round(predict))
            sentiment = sentimentValue(predict)
            ratings = ""
            for i in range(0, predict):
                ratings += "*"
            result.append(predict)
            output+='<td><font size="" color="black">'+comment+'</td><td><font size="" color="black">'+str(sentiment)+'</td><td><font size="" color="black">'+str(ratings)+'</td></tr>'
        output+= "</table></br>"
        unique, count = np.unique(np.asarray(result), return_counts=True)
        plt.pie(count,labels=unique,autopct='%1.1f%%')
        plt.title('Sentiment Prediction Graph')
        plt.axis('equal')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)      

def TrainBERT(request):
    if request.method == 'GET':
        global mse_error
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th></tr>'
        output+='<td><font size="" color="black">BERT Algorithm</td><td><font size="" color="black">'+str(bert_accuracy)+'</td></tr>'
        output+='<td><font size="" color="black">LSTM Algorithm</td><td><font size="" color="black">'+str(lstm_accuracy)+'</td></tr>'
        output+='<td><font size="" color="black">Extension Bi-GRU Model</td><td><font size="" color="black">'+str(extension_accuracy)+'</td></tr>'
        output+= "</table></br></br></br>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)
    

def LoadDatasetAction(request):
    if request.method == 'POST':
        global dataset
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("SentimentApp/static/"+fname):
            os.remove("SentimentApp/static/"+fname)
        with open("SentimentApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        datasets = pd.read_csv("SentimentApp/static/"+fname,nrows=1000)
        datasets.fillna(0, inplace = True)
        columns = datasets.columns
        datasets = datasets.values
        output='<table border=1 align=center width=100%><tr>'
        for i in range(len(columns)):
            output += '<th><font size="" color="black">'+columns[i]+'</th>'
        output += '</tr>'
        for i in range(0, 300):
            output += '<tr>'
            for j in range(len(columns)):
                output += '<td><font size="" color="black">'+str(datasets[i,j])+'</td>'
            output += '</tr>'
        output+= "</table></br></br></br></br>"
        #print(output)
        context= {'data':output}
        return render(request, 'UserScreen.html', context)    

def LoadDataset(request):
    if request.method == 'GET':
       return render(request, 'LoadDataset.html', {})  

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})

def SignupAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)

        if User.objects.filter(username=username).exists():
            context = {'data': 'Username already exists'}
            return render(request, 'Signup.html', context)

        user = User.objects.create_user(username=username, password=password, email=email)
        return redirect('UserLogin')

    return render(request, 'Signup.html')

def UserLoginAction(request):
    if request.method == 'POST':
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            context = {'data': f'Welcome {username}'}
            return render(request, 'UserScreen.html', context)
        else:
            context = {'data': 'Invalid login details'}
            return render(request, 'UserLogin.html', context)

    return render(request, 'UserLogin.html')
