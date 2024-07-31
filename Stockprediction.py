import numpy as np
import yfinance as yf
import keras
import tensorflow as tf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

model =keras.models.load_model("C:/Users/ADMIN/Stock Price Prediction/Stock Prediction.keras")



st.header("Stock Price Prediction")

stock= st.text_input("Enter the stock", "GOOG")
start='2012-01-01'
end='2024-07-07'
data=yf.download(stock,start,end)
st.subheader('Stock Data')
st.write(data)



data_train=pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test=pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

from sklearn.preprocessing import MinMaxScaler
Scaler=MinMaxScaler(feature_range=(0,1))

pas_100_day=data_train.tail(100)
data_test=pd.concat([pas_100_day, data_test], ignore_index=True)
data_tes_sc=Scaler.fit_transform(data_test)

st.subheader('MA50 data')
ma_50_stock=data.Close.rolling(50).mean()
figure1=plt.figure(figsize=(10,8))
plt.plot(ma_50_stock, 'r', "Mean of 50 days")
plt.plot(data.Close, 'g', "Actual data")
plt.show()
st.plot(figure1)



x=[]
y=[]
for i in range(100, data_tes_sc.shape[0]):
    x.append(data_tes_sc[i-100:i])
    y.append(data_tes_sc[i,0])

x=np.array(x)
y=np.array(y)

predict=model.predict(x)

scale= 1/Scaler.scale_
