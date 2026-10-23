"""ORIGINAL B.Tech code (2022), recovered from the project report.

Kept for reference. Indentation was restored; the logic is unchanged, including
its known problems (fixed in src/spp):
  * the scaler is fitted on the whole dataset, so test data leaks into training;
  * training runs for a single epoch with batch size 1;
  * there is no baseline to judge whether the predictions are any good.

Needs NSE-TATA.csv (NSE Tata Global Beverages daily prices) next to this file.
Originally a Jupyter notebook ("%matplotlib inline" removed).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler

rcParams["figure.figsize"] = 20, 10

# 2. Read the dataset
df = pd.read_csv("NSE-TATA.csv")
print(df.head())

# 3. Analyse the closing prices
df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
df.index = df["Date"]
plt.figure(figsize=(16, 8))
plt.plot(df["Close"], label="Close Price history")

# 4. Sort by date and keep Date and Close
data = df.sort_index(ascending=True, axis=0)
new_dataset = pd.DataFrame(index=range(0, len(df)), columns=["Date", "Close"])
for i in range(0, len(data)):
    new_dataset.loc[i, "Date"] = data["Date"].iloc[i]
    new_dataset.loc[i, "Close"] = data["Close"].iloc[i]

# 5. Normalise
new_dataset.index = new_dataset.Date
new_dataset.drop("Date", axis=1, inplace=True)
final_dataset = new_dataset.values
train_data = final_dataset[0:987, :]
valid_data = final_dataset[987:, :]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(final_dataset)  # NOTE: fitted on ALL data (leakage)

x_train_data, y_train_data = [], []
for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i - 60 : i, 0])
    y_train_data.append(scaled_data[i, 0])
x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

# 6. Build and train the LSTM
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60 :].values
inputs_data = inputs_data.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)

lstm_model.compile(loss="mean_squared_error", optimizer="adam")
lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

# 7. Predict on the validation period
X_test = []
for i in range(60, inputs_data.shape[0]):
    X_test.append(inputs_data[i - 60 : i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_closing_price = lstm_model.predict(X_test)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

# 8. Save the model
lstm_model.save("saved_model.h5")

# 9. Plot predictions against actual prices
train_data = new_dataset[:987]
valid_data = new_dataset[987:].copy()
valid_data["Predictions"] = predicted_closing_price
plt.plot(train_data["Close"])
plt.plot(valid_data[["Close", "Predictions"]])
plt.show()
