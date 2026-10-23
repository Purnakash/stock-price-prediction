"""ORIGINAL B.Tech dashboard (2022), recovered from the project report.

Plotly Dash app with two tabs: Tata Global actual vs LSTM-predicted closing
price, and high/low and volume comparison for Tesla, Apple, Facebook and
Microsoft. The report's listing had lost its indentation and had broken
brackets in the layout; those are repaired here, imports are updated to the
current Dash API (``from dash import dcc, html``), and the first chart's x-axis
now uses the validation dates (the original paired training dates with
validation prices). Nothing else is changed.

Needs NSE-TATA.csv, stock_data.csv and saved_model.h5 (from lstm_model.py).
Run:  python stock_app.py   then open http://127.0.0.1:8050
The modern version of this dashboard is ../app.py (Streamlit).
"""

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Dash()
server = app.server

df_nse = pd.read_csv("./NSE-TATA.csv")
df_nse["Date"] = pd.to_datetime(df_nse.Date, format="%Y-%m-%d")
df_nse.index = df_nse["Date"]

data = df_nse.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df_nse)), columns=["Date", "Close"])
for i in range(0, len(data)):
    new_data.loc[i, "Date"] = data["Date"].iloc[i]
    new_data.loc[i, "Close"] = data["Close"].iloc[i]
new_data.index = new_data.Date
new_data.drop("Date", axis=1, inplace=True)

dataset = new_data.values
train = dataset[0:987, :]
valid = dataset[987:, :]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

model = load_model("saved_model.h5")

inputs = new_data[len(new_data) - len(valid) - 60 :].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60 : i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

train = new_data[:987]
valid = new_data[987:].copy()
valid["Predictions"] = closing_price

df = pd.read_csv("./stock_data.csv")

STOCKS = [
    {"label": "Tesla", "value": "TSLA"},
    {"label": "Apple", "value": "AAPL"},
    {"label": "Facebook", "value": "FB"},
    {"label": "Microsoft", "value": "MSFT"},
]
CENTER = {"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "60%"}

app.layout = html.Div(
    [
        html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
        dcc.Tabs(
            id="tabs",
            children=[
                dcc.Tab(
                    label="NSE-TATAGLOBAL Stock Data",
                    children=[
                        html.Div(
                            [
                                html.H2("Actual closing price", style={"textAlign": "center"}),
                                dcc.Graph(
                                    id="Actual Data",
                                    figure={
                                        "data": [go.Scatter(x=valid.index, y=valid["Close"], mode="markers")],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                                html.H2("LSTM Predicted closing price", style={"textAlign": "center"}),
                                dcc.Graph(
                                    id="Predicted Data",
                                    figure={
                                        "data": [
                                            go.Scatter(x=valid.index, y=valid["Predictions"], mode="markers")
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="Facebook Stock Data",
                    children=[
                        html.Div(
                            [
                                html.H1("Facebook Stocks High vs Lows", style={"textAlign": "center"}),
                                dcc.Dropdown(id="my-dropdown", options=STOCKS, multi=True, value=["FB"],
                                             style=CENTER),  # fmt: skip
                                dcc.Graph(id="highlow"),
                                html.H1("Facebook Market Volume", style={"textAlign": "center"}),
                                dcc.Dropdown(id="my-dropdown2", options=STOCKS, multi=True, value=["FB"],
                                             style=CENTER),  # fmt: skip
                                dcc.Graph(id="volume"),
                            ],
                            className="container",
                        ),
                    ],
                ),
            ],
        ),
    ]
)

DROPDOWN = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft"}
COLORS = ["#5E0DAC", "#FF4F00", "#375CB1", "#FF7400", "#FFF400", "#FF0056"]
RANGE_X = {
    "title": "Date",
    "rangeselector": {
        "buttons": [
            {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
            {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
            {"step": "all"},
        ]
    },
    "rangeslider": {"visible": True},
    "type": "date",
}


@app.callback(Output("highlow", "figure"), [Input("my-dropdown", "value")])
def update_highlow(selected_dropdown):
    trace1, trace2 = [], []
    for stock in selected_dropdown:
        rows = df[df["Stock"] == stock]
        trace1.append(go.Scatter(x=rows["Date"], y=rows["High"], mode="lines", opacity=0.7,
                                 name=f"High {DROPDOWN[stock]}"))  # fmt: skip
        trace2.append(go.Scatter(x=rows["Date"], y=rows["Low"], mode="lines", opacity=0.6,
                                 name=f"Low {DROPDOWN[stock]}"))  # fmt: skip
    names = ", ".join(DROPDOWN[i] for i in selected_dropdown)
    return {
        "data": trace1 + trace2,
        "layout": go.Layout(colorway=COLORS, height=600, title=f"High and Low Prices for {names} Over Time",
                            xaxis=RANGE_X, yaxis={"title": "Price (USD)"}),  # fmt: skip
    }


@app.callback(Output("volume", "figure"), [Input("my-dropdown2", "value")])
def update_volume(selected_dropdown_value):
    traces = []
    for stock in selected_dropdown_value:
        rows = df[df["Stock"] == stock]
        traces.append(go.Scatter(x=rows["Date"], y=rows["Volume"], mode="lines", opacity=0.7,
                                 name=f"Volume {DROPDOWN[stock]}"))  # fmt: skip
    names = ", ".join(DROPDOWN[i] for i in selected_dropdown_value)
    return {
        "data": traces,
        "layout": go.Layout(colorway=COLORS, height=600, title=f"Market Volume for {names} Over Time",
                            xaxis=RANGE_X, yaxis={"title": "Transactions Volume"}),  # fmt: skip
    }


if __name__ == "__main__":
    app.run(debug=True)
