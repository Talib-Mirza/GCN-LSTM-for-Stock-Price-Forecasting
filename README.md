# LSTM-GCN Stock Price Forecasting
This project implements a hybrid LSTM-GCN (Long Short-Term Memory - Graph Convolutional Network) model for stock price forecasting. The model captures both temporal and relational dependencies among stocks, leveraging financial OHLCV (Open, High, Low, Close, Volume) data and a stock correlation graph.

# Features

Data Collection: Fetches historical OHLCV data for selected stocks using yfinance.

Graph Construction: Builds a static stock correlation graph based on adjusted close price correlations.

Model Architecture: Combines LSTM for temporal modeling and GCN for relational modeling.

Training and Validation: Predicts stock prices for the next day and evaluates predictions using unseen data.

# License

This project is licensed under the MIT License

# Contributing

Any and all contributions are welcome. Feel free to open issues or submit pull requests.
