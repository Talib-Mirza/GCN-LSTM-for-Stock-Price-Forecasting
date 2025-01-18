import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JNJ', 'JPM', 'V']
    start_date = "2017-01-01"
    end_date = "2024-01-01"
    window_size = 60

    # Fetch data
    stock_data = fetch_stock_data(tickers, start_date, end_date)
    ohlcv_data = combine_ohlcv_data(stock_data)

    # Normalize and prepare data
    normalized_data = (ohlcv_data - ohlcv_data.mean()) / ohlcv_data.std()
    all_data = normalized_data.values  # Convert to NumPy array

    all_data = all_data.reshape(1760, 10, 5)
    print("all_data:", all_data.shape)

    # Create dataset and DataLoader
    dataset = StockDataset(all_data, window_size)
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    adj_close_data = combine_adjusted_close(stock_data)
    graph, correlation_matrix = build_correlation_graph(adj_close_data, threshold=0.6)

    
    edge_index, node_mapping = convert_edge_list_to_indices(graph)

    visualize_graph(graph, correlation_matrix)
    print(sns.heatmap(adj_close_data.pct_change().dropna().corr(), xticklabels=tickers, yticklabels=tickers, annot=True))

    model = LSTM_GCN_Model(
        input_dim=5,  # OHLCV features
        lstm_hidden_dim=64,
        gcn_hidden_dim=32,
        num_lstm_layers=2,
        num_gcn_layers=2,
        dense_hidden_dim=128,
        dropout=0.3
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train the model
    model.load_state_dict(torch.load("/kaggle/working/model_weights.pth"))
    model.to(device)

    predictions, targets = validate_model(model, test_loader, edge_index, device)

    # Compare predictions with actual prices
    for i, (pred, actual) in enumerate(zip(predictions[:10], targets[:10])):
        actual = actual.reshape(5)
        print("pred:", pred, "actual:", actual)
        
        #print(f"Sample {i + 1}: Predicted: {pred:.4f}, Actual: {float(actual):.4f}")
if __name__ == "__main__":
    main()
