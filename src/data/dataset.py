def combine_adjusted_close(stock_data):
    return pd.DataFrame({ticker: data['Adj Close'] for ticker, data in stock_data.items()})

class StockDataset(Dataset):
    def __init__(self, data, window_size):
        self.X, self.y = create_sliding_windows(data, window_size)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
      
def convert_edge_list_to_indices(graph):
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
    edge_index = [
        (node_mapping[edge[0]], node_mapping[edge[1]]) for edge in graph.edges()
    ]
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index_tensor, node_mapping

def combine_ohlcv_data(stock_data):
    return pd.concat({
        ticker: data[['Open', 'High', 'Low', 'Close', 'Volume']] for ticker, data in stock_data.items()
    }, axis=1)
