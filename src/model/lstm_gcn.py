class LSTM_GCN_Model(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, gcn_hidden_dim, num_lstm_layers, num_gcn_layers, dense_hidden_dim, dropout):
        super(LSTM_GCN_Model, self).__init__()

        # LSTM for temporal encoding
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_lstm_layers, batch_first=True, dropout=dropout)

        # GCN for spatial encoding
        self.gcn_layers = nn.ModuleList([
            GCNConv(lstm_hidden_dim if i == 0 else gcn_hidden_dim, gcn_hidden_dim)
            for i in range(num_gcn_layers)
        ])

        # Fully connected layers for final regression
        self.fc = nn.Sequential(
            nn.Linear(gcn_hidden_dim, dense_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden_dim, 1)  # Output: 1 neuron for price regression
        )

    def forward(self, x, edge_index):
        batch_size, time_steps, num_stocks, feature_dim = x.size()  # x: (batch_size, t, s, 5)
        x = x.permute(0, 2, 1, 3).contiguous()  # Reshape to (batch_size, s, t, 5)
        x = x.view(-1, time_steps, feature_dim)  # Merge batch and stock dimensions
        lstm_out, _ = self.lstm(x)  # Output: (batch_size * s, t, lstm_hidden_dim)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step output
        lstm_out = lstm_out.view(batch_size, num_stocks, -1).permute(1, 0, 2).contiguous()  # Reshape to (s, batch_size, lstm_hidden_dim)

        # GCN encoding
        node_features = lstm_out.mean(dim=1)  # Aggregate over batch (s, lstm_hidden_dim)
        for gcn in self.gcn_layers:
            node_features = F.relu(gcn(node_features, edge_index))
          
        # Fully connected layers for price prediction
        out = self.fc(node_features)
        return out
