def create_sliding_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size - 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 3])  # Use "Close" price as the target
    return np.array(X), np.array(y)
