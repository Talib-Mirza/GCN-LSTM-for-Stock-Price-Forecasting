def validate_model(model, data_loader, edge_index, device):
    model.eval()
    edge_index = edge_index.to(device)
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            google_predictions = model(X_batch, edge_index).squeeze()
            #google_predictions = predictions[2]  # Assuming Google is at index 2

            #print("predictions:", predictions.shape)
            
            all_predictions.extend(google_predictions.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    return np.array(all_predictions), np.array(all_targets)
  
