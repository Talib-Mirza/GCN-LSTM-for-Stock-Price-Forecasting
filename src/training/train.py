def train_model(model, data_loader, edge_index, optimizer, criterion, device, num_epochs):
    model.to(device)
    edge_index = edge_index.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(data_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(X_batch, edge_index).squeeze()  # Predictions for all nodes

            # We're predicting Google's price (assume it's node index 2)
            google_predictions = predictions[2]  # Google is at index 2

            # Compute loss
            loss = criterion(google_predictions, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
