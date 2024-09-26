import numpy as np  
import torch
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt

# Function to convert the predictions to PyTorch tensors if they are not already
def ensure_tensors(data_list):
    return [item if isinstance(item, torch.Tensor) else torch.tensor(item) for item in data_list]

# Plotting function for training and testing results
def plot_results(losses, predict, test_losses, test_predict, X_train, look_back, shift_test=1000):
    plt.figure(figsize=(12, 8))
    
    # Convert X_train to a NumPy array for plotting
    X_train_np = X_train.detach().numpy()

    # Ensure all predictions are tensors
    predict = ensure_tensors(predict)
    test_predict = ensure_tensors(test_predict)
    
    # Combine the list of predictions into a single tensor and convert to NumPy
    train_predict_np = torch.cat(predict).detach().numpy()
    test_predict_np = torch.cat(test_predict).detach().numpy()
    
    # Convert losses to NumPy arrays for plotting
    losses_np = torch.tensor(losses).detach().numpy()
    test_losses_np = torch.tensor(test_losses).detach().numpy()
    
    # Plot actual values vs. predictions
    plt.subplot(2, 1, 1) 
    plt.plot(X_train_np, label='Actual Values', color='blue')
    
    train_range = range(len(train_predict_np))
    test_range = range(shift_test, shift_test + len(test_predict_np))

    plt.plot(train_range, train_predict_np, label='Train Predictions', color='green')
    plt.plot(test_range, test_predict_np, label='Test Predictions', color='red')
    
    plt.title('Actual vs Predictions')
    plt.legend()

    # Plot losses
    plt.subplot(2, 1, 2)  
    plt.plot(losses_np, label='Train Losses', color='green')
    plt.plot(range(shift_test, shift_test + len(test_losses_np)), test_losses_np, label='Test Losses', color='red')
    
    plt.title('Training and Testing Losses')
    plt.legend()

    plt.tight_layout()
    plt.show()

