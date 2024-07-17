import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn

# Define the training function
def train(model, train_dataloader, val_dataset, loss_func, evaluator, optimizer, num_epochs, verbose=True):
    '''
    Parameters:
    model: torch.nn.Module
        The model to train
    train_dataloader: torch.utils.data.DataLoader
        The dataloader containing the training data
    val_dataset: torch.utils.data.Dataset
        The validation dataset
    loss_func: function
        The loss function to use to train the model. This needs to be a differential function.
    evaluator: function
        The evaluation function to evaluate the model performace after each epoch
    optimizer: torch.optim.Optimizer
        The optimizer to use
    num_epochs: int
        The number of epochs to train
    verbose: bool
        Whether to print the training and validation evaluation after each epoch
    '''
    train_eval_hist = []
    val_eval_hist = []

    # Instantiate the model to train
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # print each epoch training and validation loss
        train_eval = test(model, train_dataloader.dataset, evaluator)
        val_eval = test(model, val_dataset, evaluator)
        if verbose == True:
            print(f'Epoch {epoch+1}/{num_epochs}, Training {evaluator.__name__}: {train_eval}, Validation {evaluator.__name__}: {val_eval}')
        train_eval_hist.append(train_eval)
        val_eval_hist.append(val_eval)
    return model, train_eval_hist, val_eval_hist

# Test function here
@torch.no_grad()
def test(model, test_dataset, evaluator, verbose=False):
    '''
    Parameters:
    model: torch.nn.Module
        The model to test
    test_dataset: torch.utils.data.Dataset
        The test dataset
    evaluator: function
        The evaluation function to evaluate the model performace
    verbose: bool
        Whether to print the test results
    '''
    X_test, y_test = test_dataset.tensors
    model.eval()
    outputs = model(X_test)
    _, y_pred = torch.max(outputs.data, 1)
    test_res = evaluator(y_pred, y_test)
    if verbose == True:
        print(f'Test {evaluator.__name__}: {test_res}')
    return test_res