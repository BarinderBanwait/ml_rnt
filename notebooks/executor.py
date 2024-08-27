import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from pathlib import Path
import copy

# Define the training function
def train(model, train_dataloader, val_dataset, loss_func, evaluator, optimizer, num_epochs, if_regression=False, verbose=True):
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
    if_regression: bool
        Whether it is a regression problem. If True, the output will be squeezed to 1 dimension
    verbose: bool
        Whether to print the training and validation evaluation after each epoch
    '''
    train_eval_hist = []
    val_eval_hist = []

    # # Instantiate the model to train
    # model.train()

    best_val_eval = float('-inf')
    best_epoch = 0
    for epoch in range(num_epochs):
        # Instantiate the model to train
        model.train()

        for inputs, labels in train_dataloader:
            # Forward pass
            outputs = model(inputs)
            if if_regression == True:
                outputs = outputs.squeeze()
            loss = loss_func(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # print each epoch training and validation loss
        train_eval = test(model, train_dataloader.dataset, evaluator, if_regression = if_regression)
        val_eval = test(model, val_dataset, evaluator, if_regression = if_regression)
        if val_eval > best_val_eval:
            best_val_eval = val_eval
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        if verbose == True:
            print(f'Epoch {epoch+1}/{num_epochs}, Training {evaluator.__name__}: {train_eval}, Validation {evaluator.__name__}: {val_eval}')
        train_eval_hist.append(train_eval)
        val_eval_hist.append(val_eval)

    # save the trained model locally
    model_path = Path("..") / "trained_models" / "model.pth"
    print(f'Save the model from epoch {best_epoch} with {evaluator.__name__} {best_val_eval} to {model_path}')
    torch.save(best_model, model_path)  
    return best_model, train_eval_hist, val_eval_hist

# Test function here
@torch.no_grad()
def test(model, test_dataset, evaluator, if_regression = False, verbose=False):
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
    if if_regression == False:
        _, y_pred = torch.max(outputs.data, 1)
    else:
        y_pred = outputs.squeeze()
    test_res = evaluator(y_test, y_pred)
    if verbose == True:
        print(f'Test {evaluator.__name__}: {test_res}')
    return test_res