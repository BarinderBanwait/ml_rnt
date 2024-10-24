# python version 3.10.7
# scikit-learn == 1.1.2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# load the dataset
def load_data(path):
    df = pd.read_parquet(path)
    print(f"Loaded the dataset with {len(df.columns) - 1} features and {len(df)} curves..")
    return df

# convert the rank column to binary accodint to a threshold
def convert_rank_to_binary(df, threshold):
    df['rank'] = df['rank'].apply(lambda x: 1 if x > threshold else 0)
    print(f'Converted the rank column to binary. The value of 1 means the rank is greater than {threshold}, otherwise 0. Rank counts:')
    print(df['rank'].value_counts().to_frame().rename(columns={'rank': 'count'}))
    return df

def get_input_output_dim(df, label_col, if_regression = False):
    in_dim = len(df.columns) - 1
    if if_regression == True:
        out_dim = 1
    else:
        out_dim = df[label_col].nunique()
    print(f'The input dimension is {in_dim} and the output dimension is {out_dim}.')
    return in_dim, out_dim

# returns cuda if we have cuda available, otherwise return cpu
def get_device():
    '''
    Check if we have cuda available. Return cuda version if available
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}.'.format(device))
    return device

# print model summary including architecture and number of parameters
def model_summary(model):
    '''
    Print the model architecture and number of parameters
    '''
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters..')
    print(model)

# round to the nearest perfect square
def round_to_nearest_square(y):
    if y <= 0:
        return 1
    lower = min(1,torch.floor(torch.sqrt(y)))**2
    upper = torch.ceil(torch.sqrt(y))**2
    if y - lower <= upper - y:
        return lower
    else:
        return upper
    
def array_round_to_nearest_square(arr):
    rounded_arr = torch.zeros_like(arr)
    for i,y in enumerate(arr):
        rounded_arr[i] = round_to_nearest_square(y)
    return rounded_arr

# define an accuracy_score function to evaluate the model:
# step 1 : round y_pred to the nearest perfect square
# step 2 : calculate the accuracy score
def perfect_square_acc(y_true, y_pred):
    y_pred = array_round_to_nearest_square(y_pred)
    try:
        res = accuracy_score(y_true, y_pred)
    except TypeError:
        res = accuracy_score(y_true.cpu(), y_pred.cpu())
    return res

# define a nearest integer_accuracy_score function to evaluate the model:
# step 1 : round y_pred to the nearest integer
# step 2 : calculate the accuracy score
def nearest_integer_acc(y_true, y_pred):
    try:
        y_pred = torch.round(y_pred)
        res = torch.sum(y_pred.squeeze() == y_true) / len(y_true)
    except TypeError:
        res = accuracy_score(y_true.cpu(), y_pred.cpu())
    return res

# split the data into training and test sets and use dataloaders to create batches
def prepare_data(data, label_col, device, test_size=0.2, batch_size=32, random_state=42, shuffle=True, if_regression=False, drop_last=True, if_standardize = False):
    X = data.drop(columns=[label_col]).values
    y = data[label_col].values
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    if if_regression == True:
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    else:
        y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    # Split the data into training, validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    # # standardize the data
    if if_standardize:
        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train.cpu()), dtype=torch.float32).to(device) # learn the mean and std from the training set
        X_test = torch.tensor(scaler.transform(X_test.cpu()), dtype=torch.float32).to(device)   # apply the mean and std to the test set
        X_val = torch.tensor(scaler.transform(X_val.cpu()), dtype=torch.float32).to(device)  # apply the mean and std to the test set

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = drop_last)

    return train_dataloader, val_dataset, test_dataset

def plot_train_eval_hist(train_eval_hist, val_eval_hist, size = (12, 6), title = '', show = True):
    plt.figure(figsize=size)
    plt.plot(train_eval_hist, label='train evaluation')
    plt.plot(val_eval_hist, label='validation evaluation')
    plt.xlabel('Epochs')
    plt.ylabel('Evaluation Metric')
    plt.title('Evaluation Metric by Epochs')
    plt.legend()  # Show legend to identify the lines
    plt.title(title)
    # save the plot
    plt.savefig(f'{title}.png')
    if show:
        plt.show()
    
def plot_train_loss_hist(train_loss_hist, eval_loss_hist, size = (12, 6), title = '', show = True):
    plt.figure(figsize=size)
    plt.plot(train_loss_hist, label='train loss')
    plt.plot(eval_loss_hist, label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss by Epochs')
    plt.legend()  # Show legend to identify the lines
    plt.title(title)
    # save the plot
    plt.savefig(f'{title}.png')
    if show:
        plt.show()
    
def sliced_data(df, lower_bound, upper_bound):
    # slice the dataset by the desired lowerbound and upperbound of conductors
    # print(f"Sliced the dataset to have curves with conductor witin the range of [{lower_bound}, {upper_bound}]..")
    return df.loc[df['conductor'] >= lower_bound].loc[df['conductor'] <= upper_bound]

def getRes(sliced_df, model, metric, test_ratio, shuffle, random_state):
    '''
    This function takes a the sliced dataframe and returns the metric result of the model according to the number of a_p's (in test data)

    Parameters:
    sliced_df: pd.DataFrame. 
        The sliced dataframe with the desired lower and upperbound of conductors
    model: 
        your chosen model to train and test 
    n_ap: int. 
        The number of a_p's to use as features
    metric: function.
        The metric to use to evaluate the model
    test_ratio: float  
    shuffle: bool.
        If True, the data will be shuffled before splitting into training and testing sets.
    random_state: int.
        The random seed to use for train test split.
    '''

    X = sliced_df.drop(columns=['rank']).values
    y = sliced_df['rank'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, shuffle = shuffle, random_state=random_state)

    # train the model
    model.fit(X_train, y_train)

    # return the metric result of the model on test data
    y_test_pred = model.predict(X_test)
    return metric(y_test, y_test_pred)    

def Generate_AccByNumAps_df(df, lower_bound, upper_bound, model, step_size = 10, metric = accuracy_score, test_ratio = 0.25, if_using_cond = False, shuffle = True, random_state = 42):
    '''
    This function generates a dataframe of the number of a_p's and the accuracy of the model for a given sliced dataframe

    Parameters:
    df: pd.DataFrame.
        The dataframe to be sliced
    lower_bound: int. 
        The lower bound of the value of conductors
    upper_bound: int.
        The upper bound of the value of conductors
    model: class
        your chosen model to train and test
    step_size: int. 
        The step size to increment the number of a_p's by
    metric: function. 
        The metric to use to evaluate the model. Default is accuracy_score.
    test_ratio: float. 
        The ratio of the test set size to the training set size. Default is 0.25.
    if_using_cond: bool. 
        If True, the model will use the number of conductors as a feature. Default is False.
    shuffle: bool.
        If True, the data will be shuffled before splitting into training and testing sets. Default is True.
    random_state: int.
        The random seed to use for train test split. Default is 42.
    '''

    print('*'*50)
    print(f"Generating the accuracy by the number of a_p's dataframe for range [{lower_bound}, {upper_bound}]..")

    # slice the dataframe accodring to 
    sliced_df = sliced_data(df, lower_bound, upper_bound)
    print(f"There are {len(sliced_df)} curves within the conductor range [{lower_bound}, {upper_bound}].")

    # do we want conductors as a feature?
    if if_using_cond == False:
        sliced_df = sliced_df.drop(columns = ['conductor'])
    else:
        # gotta normilize the conductor column since it's too big:
        # log the conductor column first
        # then normalize the conductor column by dividing by the log of max value
        sliced_df['conductor'] = np.log(sliced_df['conductor'])/np.log(sliced_df['conductor'].max())

    # create a dataframe to store the number of a_p's and the accuracy
    res_df = pd.DataFrame(columns = ['num_a_p', 'performance'])

    # iterate through the number of a_p's
    tot_n_aps = len(sliced_df.columns) - 2
    for i in range(step_size, tot_n_aps+step_size, step_size):
        # slice the dataframe to have i a_p's
        # add the conductor column if we are using it
        if if_using_cond == False:
            cur_df = sliced_df.iloc[:, :i].join(sliced_df['rank'])
        else:
            cur_df = sliced_df.iloc[:, :i].join(sliced_df[['conductor','rank']])

        # get the metric result of the model within test data
        res = getRes(cur_df, model, metric, test_ratio, shuffle, random_state)

        # append the metric result to the dataframe
        res_df = pd.concat([res_df, pd.DataFrame({'num_a_p': i, 'performance': res}, index = [0])], ignore_index = True)

    return res_df

def plot_AccuracycByNumAps(res_dict, metric_name = 'accuracy', size=(12, 6)):
    plt.figure(figsize=size)
    for bounds, acc_df in res_dict.items():
        lower_bound, upper_bound = bounds
        plt.plot(acc_df['num_a_p'], acc_df['performance'], label=f'Bounds: {lower_bound} to {upper_bound}')   

    # if metric_name = matthews_corrcoef, change it to MCC
    if metric_name == 'matthews_corrcoef':
        metric_name = 'MCC'

    plt.xlabel('Number of a_p\'s')
    plt.ylabel(metric_name)
    plt.title('{} by number of a_p for Different Bounds'.format(metric_name))
    plt.legend()  # Show legend to identify the lines
    plt.tight_layout()
    plt.show()

def plot_Heatmap(res_dict, metric_name = 'accuracy', size=(800,800)):
    bounds_list = list(res_dict.keys())

    # get the x axis of the 3d surface - number of a_p's
    x = res_dict[bounds_list[0]]['num_a_p']
    # get the y axis of the 3d surface - end points of bounds
    y = [bounds[1] for bounds in bounds_list]
    # get log 2 of the y
    y = [np.log2(y_val) for y_val in y]
    # get the z axis of the 3d surface - the metric
    z = [res_dict[bounds]['performance'] for bounds in bounds_list]

    # if metric_name = matthews_corrcoef, change it to MCC
    if metric_name == 'matthews_corrcoef':
        metric_name = 'MCC'

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title=f'{metric_name} by the number of ap\'s and log_2(conductor) upperbound', autosize=False,
                    width=500, height=500,
                    margin=dict(l=65, r=50, b=65, t=90))
    # change the size of fig
    fig.update_layout(width=size[0], height=size[1])

    # change fig axis labels
    fig.update_layout(scene = dict(
                        xaxis_title='Number of a_p\'s',
                        yaxis_title='Log_2(Conductor) upperbound',
                        zaxis_title= metric_name),
                        )

    return fig

def Generate_AccByApRange_df(df, lower_bound, upper_bound, model, n_ap, ap_selection = "rolling", stride = 10, metric = accuracy_score, test_ratio = 0.25, if_using_cond = False, shuffle = True, random_state = 42):
    '''
    This function generates a dataframe of the number of a_p's and the accuracy of the model for a given sliced dataframe

    Parameters:
    df: pd.DataFrame.
        The dataframe to be sliced
    lower_bound: int.
        The lower bound of the value of conductors
    upper_bound: int.
        The upper bound of the value of conductors
    n_ap: int.
        The number of a_p's to use as features
    ap_selection: str.
        The method to select the a_p's. Default is "rolling".
        Choices are "rolling", "rolling non-overlapped" and "random".
        "rolling": selects the first n_ap a_p's, then the next range of n_ap a_p's, etc. 
            The rolling window will overlap by "rolling_jump" argument amount of a_p's. 
            e.g. if rolling_jump is 2 and a_ap = 4, then first selection is [a_2, a_7], and next selection is [a_5, a_13], etc.
        "rolling non-overlapped": selects the first n_ap a_p's, then the next range of n_ap a_p's, etc.
            e.g. [a_2, a_5], then [a_7, a_13], etc.
    rolling_jump: int.
        The amount of a_p's to not overlap in the rolling window. Default is 10.
    model: class.
        your chosen model to train and test
    step_size: int. 
        The step size to increment the number of a_p's by
    metric: function. 
        The metric to use to evaluate the model. Default is accuracy_score.
    test_ratio: float. 
        The ratio of the test set size to the training set size. Default is 0.25.
    if_using_cond: bool. 
        If True, the model will use the number of conductors as a feature. Default is False.
    shuffle: bool.
        If True, the data will be shuffled before splitting into training and testing sets. Default is True.
    random_state: int.
        The random seed to use for train test split. Default is 42.
    '''

    print('*'*50)
    print(f"Generating the accuracy by the a_p ranges dataframe for curves with condutor range [{lower_bound}, {upper_bound}]..")

    # slice the dataframe accodring to 
    sliced_df = sliced_data(df, lower_bound, upper_bound)
    print(f"There are {len(sliced_df)} curves within the conductor range [{lower_bound}, {upper_bound}].")

    # do we want conductors as a feature?
    if if_using_cond == False:
        sliced_df = sliced_df.drop(columns = ['conductor'])
    else:
        # gotta normilize the conductor column since it's too big:
        # log the conductor column first
        # then normalize the conductor column by dividing by the log of max value
        sliced_df['conductor'] = np.log(sliced_df['conductor'])/np.log(sliced_df['conductor'].max())

    # create a dataframe to store the number of a_p's and the accuracy
    res_df = pd.DataFrame(columns = ['a_p range', 'performance'])

    # iterate through the number of a_p's
    # first get the start of a_p ranges
    tot_n_aps = len(sliced_df.columns) - 2
    if ap_selection == "rolling":
        apStart_list = [i for i in range(0, tot_n_aps-n_ap+stride, stride)]
    elif ap_selection == "rolling non-overlapped":
        apStart_list = [i for i in range(0, tot_n_aps-n_ap+stride, n_ap)]

    # iterate through the starts of a_p ranges
    for ap_start in apStart_list:
        # slice the dataframe to have i a_p's
        # add the conductor column if we are using it
        ap_end = min(ap_start + n_ap,tot_n_aps + 1)
        if if_using_cond == False:
            cur_df = sliced_df.iloc[:, ap_start:ap_end].join(sliced_df['rank'])
        else:
            cur_df = sliced_df.iloc[:, ap_start:ap_end].join(sliced_df[['conductor','rank']])

        # get the metric result of the model within test data
        res = getRes(cur_df, model, metric, test_ratio, shuffle, random_state)

        # append the metric result to the dataframe
        res_df = pd.concat([res_df, pd.DataFrame({'a_p range': f"[{ap_start},{ap_end}]", 'performance': res}, index = [0])], ignore_index = True)

    return res_df

def plot_AccuracyByApRange(res_dict,  metric_name = 'Accuracy', size=(12, 6)):
    plt.figure(figsize=size)
    for bounds, acc_df in res_dict.items():
        lower_bound, upper_bound = bounds
        plt.plot(acc_df['a_p range'], acc_df['performance'], label=f'Bounds: {lower_bound} to {upper_bound}')    

    # if metric_name = matthews_corrcoef, change it to MCC
    if metric_name == 'matthews_corrcoef':
        metric_name = 'MCC'  
            
    plt.title('{} by Range of ap\'s for Different Conductor Bounds'.format(metric_name))
    plt.xlabel('a_p Range')
    plt.ylabel(metric_name)
    plt.legend()  # Show legend to identify the lines
    plt.tight_layout()
    plt.show()

def find_min_num_a_p_for_accuracy_thresholds(res_df):
    accuracy_thresholds = [0.99, 0.98, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    threshold_to_min_num_a_p = {}
    
    for threshold in accuracy_thresholds:
        # Filter rows where accuracy is greater than or equal to the current threshold
        filtered_df = res_df[res_df['accuracy'] >= threshold]
        
        # Find the minimum 'num_a_p' if any rows meet the condition
        if not filtered_df.empty:
            min_num_a_p = filtered_df['num_a_p'].min()
            threshold_to_min_num_a_p[threshold] = min_num_a_p
        else:
            # Assign a default value if no rows meet the condition
            threshold_to_min_num_a_p[threshold] = np.nan
    
    return threshold_to_min_num_a_p

# def plot_side_by_side(bounds_list, df, model, step_size):
#     # Initialize subplots
#     fig, axs = plt.subplots(1, len(bounds_list), figsize=(5 * len(bounds_list), 5), sharey=True)
    
#     # Check if there's only one plot to adjust indexing accordingly
#     if len(bounds_list) == 1:
#         axs = [axs]
    
#     # Iterate through each bounds pair and plot
#     for idx, bounds in enumerate(bounds_list):
#         lower_bound, upper_bound = bounds
#         # Generate the DataFrame
#         acc_df = Generate_AccByNumAps_df(df, lower_bound, upper_bound, model, step_size=step_size)
#         # Plot on the respective subplot
#         axs[idx].plot(acc_df['num_a_p'], acc_df['accuracy'])
#         axs[idx].set_title(f'Bounds: {lower_bound} to {upper_bound}')
#         axs[idx].set_xlabel('Number of APs')
#         if idx == 0:
#             axs[idx].set_ylabel('Accuracy')
    
#     plt.tight_layout()
#     plt.show()

def plot_on_same_graph(bounds_list, df, model, step_size):
    plt.figure(figsize=(10, 6))  # Initialize the plot with a specified figure size
    
    for bounds in bounds_list:
        lower_bound, upper_bound = bounds
        # Generate the DataFrame
        acc_df = Generate_AccByNumAps_df(df, lower_bound, upper_bound, model, step_size=step_size)
        # Plot on the same graph
        plt.plot(acc_df['num_a_p'], acc_df['performance'], label=f'Bounds: {lower_bound} to {upper_bound}')
        
    plt.title('Accuracy by Number of APs for Different Bounds')
    plt.xlabel('Number of APs')
    plt.ylabel('Performance')
    plt.legend()  # Show legend to identify the lines
    plt.tight_layout()
    plt.show()

# convert PARI Kodaira symbol encoding: disregard n in I_n and I_n* 
def normalise_kodaira_symbol(ks_list):
    output = []
    for ks in ks_list:
        # if the Kodaira symbol is I_n or I_n*, we can disregard the n
        if ks >= 5:
            output.append(5)    # Kodaira symbol I_n: multiplicative reduction with ν(j) = -n
        elif ks <= -5:
            output.append(-5)   # Kodaira symbol I_n*: potential multiplicative reduction with ν(j) = -n
        else:
            output.append(ks)   # potential good reduction
    return output

# process the kodaira symbol column to normalise the values
# we convert PARI Kodaira symbol encoding to disregard the n in I_n and I_n* 
def process_kodaira_symbol(df):
    # Step 0: disregard n in I_n and I_n*
    df['kodaira_symbols'] = df['kodaira_symbols'].apply(normalise_kodaira_symbol).apply(np.unique)
    # double check if any good reudction data is there
    contains_1 = df['kodaira_symbols'].apply(lambda x: 1 in x)
    if contains_1.any():
        print(f"Found curves with Kodaira symbol I_1 in the dataset. The number of curves with good reduction is {contains_1.sum()}. Please double check your dataset.")
        return 
    # Step 1: Split the lists into separate rows
    df_split = df['kodaira_symbols'].apply(pd.Series)
    # Step 2: Stack the DataFrame to get a Series with a MultiIndex
    df_split = df_split.stack()
    # Step 3: Perform one-hot encoding
    df_dummies = pd.get_dummies(df_split, prefix='kodaira')
    # Step 4: Sum the DataFrame level-wise
    df_dummies = df_dummies.sum(level=0)
    # Step 5: Join the original DataFrame with the one-hot encoded DataFrame
    df = df.join(df_dummies)
    df.drop('kodaira_symbols', axis=1, inplace=True)
    return df