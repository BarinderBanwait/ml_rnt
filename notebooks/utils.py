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

# load the dataset
def load_data(path):
    df = pd.read_parquet(path)
    print(f"Loaded the big dataset with {len(df.columns) - 2} a_p's and {len(df)} curves..")
    return df

# convert the rank column to binary accodint to a threshold
def convert_rank_to_binary(df, threshold):
    df['rank'] = df['rank'].apply(lambda x: 1 if x > threshold else 0)
    print(f'Converted the rank column to binary. The value of 1 means the rank is greater than {threshold}, otherwise 0. Rank counts:')
    print(df['rank'].value_counts().to_frame().rename(columns={'rank': 'count'}))
    return df

def get_input_output_dim(df):
    in_dim = len(df.columns) - 1
    out_dim = df['rank'].nunique()
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

# split the data into training and test sets and use dataloaders to create batches
def prepare_data(data, device, test_size=0.2, batch_size=32, random_state=42, shuffle=True):
    X = data.drop(columns=['rank']).values
    y = data['rank'].values
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    # Split the data into training, validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataset, test_dataset


def plot_train_eval_hist(train_eval_hist, val_eval_hist, size = (12, 6)):
    plt.figure(figsize=size)
    plt.plot(train_eval_hist, label='train evaluation')
    plt.plot(val_eval_hist, label='validation evaluation')
    plt.xlabel('Epochs')
    plt.ylabel('Evaluation Metric')
    plt.title('Evaluation Metric by Epochs')
    plt.legend()  # Show legend to identify the lines
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