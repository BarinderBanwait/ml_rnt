# python version 3.10.7
# scikit-learn == 1.1.2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# load the dataset
def load_data(path):
    df = pd.read_parquet(path)
    print(f"Loaded the big dataset with {len(df.columns) - 2} a_p's and {len(df)} curves..")
    return df

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
    res_df = pd.DataFrame(columns = ['num_a_p', 'accuracy'])

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
        res_df = pd.concat([res_df, pd.DataFrame({'num_a_p': i, 'accuracy': res}, index = [0])], ignore_index = True)

    return res_df


def Generate_AccByApRange_df(df, lower_bound, upper_bound, model, n_ap, ap_selection = "rolling", rolling_jump = 10, metric = accuracy_score, test_ratio = 0.25, if_using_cond = False, shuffle = True, random_state = 42):
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
        apStart_list = [i for i in range(0, tot_n_aps-n_ap+rolling_jump, rolling_jump)]
    elif ap_selection == "rolling non-overlapped":
        apStart_list = [i for i in range(0, tot_n_aps-n_ap+rolling_jump, n_ap)]

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

def plot_AccByNumAps(res_df, lower_bound, upper_bound):
    '''
    This function plots the accuracy by the number of a_p's

    Parameters:
    res_df: pd.DataFrame.
        The dataframe containing the number of a_p's and the accuracy
    '''

    plt.plot(res_df['num_a_p'], res_df['accuracy'])
    plt.xlabel('Number of a_p\'s')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by the Number of a_p\'s for conductor range [{}, {}]'.format(lower_bound, upper_bound))
    plt.show()

def plot_AccByApRange(res_df, lower_bound, upper_bound):
    '''
    This function plots the accuracy by the a_p ranges

    Parameters:
    res_df: pd.DataFrame.
        The dataframe containing the a_p ranges and the accuracy
    '''

    plt.plot(res_df['a_p range'], res_df['performance'])
    plt.xlabel('a_p\' range')
    plt.ylabel('Performance')
    plt.title('Accuracy by the a_p ranges for conductor range [{}, {}]'.format(lower_bound, upper_bound))
    plt.show()


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
        plt.plot(acc_df['num_a_p'], acc_df['accuracy'], label=f'Bounds: {lower_bound} to {upper_bound}')
        
    
    plt.title('Accuracy by Number of APs for Different Bounds')
    plt.xlabel('Number of APs')
    plt.ylabel('Accuracy')
    plt.legend()  # Show legend to identify the lines
    plt.tight_layout()
    plt.show()

def plot_AccuracyByApRange(res_dict):
    for bounds, acc_df in res_dict.items():
        lower_bound, upper_bound = bounds
        plt.plot(acc_df['a_p range'], acc_df['performance'], label=f'Bounds: {lower_bound} to {upper_bound}')      
            
    plt.title('Performance by Range of a_p for Different Bounds')
    plt.xlabel('a_p Range')
    plt.ylabel('Accuracy')
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