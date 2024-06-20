# python version 3.10.7
# scikit-learn == 1.1.2
import pandas as pd
import numpy as np
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

def getRes(sliced_df, model, metric, test_ratio, shuffle):
    '''
    This function takes a the sliced dataframe and returns the metric result of the model accordingto the number of a_p's (in test data)

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
    '''
    X = sliced_df.drop(columns=['rank']).values
    y = sliced_df['rank'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, shuffle = shuffle)

    # train the model
    model.fit(X_train, y_train)

    # return the metric result of the model on test data
    y_test_pred = model.predict(X_test)
    return metric(y_test, y_test_pred)    


def Generate_AccByNumAps_df(df, lower_bound, upper_bound, model, step_size = 10, metric = accuracy_score, test_ratio = 0.25, if_using_cond = False, shuffle = True):
    '''
    This function generates a dataframe of the number of a_p's and the accuracy of the model for a given sliced dataframe

    Parameters:
    df: pd.DataFrame.
        The dataframe to be sliced
    lower_bound: int. 
        The lower bound of the number of conductors
    upper_bound: int.
        The upper bound of the number of conductors
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
    for i in range(step_size, tot_n_aps, step_size):
        # slice the dataframe to have i a_p's
        # add the conductor column if we are using it
        if if_using_cond == False:
            cur_df = sliced_df.iloc[:, :i].join(sliced_df['rank'])
        else:
            cur_df = sliced_df.iloc[:, :i].join(sliced_df[['conductor','rank']])

        # get the metric result of the model within test data
        res = getRes(cur_df, model, metric, test_ratio, shuffle)

        # append the metric result to the dataframe
        res_df = pd.concat([res_df, pd.DataFrame({'num_a_p': i, 'accuracy': res}, index = [0])], ignore_index = True)

    return res_df
