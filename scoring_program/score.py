import os
from sys import argv, path
root_dir = os.getcwd()
from environment import Meta_Learning_Environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import base64

#=== Verbose mode
verbose = True

#=== Setup input/output directories
root_dir = os.getcwd()
default_input_dir = os.path.join(root_dir, "sample_data/")
default_output_dir = os.path.join(root_dir, "output/")
default_program_dir = os.path.join(root_dir, "ingestion_program/")
default_submission_dir = os.path.join(root_dir, "sample_code_submission/")

#=== Normalize time as implemented in the AutoML challenge
normalize_t = True
t_0 = 60.0 # Hyperparameters used for computing scaled time (t_tilde). It controls how important performing well at the beginning is.

def vprint(mode, t):
    """
    Print to stdout, only if in verbose mode.

    Parameters
    ----------
    mode : bool
        True if the verbose mode is on, False otherwise.

    Examples
    --------
    >>> vprint(True, "hello world")
    hello world

    >>> vprint(False, "hello world")

    """

    if(mode):
        print(str(t))

def visualize_agent_learning_curve(dataset_name, total_time_budget, df, alc):
    """
    Visualize agent's learning curves on the test set.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset at hand.
    total_time_budget : float
        Total time budget given to the agent for searching algorithms on the given dataset.
    df : Dataframe
        Data for plotting the learning curve, with columns = ['algo_index', 'algo_time_spent', 'total_time_spent', 'score']

    """

    plt.title(label = "Dataset: " + dataset_name)
    plt.step(df['total_time_spent'], df['best_score'], where='post', label = dataset_name)
    plt.scatter(df['total_time_spent'], df['best_score'], s=8, marker="D")
    L=plt.legend()
    L.get_texts()[0].set_text('ALC = ' + str(np.around(alc, decimals=2)))
    plt.ylim(0, 1.05)
    plt.ylabel('test score')
    plt.fill_between(df['total_time_spent'], df['best_score'], step="post", alpha=0.4, color='dodgerblue')
    plt.grid()

    #=== Normalize time on the plot
    if normalize_t:
        plt.xlim(0, 1)
        plt.xlabel('normalized time')
        plt.text(1.0, df.iloc[-1]['best_score'], str(df.iloc[-1]['best_score']), fontdict=None)
    else:
        plt.xlim(0, total_time_budget)
        plt.xlabel('time (s)')
        plt.text(df['total_time_spent'].max(), df.iloc[-1]['best_score'], str(df.iloc[-1]['best_score']), fontdict=None)

    #=== Save figure and clear the plot
    plt.savefig(output_visualizations_dir + dataset_name + ".png", dpi=120, format='png', bbox_inches='tight')
    plt.cla()
    plt.clf()

def write_scores_html(output_dir, output_visualizations_dir, auto_refresh=True, append=False):
    filename = 'scores.html'
    image_paths = glob(os.path.join(output_visualizations_dir, '*.png'))

    try: # Try to sort by numerical values
      image_paths = sorted(image_paths,key=lambda x: int(x.split('/')[-1].split('.')[0]))
    except:
      image_paths = sorted(image_paths)

    if auto_refresh:
      html_head = '<html><head> <meta http-equiv="refresh" content="5"> ' +\
                  '</head><body><pre>'
    else:
      html_head = """<html><body><pre>"""
    html_end = '</pre></body></html>'
    if append:
      mode = 'a'
    else:
      mode = 'w'
    filepath = os.path.join(output_dir, filename)
    with open(filepath, mode) as html_file:
        html_file.write(html_head)
        for image_path in image_paths:
          with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            encoded_string = encoded_string.decode('utf-8')
            s = '<img src="data:image/png;charset=utf-8;base64,{}"/>'\
                .format(encoded_string)
            html_file.write(s + '<br>')
        html_file.write(html_end)
    vprint(verbose, "\n[+] Write agent's learning curve visualizations to {}".format(filepath))

def normalize_time(df, total_time_budget, t_0):
    """
    Normalize time using log functions and t_0.

    Parameters
    ----------
    df : Dataframe
        Learning curve data with 'total_time_spent' column to be normalized.
    total_time_budget : float
        Total time budget for searching algorithms on the dataset at hand
    t_0 : float
        Hyperparameter controlling how important performing well at the beginning is.

    Returns
    ----------
    df : Dataframe
        The dataframe after normalizing the time.

    Examples
    ----------

    >>> df
        algo_index  algo_time_spent  total_time_spent  score
    0            9            36.90               140        0.72
    1            9            73.44               180        0.72
    2            9            73.44               190        0.72

    >>> normalize_time(df, 1200, 60)
        algo_index  algo_time_spent  total_time_spent  score
    0            9            36.90          0.395455        0.72
    1            9            73.44          0.455340        0.72
    2            9            73.44          0.468749        0.72

    """

    df['total_time_spent'] = df['total_time_spent'].apply(lambda x: np.log(1+x/t_0)/np.log(1+total_time_budget/t_0))

    return df

def reveal_score_on_the_test_set(env, dataset_name, df):
    """
    Query the scores on the learning curve on the test set

    Parameters
    ----------
    env : Meta_Learning_Environment
        The environment which contains all the learning curve data.
    dataset_name : str
        Name of the dataset at hand
    df : Dataframe
        Agent's learning curve data

    Returns
    ----------
    df : Dataframe
        The dataframe with the updated 'score' column.

    Examples
    ----------
    >>> df
             algo_index  algo_time_spent  total_time_spent  score
    0            9            15.28                70         NaN
    1            9            34.50               100         NaN
    2            9            34.50               180         NaN

    >>> reveal_score_on_the_test_set(env, dataset_name, df)
             algo_index  algo_time_spent  total_time_spent  score   best_score
    0            9            15.28                70        0.35       0.35
    1            9            34.50               100        0.41       0.41
    2            9            34.50               180        0.41       0.41

    """
    #=== Iterate through each timestamp and compute the corresponding score on the test learning curve
    best_score = 0.0
    for i, row in df.iterrows():
        algo_time_spent = row['algo_time_spent']
        if algo_time_spent==0:
            score = 0.0
        else:
            algo_index = row['algo_index']
            #=== Get the learning curve of the algorithm on the dataset at hand
            lc = env.test_learning_curves[dataset_name][algo_index]
            index = list(lc.timestamps).index(algo_time_spent)
            score = lc.scores[index]
        best_score = max(best_score, score)

        #=== Update the test scores in the dataframe
        df.at[i,'score'] = score
        df.at[i,'best_score'] = best_score
    return df

def reveal_score_on_the_validation_set(env, dataset_name, df):
    """
    Query the scores on the learning curve on the validation set

    Parameters
    ----------
    env : Meta_Learning_Environment
        The environment which contains all the learning curve data.
    dataset_name : str
        Name of the dataset at hand
    df : Dataframe
        Agent's learning curve data

    Returns
    ----------
    df : Dataframe
        The dataframe with the updated 'score' column.

    Examples
    ----------
    >>> df
             algo_index  algo_time_spent  total_time_spent  score
    0            9            15.28                70         NaN
    1            9            34.50               100         NaN
    2            9            34.50               180         NaN

    >>> reveal_score_on_the_validation_set(env, dataset_name, df)
             algo_index  algo_time_spent  total_time_spent  score   best_score
    0            9            15.28                70        0.39       0.39
    1            9            34.50               100        0.44       0.44
    2            9            34.50               180        0.44       0.44

    """
    #=== Iterate through each timestamp and compute the corresponding score on the test learning curve
    best_score = 0.0
    for i, row in df.iterrows():
        algo_time_spent = row['algo_time_spent']
        #=== Get the test score associated with the timestamp
        if algo_time_spent==0:
            score = 0.0
        else:
            algo_index = row['algo_index']
            #=== Get the learning curve of the algorithm on the dataset at hand
            lc = env.validation_learning_curves[dataset_name][algo_index]
            index = list(lc.timestamps).index(algo_time_spent)
            score = lc.scores[index]
        best_score = max(best_score, score)

        #=== Update the test scores in the dataframe
        df.at[i,'score'] = score
        df.at[i,'best_score'] = best_score

    return df

def compute_ALC(df, total_time_budget):
    """
    Compute the area under the Learning Curve

    Parameters
    ----------
    df : Dataframe
        Learning curve data.
    total_time_budget : float
        Total time budget for searching algorithms on the dataset at hand

    Returns
    ----------
    average_alc : float

    Examples
    ----------
    >>> df
             algo_index  algo_time_spent  total_time_spent  score   best_score
    0            9            15.28          0.253961        0.39       0.39
    1            9            34.50          0.322162        0.44       0.44
    2            9            34.50          0.455340        0.44
    ...
    26           9           151.73          1.000000        0.49       0.49

    >>> compute_average_ALC(df, 1200):
    0.34693767966301176

    """

    alc = 0.0
    for i in range(len(df)):
        if i==0:
            if normalize_t:
                alc += df.iloc[i]['best_score'] * (1-df.iloc[i]['total_time_spent'])
            else:
                alc += df.iloc[i]['best_score'] * (total_time_budget-df.iloc[i]['total_time_spent'])
        elif i>0:
            if normalize_t:
                alc += (df.iloc[i]['best_score']-df.iloc[i-1]['best_score']) * (1-df.iloc[i]['total_time_spent'])
            else:
                alc += (df.iloc[i]['best_score']-df.iloc[i-1]['best_score']) * (total_time_budget-df.iloc[i]['total_time_spent'])
    return alc

if __name__ == "__main__":
    #=== Get input and output directories
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        validation_data_dir = os.path.join(input_dir, 'valid')
        test_data_dir = os.path.join(input_dir, 'test')
        meta_features_dir = os.path.join(input_dir, 'dataset_meta_features')
        algorithms_meta_features_dir = os.path.join(input_dir, 'algorithms_meta_features')
        output_from_ingestion_program_dir = output_dir # Output from the ingestion_program
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])
        validation_data_dir = os.path.join(input_dir, 'ref/valid')
        test_data_dir = os.path.join(input_dir, 'ref/test')
        meta_features_dir = os.path.join(input_dir, 'ref/dataset_meta_features')
        algorithms_meta_features_dir = os.path.join(input_dir, 'ref/algorithms_meta_features')
        output_from_ingestion_program_dir = os.path.join(input_dir, 'res') # Output from the ingestion_program

    vprint(verbose, "Using input_dir: " + input_dir)
    vprint(verbose, "Using output_dir: " + output_dir)
    vprint(verbose, "Using validation_data_dir: " + validation_data_dir)
    vprint(verbose, "Using test_data_dir: " + test_data_dir)
    vprint(verbose, "Using meta_features_dir: " + meta_features_dir)
    vprint(verbose, "Using algorithms_meta_features_dir: " + algorithms_meta_features_dir)
    vprint(verbose, "Using output_from_ingestion_program_dir: " + output_from_ingestion_program_dir)

    #=== Directory for visualizations of agent's learning curves
    output_visualizations_dir = os.path.join(output_dir, "output_visualizations/")
    if not os.path.exists(output_visualizations_dir):
        os.makedirs(output_visualizations_dir)

    #=== List of dataset names
    list_datasets = os.listdir(test_data_dir)
    if '.DS_Store' in list_datasets:
        list_datasets.remove('.DS_Store')
    list_datasets.sort()

    #=== List of algorithms
    list_algorithms = os.listdir(os.path.join(test_data_dir, list_datasets[0]))
    if '.DS_Store' in list_algorithms:
        list_algorithms.remove('.DS_Store')
    list_algorithms.sort()

    #=== Create files for storing output scores
    score_file = open(os.path.join(output_dir, 'scores.txt'), 'w')
    html_file = open(os.path.join(output_dir, 'scores.html'), 'w')

    ################## MAIN LOOP ##################
    #=== Create an environment
    env = Meta_Learning_Environment(validation_data_dir, test_data_dir, meta_features_dir, algorithms_meta_features_dir, output_dir)

    #=== As we are using k-fold cross-validation, each dataset is used for testing once
    #=== We iterate through each dataset to compute the agent's test performance on each dataset
    list_final_score, list_alc = [], []
    average_final_score = 0.0
    average_alc = 0.0
    for dataset_name in list_datasets:
        vprint(verbose, "\n#====================== Results on dataset: " + dataset_name + " ======================#")
        alc, final_score = 0.0, 0.0

        #=== Get total_time_budget from meta_features of the dataset
        meta_features = env.meta_features[dataset_name]
        total_time_budget = int(meta_features['time_budget'])

        #=== Read output file from the ingestion program
        #=== The agent's learning curve on a dataset is stored in a csv file
        try:
            output_file = os.path.join(output_from_ingestion_program_dir, dataset_name + '.csv')
            df = pd.read_csv(output_file, names=['algo_index', 'algo_time_spent', 'total_time_spent', 'score'],
                            dtype={'algo_index': str, 'algo_time_spent': float, 'total_time_spent': float, 'score': float})

            #=== Update the dataframe with scores on the learning curve on the validation set
            updated_df = reveal_score_on_the_test_set(env, dataset_name, df)

            #=== Normalizing t
            if normalize_t:
              normalize_time(updated_df, total_time_budget, t_0)

            #=== Get the final score
            final_score = updated_df.iloc[-1]['score']
            vprint(verbose, "final_score = " + str(final_score))
            list_final_score.append(final_score)

            #=== Compute ALC
            alc = compute_ALC(updated_df, total_time_budget)
            vprint(verbose, "alc = " + str(alc))
            list_alc.append(alc)

            #=== Visualization
            visualize_agent_learning_curve(dataset_name, total_time_budget, updated_df, alc)
        except Exception as e:
          vprint(verbose, e)

        # break
    #############################################

    #=== Compute average final score and average ALC
    if len(list_final_score)!=0:
        average_final_score = sum(list_final_score) / len(list_final_score)
    if len(list_alc)!=0:
        average_alc = sum(list_alc) / len(list_alc)

    #=== Write scores.html
    write_scores_html(output_dir, output_visualizations_dir)

    #=== Write out the scores to scores.txt
    score_file.write("average_final_score: " + str(average_final_score) + "\n")
    score_file.write("average_ALC: " + str(average_alc) + "\n")
    vprint(verbose, "\n######################################## FINAL AVERAGE RESULTS ########################################")
    vprint(verbose, "\naverage_ALC = " + str(average_alc))
    vprint(verbose, "\naverage_final_score = " + str(average_final_score))
    vprint(verbose, "\n[+]Finished running scoring program")
