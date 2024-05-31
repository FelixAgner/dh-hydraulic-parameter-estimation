import numpy as np
import pandas as pd
import pickle
import os
import logging

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(ROOT_DIR), "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

def load_data(data_name):
    file_name = os.path.join(DATA_DIR, f"{data_name}.csv")
    return pd.read_csv(file_name)

def ramp(x, a, b, tol = 1e-8):
    """
    Ramp function
    * x < a: 0 
    * a < x < b: (x-a)/(b-a)
    * x > b: 1
    * tol: Tolerance for x < a to avoid division by 0
    """
    if b < a:
        a, b = b, a
    return min(max(x-a+tol, tol), b-a) / (b-a)

def load_model(name, training_data):

    filename = os.path.join(MODEL_DIR, "parameters", f"{name}_{training_data}.pkl")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such model {name}_{training_data}.pkl.")

    logging.debug(f"Loading model from {filename}.")
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def save_model(model, training_data, overwrite = False):
    name = model["settings"]["name"]
    filename = os.path.join(MODEL_DIR, "parameters", f"{name}_{training_data}.pkl")
    # Check if file exists
    if not os.path.exists(filename):
        logging.info("File does not exist. Saving...")
    elif overwrite:
        logging.info(f"Overwriting existing file {filename}.")
    else:
        print(f"Model with name {name}_{training_data}.pkl already exists. Overwrite? (y/n)")
        if input() != "y":
            print("Aborting.")
            return
        

    logging.debug(f"Saving model to {filename}.")
    logging.debug(f"Model: {model}")
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print("Model saved.")

def save_results(results, model_name, training_data, test_data, overwrite = False):
    filename = os.path.join(DATA_DIR, "results", f"{model_name}_{training_data}_{test_data}.csv")

    # Check if file exists
    if not os.path.exists(filename):
        logging.info(f"File {filename} does not exist. Saving...")
    elif overwrite:
        logging.info(f"Overwriting existing file {filename}.")
    else:
        print(f"Results with name {model_name}_{training_data}_{test_data}.csv already exists. Overwrite? (y/n)")
        if input() != "y":
            print("Aborting.")
            return
        
    results.to_csv(filename)

def load_results(model_name, training_data, test_data):
    filename = os.path.join(DATA_DIR, "results", f"{model_name}_{training_data}_{test_data}.csv")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such results {model_name}_{training_data}_{test_data}.csv.")

    results = pd.read_csv(filename)
    return results

# pipe-map:
pipemap = [ # Maps valve indices to pipes in their loops
    [0,4],
    [1,4,5],
    [2,4,5,6],
    [3,4,5,6]
]

def blkdiag(vectors):
    """
    Make a block diagonal matrix from a list of vectors
    """
    n = len(vectors)
    m = len(vectors[0])
    result = np.zeros((n, n*m))
    for i in range(n):
        result[i, i*m:(i+1)*m] = vectors[i]
    return result

def get_all_models():
    files = os.listdir(MODEL_DIR)
    # filter out all files ending with .json and remove the extension
    models = [f[:-5] for f in files if f[-5:] == ".json"]
    return models

def print_model(model, print_settings = True, print_theta = True, print_s = True):
    if print_settings:
        print("Settings:")
        for k, v in model["settings"].items():
            print(f"{k}:\t{v}")
    
    if print_theta:
        print("Valve parameters theta:")
        for v in range(4):
            theta = model["theta"][v]
            print(f"Valve {v}:\t{[round(x, 4) for x in theta]}")
        
    if print_s:        
        print("Pipe parameters s:")
        s = model["s"]
        for i in range(7):
            print(f"Pipe {i}:\t{round(s[i], 6)}")
    
def save_data_matrices(settings, data_set, phi, y):
    name = settings["name"]
    dirname = os.path.join(DATA_DIR, "data_matrices", f"{name}_{data_set}")
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    with open(f"{dirname}/phi.pkl", "wb") as f:
        pickle.dump(phi, f)

    with open(f"{dirname}/y.pkl", "wb") as f:
        pickle.dump(y, f)

def load_data_matrices(model_name, training_data):
    dirname = os.path.join(DATA_DIR, "data_matrices", f"{model_name}_{training_data}")
    with open(os.path.join(dirname, "phi.pkl"), "rb") as f:
        phi = pickle.load(f)
    
    with open(os.path.join(dirname, "y.pkl"), "rb") as f:
        y = pickle.load(f)

    return phi, y

def print_data_stats(data_set):
    
    print("-------------------------------------------------------")
    print(f"Data set: {data_set}")
    print("-------------------------------------------------------")
    data = load_data(data_set)
    training_data = data[:int(0.7 * len(data))]

    print("Data set size:")
    print(f"Total number of samples: {len(data)}")
    print(f"Training data (70 %): {int(0.7 * len(data))} samples")
    print(f"Test data (30 %): {len(data) - int(0.7 * len(data))} samples")

    print(f"Min, 5th quantile, mean, 95th quantile and max flow rates in {data_set} data set:")
    for i in range(4):
        vals = [
            np.round(data[f'q{i}'].min(), 3),
            np.round(data[f'q{i}'].quantile(0.05), 3),
            np.round(data[f'q{i}'].mean(), 3),
            np.round(data[f'q{i}'].quantile(0.95), 3),
            np.round(data[f'q{i}'].max(), 3)
        ]

        print(f"Full data, flow rate {i}: \t{vals[0]}, \t{vals[1]}, \t{vals[2]}, \t{vals[3]}, \t{vals[4]}")
    
    for i in range(4):
        vals = [
            np.round(training_data[f'q{i}'].min(), 3),
            np.round(training_data[f'q{i}'].quantile(0.05), 3),
            np.round(training_data[f'q{i}'].mean(), 3),
            np.round(training_data[f'q{i}'].quantile(0.95), 3),
            np.round(training_data[f'q{i}'].max(), 3)
        ]
        print(f"Training data, flow rate {i}: \t{vals[0]}, \t{vals[1]}, \t{vals[2]}, \t{vals[3]}, \t{vals[4]}")
    
    print(f"Min, 5th quantile, mean, 95th quantile and max valve positions in {data_set} data set:")
    for i in range(4):
        vals = [
            np.round(data[f'v{i}'].min(), 3),
            np.round(data[f'v{i}'].quantile(0.05), 3),
            np.round(data[f'v{i}'].mean(), 3),
            np.round(data[f'v{i}'].quantile(0.95), 3),
            np.round(data[f'v{i}'].max(), 3)
        ]

        print(f"Full data, valve {i}: \t{vals[0]}, \t{vals[1]}, \t{vals[2]}, \t{vals[3]}, \t{vals[4]}")
    
    training_data = data[:int(0.7 * len(data))]
    for i in range(4):
        vals = [
            np.round(training_data[f'v{i}'].min(), 3),
            np.round(training_data[f'v{i}'].quantile(0.05), 3),
            np.round(training_data[f'v{i}'].mean(), 3),
            np.round(training_data[f'v{i}'].quantile(0.95), 3),
            np.round(training_data[f'v{i}'].max(), 3)
        ]
        print(f"Training data, valve {i}: \t{vals[0]}, \t{vals[1]}, \t{vals[2]}, \t{vals[3]}, \t{vals[4]}")
