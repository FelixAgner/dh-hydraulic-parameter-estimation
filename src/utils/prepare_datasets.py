# Prepare compiled data sets from the raw experimental data
import pandas as pd
import numpy as np
import os

def filter_realistic(file, step_length):
    data = pd.read_csv("data/raw_data/realistic/" + file)
    # drop first 1000 samples which correspond to a start-up-period
    data = data.drop(range(1000))
    # rearrange the index to start at 0
    l = len(data)
    data.index = np.arange(l)
    # compute the mean over step_length samples and drop the last sample
    data = data.groupby(np.arange(l) // step_length).mean()
    data = data.drop(data.index[-1])
    return data

def filter_exciting(file, drop_first, step_length):
    data = pd.read_csv("data/raw_data/exciting/" + file)

    # drop first 20 samples which correspond to a start-up-period.
    data_mean = pd.concat(
        [data.iloc[i+drop_first:i+step_length].mean() for i in range(20, len(data), step_length)], 
        axis=1).transpose()
    
    # drop the last sample to avoid nan values due to mismatch in samle size
    data_mean = data_mean.drop(data_mean.index[-1])
    return data_mean

def make_filtered_data_set(method, step_length, drop_first=10):
    if method == "realistic":
        prep_method = lambda file: filter_realistic(file, step_length)
    elif method == "exciting":
        prep_method = lambda file: filter_exciting(file, drop_first, step_length)
    else:
        raise ValueError("Invalid method: " + method)
    
    # load consumer data and drop first 1000 samples which correspond to a start-up-period
    consumer_1 = prep_method('consumer_1.csv')
    consumer_2 = prep_method('consumer_2.csv')
    consumer_3 = prep_method('consumer_3.csv')
    consumer_4 = prep_method('consumer_4.csv')

    # load pipe data and compute mean over step_length samples
    supply_pipe = prep_method('pipe_20.csv')
    return_pipe = prep_method('pipe_24.csv')

    # pumping station
    pumping_station = prep_method('pump_41.csv')

    data = pd.DataFrame({
        "q0": consumer_1["q_pipe"],
        "q1": consumer_2["q_pipe"],
        "q2": consumer_3["q_pipe"],
        "q3": consumer_4["q_pipe"],
        "q4": consumer_1["q_pipe"] + consumer_2["q_pipe"] + consumer_3["q_pipe"] + consumer_4["q_pipe"],
        "q5": consumer_2["q_pipe"] + consumer_3["q_pipe"] + consumer_4["q_pipe"],
        "q6": consumer_3["q_pipe"] + consumer_4["q_pipe"],
        "v0": consumer_1["v"] / 100, # normalize valve set-points to range 0-1
        "v1": consumer_2["v"] / 100,
        "v2": consumer_3["v"] / 100,
        "v3": consumer_4["v"] / 100,
        "ps_pump": pumping_station["p3_3"],
        "pr_pump": pumping_station["p3_1"],
        "dp_pump": pumping_station["p3_3"] - pumping_station["p3_1"],
        "dp0": supply_pipe["p1_2"] - return_pipe["p1_2"],
        "dp1": supply_pipe["p2_2"] - return_pipe["p2_2"],
        "dp2": supply_pipe["p3_2"] - return_pipe["p3_2"],
        "dp3": supply_pipe["p4_2"] - return_pipe["p4_2"],
    })

    if method == "realistic":
        # add a time column in real world hours from 0 to 24*14
        data["time"] = [i * step_length / 1200 for i in range(len(data))]
        # add a lab time column, incremented by step_length * 0.1 seconds
        data["lab_time_seconds"] = [i * step_length * 0.1 for i in range(len(data))]
        data["lab_time_minutes"] = [i * step_length * 0.1 / 60 for i in range(len(data))]
        # add reference column
        data["qr0"] = consumer_1["qr"]
        data["qr1"] = consumer_2["qr"]
        data["qr2"] = consumer_3["qr"]
        data["qr3"] = consumer_4["qr"]
    elif method == "exciting":
        # make the time stamps the sample number
        data["time"] = np.arange(len(data))
    
    # name the index column
    data.index.name = "sample"
    return data

def run():
    # filter the realistic data set
    step_length = 300
    # step lengths:
    # 100 corresponds to 5 minutes real life time
    # 300 corresponds to 15 minutes real life time
    # 600 corresponds to 30 minutes real life time
    # 1200 corresponds to 1 hour real life time
    print("Preparing realistic data set.")
    data = make_filtered_data_set("realistic", step_length)
    if os.path.exists("data/realistic.csv"):
        print("data/realistic.csv already exists. Overwrite? (y/n)")
        if input() != "y":
            print("Aborting.")
        else:
            print("Saving realistic data set to data/realistic.csv")
            data.to_csv("data/realistic.csv")
    else:
        print("Saving realistic data set to data/realistic.csv")
        data.to_csv("data/realistic.csv")

    # filter the exciting data set
    step_length = 40
    drop_first = 10
    print("Preparing exciting data set.")
    data = make_filtered_data_set("exciting", step_length, drop_first)
    if os.path.exists("data/exciting.csv"):
        print("data/exciting.csv already exists. Overwrite? (y/n)")
        if input() != "y":
            print("Aborting.")
        else:
            print("Saving exciting data set to data/exciting.csv")
            data.to_csv("data/exciting.csv")
    else:
        print("Saving exciting data set to data/exciting.csv")
        data.to_csv("data/exciting.csv")

if __name__ == "__main__":
    run(overwrite=True)