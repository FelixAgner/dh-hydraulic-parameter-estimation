import numpy as np
import pandas as pd
from utils.utils import *
from utils.hysteresis import append_hysteresis
from utils.parameterization import valve_equivalent_resistances

def predict_flow_rates(row, s_valves, s, gamma = 2):
    """
    Predict the flow rates q0, q1, q2, q3 for one row of the data. 
    s_valve(row): function for evaluating the equivalent resistances in the valves
    s: the pipe parameters
    gamma: the exponent in the flow rate model (ususally 2)
    """
    # Split the resistances into grid-level (pipes 4, 5, 6) and branches (pipes + valves going through substations)
    s_branch = s_valves([row["vh0"], row["vh1"], row["vh2"], row["vh3"]]) + 2 * np.array(s[0:4])
    s_grid = 2 * np.array(s[4:7])
    
    s_hat = find_equivalent_resistance(s_grid, s_branch, gamma)
    q_hat = np.zeros(4)
    q0 = (row['dp_pump'] / s_hat[0])**(1/gamma)
    for i in range(3):
        q_hat[i] = q0 * (s_hat[i+1])**(1/gamma) / ((s_branch[i])**(1/gamma) + (s_hat[i+1])**(1/gamma))
        q0 -= q_hat[i]
    q_hat[-1] = q0

    # store as pd series with qhat0, qhat1,...
    return pd.Series( {f'qhat{i}' : q_hat[i] for i in range(4)} )

def find_equivalent_resistance(s_grid, s_branch, gamma):
    """find equivalent resistance s_hat for one row of the model data""" 
    s_hat = np.zeros(4)
    s_hat[-1] = s_branch[-1]
    # iterate backwards from 2 to 0
    for i in range(2,-1,-1):
        s_hat[i] = s_grid[i] + s_branch[i]*s_hat[i+1] / (((s_branch[i])**(1/gamma) + (s_hat[i+1])**(1/gamma))**gamma )

    return s_hat

def evaluate_model(model, test_data, is_training_data):
    data = load_data(test_data)
    
    s = model["s"]
    # function for evaluating equivalent resistance in valves
    s_valves = valve_equivalent_resistances(model)

    # append hysteresis-compensation in columns "vh{i}"
    data = append_hysteresis(data, model["settings"]["hysteresis percent"])

    # predict flow rates
    results = data.apply(lambda row : predict_flow_rates(row, s_valves, s, model["settings"]["flow rate exponent"]), axis=1)

    # make error-columns
    results["e0"] = data["q0"] - results["qhat0"]
    results["e1"] = data["q1"] - results["qhat1"]
    results["e2"] = data["q2"] - results["qhat2"]
    results["e3"] = data["q3"] - results["qhat3"]

    # name index as "sample"
    results.index.name = "sample"

    # add column to indicate if sample was used in training
    if is_training_data:
        results["training"] = [True if i < int(len(data)*model["settings"]["training data percent"]/100) else False for i in range(len(data))]
    else:
        results["training"] = [False for _ in range(len(data))]

    return results


