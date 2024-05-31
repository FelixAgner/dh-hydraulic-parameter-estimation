import numpy as np
import pandas as pd
from utils.utils import ramp

RAMP_PARAMETERS = {
    "a": [0.10, 0.15, 0.20, 0.25],
    "b": [0.80, 0.85, 0.90, 0.95, 1.0],
    "c": [2.0, 2.5, 3.0]
}

def load_parameterization(settings):
    """
    Loads the parameterization function
    """
    model_map = {
        "linear": f_linear,
        "ramps": f_ramps
    }
    parameterization = settings["parameterization"]
    if parameterization not in model_map:
        raise ValueError(f"Invalid model parameterization {parameterization}.")
    
    return model_map[parameterization]

def f_linear(v):
    """
    naive valve basis function
    """
    return np.array([1 / (v**2)])

def f_ramps(v):
    """
    Basis function for valve parameterization using ramp functions
    """
    return np.array([ 1 / ramp(v, a, b)**(c) for a in RAMP_PARAMETERS["a"] for b in RAMP_PARAMETERS["b"] for c in RAMP_PARAMETERS["c"]])


def print_curves(model):

    if model["settings"]["parameterization"] == "linear":
        funcs = [ "1 / v^2" ]
    elif model["settings"]["parameterization"] == "ramps":
        funcs = [ f"/ramp(v, {round(a,3)}, {round(b,3)})^{round(c,3)}" for a in RAMP_PARAMETERS["a"] for b in RAMP_PARAMETERS["b"] for c in RAMP_PARAMETERS["c"]]
    else:
        raise ValueError(f"Invalid model parameterization {model['settings']['parameterization']}.")
    
    for i in range(4):
        print(f"Valve {i}:")
        terms = [
            str( round(model["theta"][i][j], 4)) + funcs[j] for j in range(len(model["theta"][i])) if model["theta"][i][j] > 0 
        ]
        print(" + ".join(terms))

def valve_equivalent_resistances(model):
    """
    Returns a function for evaluating the equivalent resistances in the valves
    """
    theta = model["theta"]
    valve_features = load_parameterization(model["settings"])
    return lambda v : np.array([
        np.dot(valve_features(v[i]), theta[i]) for i in range(4)
        ])

def valve_curve(model, n_points=100):
    """
    generate values for plotting the valve curve of a model
    """
    s_valves = valve_equivalent_resistances(model)
    v_values = np.linspace(0, 1, n_points)
    kv = [[s**(-1/model["settings"]["flow rate exponent"]) for s in s_valves([v,v,v,v])] for v in v_values]
    kv = pd.DataFrame(kv, columns=[f'kv{i}' for i in range(4)])
    kv["v"] = v_values
    return kv

