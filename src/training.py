import numpy as np
import cvxpy as cp
from utils.parameterization import load_parameterization
from utils.utils import *
from utils.hysteresis import append_hysteresis

def valve_data(row, settings):
    """
    Compute the data matrix entries which hit the valve parameters for one load condition
    """
    valve_features = load_parameterization(settings)

    # evaluate all of the valve functions and put them in a long vector
    gv =  [row[f'q{i}']**settings["flow rate exponent"] * valve_features(row[f'vh{i}']) for i in range(4) ]
    
    # Split the vector into block diagonal 4 x 4K matrix.
    # the first row hits valve 1, the second valve 2, etc...
    return blkdiag(gv)

def pipe_data(row, settings):
    """
    Computer the data matrix entries which hit the pipe parameters for one load condition
    """
    gp = [
        2 * row[f'q{i}']**settings["flow rate exponent"] for i in range(7)
    ]
    # initiate empty matrix
    pipe_data = np.zeros((4, 7))
    for i in range(4):
        # select the correct pipe indices that lie in the path going through each valve
        pipe_data[i, pipemap[i]] = [ gp[i] for i in pipemap[i]]
    return pipe_data


def make_load_condition_phi(row, settings):
    """
    Compute the 4 rows of Phi which correspond to one load condition (row)
    """
    phi_entry = np.hstack([
        valve_data(row, settings), 
        pipe_data(row, settings), 
        ])
    
    # optionally drop rows where q < threshold
    phi_entry = np.delete(phi_entry, [i for i in range(4) if row[f"q{i}"] < settings["flow rate threshold"]], axis = 0)

    return phi_entry

def make_load_condition_y(row, settings):
    """
    Compute the 4 rows of y which correspond to one load condition (row)
    """
    y = row['dp_pump'] * np.ones((4,1))
    # optionally drop rows where q < threshold
    y = np.delete(y, [i for i in range(4) if row[f"q{i}"] < settings["flow rate threshold"]], axis = 0)
    return y

def make_load_condition_weights(row, settings):
    """
    Compute the 4 rows of optional weights to be used in the optimization problem in order to weigh precision towards higher flow rates
    """
    w = np.array([row[f"vh{i}"] for i in range(4)]).reshape(4,1) - 0.2
    # optionally drop rows where q < threshold
    w = np.delete(w, [i for i in range(4) if row[f"q{i}"] < settings["flow rate threshold"]], axis = 0)
    return w

def make_data_matrices(data, settings):
    """
    Iterate over each row (load condition) in the data and compute the data matrices (phi, y, w)
    """
    phi = np.vstack([make_load_condition_phi(row, settings) for _, row in data.iterrows()])
    y = np.vstack([make_load_condition_y(row, settings) for _, row in data.iterrows()])
    w = np.vstack([make_load_condition_weights(row, settings) for _, row in data.iterrows()])
    # reshape y and w to be vectors
    y = y.reshape(-1)
    w = w.reshape(-1)
    return phi, y, w

def estimate_parameters(phi, y, w, settings):
    """
    Find parameters s and theta through solving a convex optimization problem
    """
    # set up cvx problem
    # gather both s and theta in a vector beta (to be split up later)
    n_var = phi.shape[1]
    n_data = phi.shape[0]
    beta = cp.Variable(n_var, nonneg = True)

    # optional flow rate weights
    W = w**settings["flow rate weights"]

    # set up the objective function
    objective = cp.Minimize(
        1 / n_data * cp.norm( np.diag(W) @ (phi @ beta - y), settings["cost norm"])
        + settings["valve regularization gain"] * cp.norm(beta[:-7], settings["regularization norm"])
        + settings["pipe regularization gain"] * cp.norm(beta[-7:], settings["regularization norm"])
        )
    prob = cp.Problem(objective, [])

    if settings["cost norm"] == 2: # choose solver based on the problem
        prob.solve(verbose=True, solver=cp.SCS)
    elif settings["cost norm"] == 1:
        prob.solve(verbose=True, solver=cp.SCIPY)
    else:
        # use default solver
        prob.solve(verbose=True)

    # split beta into valve and pipe resistances
    n_v = int((n_var - 7) / 4) # number of valve features
    theta = [beta.value[i*n_v:(i+1)*n_v] for i in range(4)]
    s = beta.value[-7:]

    # set very small values to 0
    theta = [[0 if abs(x) < settings["zero threshold"] else x for x in theta[i]] for i in range(4)]
    s = [0 if abs(x) < settings["zero threshold"] else x for x in s]

    return theta, s

def train_model(settings, data_set):
    
    data = load_data(data_set)
    # append hysteresis-compensated columns "vh{i}" to data
    data = append_hysteresis(data, settings["hysteresis percent"])

    # extract the first "training data percent"% of the data for training
    training_data = data.iloc[:int(len(data)*settings["training data percent"]/100)]

    # make data matrices
    phi, y, w = make_data_matrices(training_data, settings)

    # train the model
    theta, s = estimate_parameters(phi, y, w, settings)

    # save the data matrices for potential later analysis
    save_data_matrices(settings, data_set, phi, y)

    model = {
        "theta": theta,
        "s": s,
        "settings": settings
    }
    return model


