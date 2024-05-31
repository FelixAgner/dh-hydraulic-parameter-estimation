import pandas as pd
from utils.utils import *
from utils.parameterization import valve_curve, RAMP_PARAMETERS
import os
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tikz")

def error_data(model_name, training_set, test_set):
    row_lim = np.Inf
    
    data = load_data(test_set)
    results = load_results(model_name, training_set, test_set)
    
    tikz_data = pd.DataFrame()
    for i in range(4):
        tikz_data[f"e{i}"] = results[f"e{i}"]
        tikz_data[f"v{i}"] = data[f"v{i}"]
        tikz_data[f"q{i}"] = data[f"q{i}"]
        qm = tikz_data[f"q{i}"].mean()
        tikz_data[f"em{i}"] = tikz_data[f"e{i}"] /qm 
    # add columns for sign-change in v[i] (default 1)
    for i in range(4):
        tikz_data[f"sign{i}"] = data[f"v{i}"].diff().apply(lambda x : 1 if x > 0 else -1 if x < 0 else 1)

    # split data into training and test data
    if training_set == test_set:
        tikz_train = tikz_data[results["training"] == True]
        tikz_test = tikz_data[results["training"] == False]
    else:
        tikz_train = None
        tikz_test = tikz_data
    
    # crop data to fit in tikz
    if training_set == test_set and tikz_train.shape[0] > row_lim:
        tikz_train = tikz_train.iloc[:row_lim]
    if tikz_test.shape[0] > row_lim:
        tikz_test = tikz_test.iloc[:row_lim]

    if training_set == test_set:
        tikz_train.to_csv(f"data/tikz/error_train_{model_name}_{training_set}_{test_set}.csv", index=False, lineterminator='\n')
    tikz_test.to_csv(f"data/tikz/error_test_{model_name}_{training_set}_{test_set}.csv", index=False, lineterminator='\n')
        

def valve_curve_data(model_name, training_set):
    model = load_model(model_name, training_set)
    kv = valve_curve(model)
    kv.to_csv(f"data/tikz/valve_curve_{model_name}_{training_set}.csv", index=False, lineterminator='\n')
    

def step_data():
    """
    Step response data
    """
    original = pd.read_csv("data/raw_data/exciting/consumer_3.csv")
    tikz_data = original.iloc[501:620][["time", "v", "q_pipe"]]
    tikz_data["time"] -= 501
    tikz_data["v"] *= 0.01
    tikz_data.rename(columns={"q_pipe" : "q"}, inplace=True)
    tikz_data.to_csv("data/tikz/step_response.csv", index=False, lineterminator='\n')

def pipe_parameter_table():
    data_short = {
        "exciting" : "E",
        "realistic" : "R"
    }
    table = """Model & Data & $s_1$ & $s_2$ & $s_3$ & $s_4$ & $s_5$ & $s_6$ & $s_7$ \\\\ \hline \n"""
    for model_name in get_all_models():
        for training_set in ["exciting", "realistic"]:
            model = load_model(model_name, training_set)
            s = model["s"]
            table += f"{model_name} & {data_short[training_set]} & "
            for i in range(7):
                if s[i] == 0:
                    table += r"\zero & "
                else:
                    table += f"{sigdig(s[i], 2)} & "
            table = table[:-2] + " \\\\\n"
    filename = os.path.join(DATA_DIR, "pipe_parameter_table.txt")
    with open(filename, "w") as f:
        f.write(table)

def model_equation():
    def lin_term(theta, i):
        return r"""\Delta p_{1} &= \frac{{ {0} }}{{v_{1}^2}}q_{1}^2 \\ """.format(sigdig(theta, 2), i)

    def ramp_term(theta, i, pars):
        a, b, c = pars
        return r"""\frac{{ {0} }}{{\ramp{{ v_{1} }}{{ {2} }}{{ {3} }}^{{ {4} }} }} +""".format(sigdig(theta, 2), i, a, b, c)
        
    for model_name in get_all_models():
        for training_data in ["exciting", "realistic"]:
            model = load_model(model_name, training_data)
            theta = model["theta"]
            s = r""
            if model["settings"]["parameterization"] == "linear":
                for i in range(4):
                    s += lin_term(theta[i][0], i+1)
            elif model["settings"]["parameterization"] == "ramps":
                pars = [(a,b,c) for a in RAMP_PARAMETERS["a"] for b in RAMP_PARAMETERS["b"] for c in RAMP_PARAMETERS["c"]]
                for i in range(4):
                    s += r"""\Delta p_{0} &= \left(""".format(i+1)
                    for k, t in enumerate(theta[i]):
                        if t > 0.0001:
                            s += ramp_term(t, i+1, pars[k])
                    s = s[:-2]
                    s += r"""\right)q_{0}^2 \\""".format(i+1)  + " \n"
                s = s[:-4]
            filename = os.path.join(DATA_DIR, "equations", f"model_equation_{model_name}_{training_data}.txt")
            with open(filename, "w") as f:
                f.write(s)


def sigdig(x, n):
    # round to n significant digits
    if abs(x) <= 1e-8:
        return 0

    return round(x, n - int(np.floor(np.log10(abs(x)))) - 1)

def q3_sorted():
    """
    Sort the data for consumer 3 in the exciting data set
    """
    data = load_data("exciting")
    # make a new dataframe with q3 sorted
    sorted_data = data.sort_values("q3")
    sorted_data["i"] = np.arange(sorted_data.shape[0])

    sorted_data.to_csv(
        os.path.join(DATA_DIR, "q3_sorted.csv"),
    )

def valve_overlap_example(plot=False):
    v_values = np.linspace(0, 1, 300)
    k1 = lambda v: 0.63 * ramp(v, 0.22, 0.9) ** 0.75
    k2 = lambda v: ramp(v, 0.1, 0.9) ** 1.5
    s = 1.5
    f1 = lambda v: k1(v)
    f2 = lambda v: k2(v) / np.sqrt(s * k2(v) ** 2 + 1)
    data = {
        "v": v_values,
        "f1": [f1(v) for v in v_values],
        "f2": [f2(v) for v in v_values],
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATA_DIR, "valve_overlap_example.csv"), index=False)
    if plot:
        plt.plot(v_values, [f1(v) for v in v_values], label="f1")
        plt.plot(v_values, [f2(v) for v in v_values], label="f2")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    pipe_parameter_table()
    model_equation()
    step_data()
    q3_sorted()
    valve_overlap_example(plot=True)
    for model_name in get_all_models():
        for training_set in ["exciting", "realistic"]:
            for test_set in ["exciting", "realistic"]:
                error_data(model_name, training_set, test_set)
            valve_curve_data(model_name, training_set)