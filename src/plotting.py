import pandas as pd
import matplotlib.pyplot as plt
from utils.parameterization import valve_curve
from utils.utils import load_model, load_data, load_results

def plot_hysteresis(results, data):
    # add a column "sign" which is 1 if v[i] is increasing and -1 if v[i] is decreasing
    results["sign0"] = results["v0"].diff().apply(lambda x : 1 if x > 0 else -1 if x < 0 else 0)
    results["sign1"] = results["v1"].diff().apply(lambda x : 1 if x > 0 else -1 if x < 0 else 0)
    results["sign2"] = results["v2"].diff().apply(lambda x : 1 if x > 0 else -1 if x < 0 else 0)
    results["sign3"] = results["v3"].diff().apply(lambda x : 1 if x > 0 else -1 if x < 0 else 0)

    # make 4 subplots 2x2
    fig, axs = plt.subplots(2, 2)
    cmap = {
        1 : "red",
        -1 : "blue",
        0 : "black"
    }
    for i, ax in enumerate(axs.flat):
        # plot the error over valve settings
        # color with sign
        ax.scatter(data[f"vh{i}"], results[f"e{i}"], c=results[f"sign{i}"].apply(lambda x : cmap[x]))
        ax.set_xlabel(f"v{i} (h compensated)")
        ax.set_ylabel(f"e{i}")
        ax.grid()
    #plt.show()

def plot_valve_curve(ax, valve, model_name):
    for training_set in ["exciting", "realistic"]:
        model = load_model(model_name, training_set)
        kv = valve_curve(model)
        ax.plot(kv["v"], kv[f"kv{valve}"], label=training_set)
        ax.set_xlabel(f"v{valve+1}")
        ax.set_ylabel("kv")
        ax.legend()
        ax.grid()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 4.5)

def scatter_errors(ax, valve, model_name, training_set, test_set):
    data = load_data(test_set)
    results = load_results(model_name, training_set, test_set)
    
    if training_set == test_set:
        train_percent = load_model(model_name, training_set)["settings"]["training data percent"]
        train_samples = int(train_percent/100 * len(data))
        # plot first #train percent of training data
        ax.scatter(data[f"v{valve}"][:train_samples], results[f"e{valve}"][:train_samples], label="train", alpha=0.1)
        # plot the rest of the data
        ax.scatter(data[f"v{valve}"][train_samples:], results[f"e{valve}"][train_samples:], label="test", alpha=0.1)
    else:
        ax.scatter(data[f"v{valve}"], results[f"e{valve}"], label="test", alpha=0.5)

    ax.set_xlabel(f"v{valve}")
    ax.set_ylabel(f"e{valve}")
    ax.legend()
    ax.grid()
    # limits
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.5)


def plot_all(model_name):
    # make 5 x 4 plots
    fig, axs = plt.subplots(5, 4)
    # set the title
    fig.suptitle(f"Results for model {model_name}")

    # Define your row and column labels
    row_labels = [
        'Exc. - Exc.',
        'Exc. - Real.',
        'Real. - Exc.',
        'Real. - Real.', 
        'Valve curves'
        ]
    col_labels = [f"Valve {i}" for i in range(1,5)]

    # Labeling the rows
    for ax, row in zip(axs[:,0], row_labels):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)

    # Labeling the columns
    for ax, col in zip(axs[0], col_labels):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, ax.xaxis.labelpad + 5),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    # plot scatter plots of errors over valve settings for all training and test data sets
    for valve in range(4):
        for i, (training_set, test_set) in enumerate([("exciting", "exciting"), ("exciting", "realistic"), ("realistic", "exciting"), ("realistic", "realistic")]):
            ax = axs[i, valve]
            scatter_errors(ax, valve, model_name, training_set, test_set)

        # plot valve curves
        plot_valve_curve(axs[4, valve], valve, model_name)

    return fig, axs

def plot_models(models, args):
    for model in models:
        plot_all(model)
    plt.show()
