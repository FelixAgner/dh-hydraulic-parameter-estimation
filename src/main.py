import json
import argparse
import os
import logging 

from utils import prepare_datasets
from training import train_model
from evaluation import evaluate_model
from plotting import plot_models
from utils.utils import load_model, save_model, load_results, save_results, print_model, get_all_models, print_data_stats
from utils.parameterization import print_curves

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def handle_training(args):
    
    if len(args.models) == 0:
        raise ValueError('Please provide a model name to train')
    elif len(args.models) == 1 and args.models[0] == "all":
        models = get_all_models()
    else:
        models = args.models
    
    for model in models:
        # read config file
        with open(os.path.join(MODEL_DIR, f"{model}.json"), 'r') as f:
            config = json.load(f)

        config["name"] = model

        for training_data in ["exciting", "realistic"]:
            model = train_model(config, training_data)
            print_model(model)
            save_model(model, training_data, overwrite=args.overwrite)
            for test_data in ["exciting", "realistic"]:
                results = evaluate_model(model, test_data, training_data == test_data)
                save_results(results, config["name"], training_data, test_data, overwrite=args.overwrite)

def handle_plotting(args):
    if len(args.models) == 0:
        raise ValueError('Please provide a model name to plot')
    elif len(args.models) == 1 and args.models[0] == "all":
        plot_models(get_all_models(), args)
    else:
        plot_models(args.models, args)

def handle_printing(args):
    if len(args.models) == 0 or args.models[0] == "all":
        names = get_all_models()
    else:
        names = args.models

    for name, in sorted(names):
        for training_data in ["exciting", "realistic"]:
            print("-------------------------------------------------------")
            print(f"Model name: {name}, Training data: {training_data}")
            model = load_model(name, training_data)
            print_model(model, print_settings=False, print_theta=False)
            print("Valve parameterizations:")
            print_curves(model)


def handle_stats(args):

    for data_set in ["exciting", "realistic"]:
        print_data_stats(data_set)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Decide if you want to prepare datasets, train a model or plot results.')

    parser.add_argument('mode', type=str, help='Which mode to run the script in - prepare, train, plot, print or statistics', choices=['prepare', 'train', 'plot', 'print', 'statistics'])

    parser.add_argument('-m', '--models',nargs='+', help='Name of the model(s) to train or plot', default='')

    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information')

    parser.add_argument('-o', '--overwrite', action='store_true', help='Automatically overwrite old model parameters and results when training.', default=False)

    args = parser.parse_args()

    # set up logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # check if data set should be prepared from raw data
    if args.mode == 'prepare':
        prepare_datasets.run()

    # check if model should be trained
    if args.mode == 'train':
        handle_training(args)

    # check if results should be plotted
    if args.mode == 'plot':
        handle_plotting(args)

    # print parameters for all models
    if args.mode == 'print':
        handle_printing(args)

    # print data set statistics, such as e.g. mean flow rates
    if args.mode == 'statistics':
        handle_stats(args)
            




