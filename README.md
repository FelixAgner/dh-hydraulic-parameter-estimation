[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11946410.svg)](https://doi.org/10.5281/zenodo.11946410)

# dh-hydraulic-parameter-estimation
This repository contains experimental data and code associated with the paper 

```
Hydraulic parameter estimation for district heating based on laboratory experiments,
Felix Agner, Christian Møller Jensen, Anders Rantzer, Carsten Skovmose Kallesøe, Rafal Wisniewski
Energy,
Volume 312,
2024,
133462,
ISSN 0360-5442,
```
URL: [https://doi.org/10.1016/j.energy.2024.133462](https://doi.org/10.1016/j.energy.2024.133462)

DOI: [https://www.sciencedirect.com/science/article/pii/S0360544224032389](https://www.sciencedirect.com/science/article/pii/S0360544224032389)

The experimental data comes from the Smart Water Infrastructure laboratory in Aalborg, Denmark.

# Usage
To use this code and data, simply clone this repo and install the required python packages found in the `requirements.txt` file.

Most of the code is found in the `src` directory. The main script is `src/main.py`, which can be run with the following command:
```bash
python src/main.py
```
The main script can be run in several modes, which is the first mandatory argument. The modes are:
- `prepare`: Prepare the data from raw data files. 
- `train`: Train a model using the data and configuration files.
- `plot`: Generate plots of the model.
- `print`: Print the model parameters.
- `statistics`: Print some statistics from the data sets.

Optional arguments can be passed to some of the scripts:
- `-m` or `--models`: Name of the model(s) to use in training, printing or plotting.
- `-o` or `--overwrite`: Automatically overwrite existing files when training.
- `-d` or `--debug`: Print debug information.

## Preparing data sets
To access the data, you must first extract the raw experimental data in the `data/raw_data.zip` file. Then you can prepare the data by running the main script with the `prepare` mode:
```bash
python src/main.py prepare
```

## Model training
To train a model, you first need to generate a config file for the model. To configure a file for training the model, create a JSON file in the `src/models` directory named e.g. "`model_name.json`" with the following fields.

Template training configuration file:
```json
{
    "parameterization": "template",
    "flow rate weights": 0,
    "hysteresis percent": 1.5,
    "cost norm": 1,
    "regularization norm": 1,
    "valve regularization gain": 0.1,
    "pipe regularization gain": 0.0,
    "flow rate exponent": 2.0,
    "flow rate threshold": 0.0,
    "training data percent": 70,
    "zero threshold": 1e-4,
    "description": ""
}
```
Description of each field:
- `parameterization`: Name of the parameterization to use. See "parameterizations" section below.
- `flow rate weights`: Exponent for use in the trarining to weight training towards higher flow rates. Not considered in published results. Set to 0 for no effect.
- `hysteresis percent`: delta value for hysteresis filtering.
- `cost norm`: Exponent for use in the training to weight the cost function.
- `regularization norm`: Exponent for use in the training to weight the regularization term.
- `valve regularization gain`: Gain lambda for the valve parameter regularization term.
- `pipe regularization gain`: Gain for pipe parameter regularization term.
- `flow rate exponent`: Exponent in relation between pressures and flow rates, applied to both valves and pipes. Corresponds to 1 + gamma.
- `flow rate threshold`: Threshold for flow rates. If any flow rate falls below this threshold, the sample will not be used in training. Set to 0 for no effect.
- `training data percent`: Percentage of the data to be used for training.
- `zero threshold`: Threshold for parameter values to be pruned to zero.
- `description`: Description of the training configuration.

Then train your model by running the main script with the `train` mode and the name of the model configuration file:
```bash
python src/main.py train -m model_name
```
where `model_name` is the name of the configuration file without the `.json` extension.

## Valve curve parameterization
To introduce a new valve curve parameterization which you can use for your models, manually edit the `src/utils/parameterizations.py` file. 

Create a new function with the following signature:
```python
def f_your_parameterization(v):
    return np.array( [
        1 / (your_k(vi)**2) for vi in v
    ])
```
Where the input `v` is a vector of valve positions and `your_k` is the function that maps valve positions to the valve curve parameterization you want to use. 

Then, add the function to the `model_map` dictionary in the `load_parameterization` function in the same file:
```python
    model_map = {
            "linear": f_linear,
            "ramps": f_ramps
            "your parameterization": f_your_parameterization
        }
```

To be able to print your model, you can also include a new option for it under the `print_curves` function in the same file.

# Data
The following sections describe the data sets used in the paper associated with this repository. Pressures are measured in meters of water column (mWC) and flow rates in liters per minute (l/min).

## Raw data files
In the `raw_data.zip` file, there are two folders corresponding to the data sets used in the paper. Each such folder contains the following files:
- `consumer_x.csv`: Data related to consumer x where x = 1,...,4.
- `pump_41.csv`: Data related to the pumping station.
- `pipe_20.csv`: Data from the pipe unit acting as the supply side.
- `pipe_24.csv`: Data from the pipe unit acting as the return side.

The columns of the data files are as follows:
### `consumer_x.csv`
- `time`: Time in seconds.
- `q_sensor`: Flow rate in the sensor equipped on the consumer module (not used due to performance issues).
- `q_pipe`: Flow rate measured at the connected pipe unit, which is of higher quality than `q_sensor` and hence used in later analysis.
- `p_in`: Pressure at the inlet of the consumer module.
- `p_out`: Pressure at the outlet of the consumer module.
- `v`: Valve position of the consumer module (0-100).
- `qr`: Set-point flow rate of the consumer module (only valid for the realistic data set).

### `pump_41.csv`
- `time`: Time in seconds.
- `p3_1`: Pressure at the inlet of the pump (return side).
- `p3_2`: Pressure at the outlet of the pump (supply side).
- `q2_2`: Flow rate through the pump.

### `pipe_x.csv`
- `time`: Time in seconds.
- `px_1`: Pressure at the inlet of first section of the unit.
- `px_2`: Pressure at the outlet of first section of the unit (connected to consumer 1).
- `px_3`: Pressure at the outlet of second section of the unit (connected to next section of pipe).
- `q3_x`: Flow rate at inlet of section number x.

## Prepared data files
Running the `main.py` script with the `prepare` keyword will generate the two filtered data sets used in the paper. These are stored in the `data` directory as `realistic.csv` and `exciting.csv`. Their columns are given by the following data.

Note that we here use pythonic indexing, i.e., the first consumer is indexed as 0.

- `sample`: Sample number.
- `lab_time_minutes`: Time in minutes since the start of the experiment (only realistic data).
- `lab_time_seconds`: Time in seconds since the start of the experiment (only realistic data).
- `time`: Real world equivalent time in hours (only realistic data).
- `q0`: Flow rate for consumer 1 and pipe 1.
- `q1`: Flow rate for consumer 2 and pipe 2.
- `q2`: Flow rate for consumer 3 and pipe 3.
- `q3`: Flow rate for consumer 4 and pipe 4.
- `q4`: Flow rate through pipe 5.
- `q5`: Flow rate through pipe 6.
- `q6`: Flow rate through pipe 7.
- `qr0`: Set-point flow rate for consumer 1 (only for realistic data).
- `qr1`: Set-point flow rate for consumer 2 (only for realistic data).
- `qr2`: Set-point flow rate for consumer 3 (only for realistic data).
- `qr3`: Set-point flow rate for consumer 4 (only for realistic data).
- `v0`: Valve position for consumer 1 (normalized to range 0-1).
- `v1`: Valve position for consumer 2 (normalized to range 0-1).
- `v2`: Valve position for consumer 3 (normalized to range 0-1).
- `v3`: Valve position for consumer 4 (normalized to range 0-1).
- `ps_pump`: Pressure at the supply side of the pump.
- `pr_pump`: Pressure at the return side of the pump.
- `dp_pump`: Pressure difference over the pump.
- `dp0`: Pressure difference over consumer 1, measured at the connection to pipe unit.
- `dp1`: Pressure difference over consumer 2, measured at the connection to pipe unit.
- `dp2`: Pressure difference over consumer 3, measured at the connection to pipe unit.
- `dp3`: Pressure difference over consumer 4, measured at the connection to pipe unit.

# Contact and questions
For questions or comments, please contact the corresponding author of the paper associated with this repository.
```
Felix Agner: felix.agner@control.lth.se
```