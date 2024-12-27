import datetime
import errno
import numpy as np
import pandas as pd
import gc
import os
import json


class BetterLogger:

    def __init__(self, curdir: str, params: dict, l_params: dict, log_params: dict, train: bool, deep_algo: bool, buffer_size: int, weights_file=None):
        OUTPUT_FILE_EXTENSION = ".csv"
        WEIGHTS_FILE_EXTENSION = ".npy"
        PARAMS_FILE_EXTENSION = ".txt"
        mode = "train" if train else "eval"
        time_now = datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S")

        base_dir = os.path.join(curdir, "runs/" + mode)
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir)
        output_dir = os.path.join(base_dir, mode + "_" + time_now)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        filename = log_params[mode + "_output_file"].replace("-", "_") + '_' + time_now + OUTPUT_FILE_EXTENSION
        self.output_file = os.path.join(output_dir, filename)
        params_filename = log_params[mode + "_params_file"] + '_' + time_now + PARAMS_FILE_EXTENSION
        self.params_file = os.path.join(output_dir, params_filename)

        weights_dir = os.path.join(curdir, "runs/weights")
        if not os.path.isdir(weights_dir):
            os.makedirs(weights_dir)
        if train:
            weights_filename = log_params[mode + "_weights_file"] + '_' + params["reward_type"] + '_' + time_now + WEIGHTS_FILE_EXTENSION
            self.weights_file = os.path.join(weights_dir, weights_filename)
        else:
            if weights_file is None:
                if not os.path.isdir(weights_dir):
                    raise FileNotFoundError(errno.ENOENT, "No such directory", weights_dir)
                self.weights_file = self._get_weight_path(weights_dir, WEIGHTS_FILE_EXTENSION)
            else:
                self.weights_file = weights_file

        self._write_params(params, l_params, log_params, train)
        # we need a list of logging functions provided by the user
        # these functions will accept an object that will collect all
        # the relevant structures and variables. Each function returns
        # a pandas.DataFrame.
        self.loggers = []

        # store all the resulting dataframes for each logging run
        self.results = []

    def collect(self, data):
        """
        Collects loggging data from the registered functions
        param: data Structure containing all data relevant for logging.
        """
        run = {}
        for logger in self.loggers:
            # call logger function
            logged = logger(data)
            run.update(logged)
        self.results.append(pd.DataFrame(data=run, index=[0]))

    def log(self):
        """
        Writes logs
        """
        df = pd.concat(self.results)
        self._write_to_csv(df)
        self.results = []
        gc.collect()

    def _write_to_csv(self, dataframe):
        if os.path.isfile(self.output_file): # check se il file esiste
            with open(self.output_file, 'a') as f:
                dataframe.to_csv(f, header=False, sep=',', index=False)
        else: # check se il file non esiste
            dataframe.to_csv(self.output_file, sep=',', index=False)

    def _write_params(self, params, l_params, log_params, train):
        # Q-Learning
        alpha = l_params["alpha"]  # DOC learning rate (0 learn nothing 1 learn suddenly)
        gamma = l_params["gamma"]  # DOC discount factor (0 care only bout immediate rewards, 1 care only about future ones)
        epsilon = l_params["epsilon"]  # DOC chance of random action
        epsilon_min = l_params["epsilon_min"]  # DOC chance of random action
        decay_type = l_params["decay_type"]  # DOC di quanto diminuisce epsilon ogni episode (e.g. 1500 episodes => decay = 0.9995)
        decay = l_params["decay"]  # DOC di quanto diminuisce epsilon ogni episode (e.g. 1500 episodes => decay = 0.9995)
        if train:
            train_episodes = l_params["train_episodes"]
            train_log_every = log_params["train_log_every"]
        else:
            test_episodes = l_params["test_episodes"]
            test_log_every = log_params["test_log_every"]

        with open(self.params_file, 'w') as f:
            f.write(f"{json.dumps(params, indent=2)}\n")
            f.write("----------\n")
            if train:
                f.write(f"TRAIN_EPISODES = {train_episodes}\n")
                f.write(f"TRAIN_LOG_EVERY = {train_log_every}\n")
            else:
                f.write(f"TEST_EPISODES = {test_episodes}\n")
                f.write(f"TEST_LOG_EVERY = {test_log_every}\n")
                f.write(f"weights_file = {self.weights_file}\n")
            f.write("----------\n")
            f.write(f"alpha = {alpha}\n")
            f.write(f"gamma = {gamma}\n")
            f.write(f"epsilon = {epsilon}\n")
            f.write(f"epsilon_min = {epsilon_min}\n")
            f.write(f"decay_type = {decay_type}\n")
            f.write(f"decay = {decay}\n")
            f.write("----------\n")

    def _get_weight_path(self, weights_dir, ext):
        if not os.path.isdir(weights_dir):
            raise FileNotFoundError(errno.ENOENT, "No such directory found", weights_dir)
        weights_filename = [f for f in os.listdir(weights_dir) if os.path.isfile(os.path.join(weights_dir, f)) and f.endswith(ext)][-1]
        if len(weights_filename) == 0:
            raise FileNotFoundError(errno.ENOENT, "No such weights file (.npy) found in", weights_dir)
        weights_file = os.path.join(weights_dir, weights_filename)
        return weights_file

    def save_model(self, weights):
        np.save(self.weights_file, weights)

    def load_model(self):
        return np.load(self.weights_file)

    def save_computation_time(self, computation_time, train=True):
        with open(self.params_file, 'a') as f:
            if train:
                f.write(f"Training time: {computation_time}\n")
            else:
                f.write(f"Testing time: {computation_time}\n")
