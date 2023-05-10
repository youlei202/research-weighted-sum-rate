from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from src.sc import SyntheticControl, SyntheticControlUnconstrained
from sklearn.linear_model import LinearRegression


def train_lr_models(powers_list, observed_signal_and_interferences_list):
    num_Rx_netA = len(observed_signal_and_interferences_list[0])
    lr_train_X = pd.DataFrame(np.array(powers_list)[:, :num_Rx_netA])
    observed_signal_and_interference_data = pd.DataFrame(
        observed_signal_and_interferences_list
    )

    lr_models = []
    for i in range(num_Rx_netA):
        lr_train_y = observed_signal_and_interference_data[i]
        lr_i = LinearRegression().fit(lr_train_X, lr_train_y)
        lr_models.append(lr_i)

    return lr_models


def train_sc_models(observed_signal_and_interferences_list, constrained=True):
    num_Rx_netA = len(observed_signal_and_interferences_list[0])
    sc_data_ = pd.DataFrame(observed_signal_and_interferences_list)
    scaling_factor = 10 / sc_data_.mean().mean()
    sc_data = sc_data_ * scaling_factor

    sc_models = []
    for i in range(num_Rx_netA):
        sc_train_X = sc_data.drop(i, axis=1)
        sc_train_y = sc_data[i]
        if constrained:
            sc_i = SyntheticControl().fit(X=sc_train_X, y=sc_train_y)
        else:
            sc_i = SyntheticControlUnconstrained().fit(X=sc_train_X, y=sc_train_y)
        sc_models.append(sc_i)

    return sc_models


class Experiment(metaclass=ABCMeta):
    def __init__(self, simulator):
        self.simulator = simulator

    @abstractmethod
    def data_generation(self):
        pass

    def modeling(self):
        pass

    @abstractmethod
    def run(self):
        pass


class ExperimentInterferenceModelComparison(Experiment):
    def __init__(self, simulator):
        super().__init__(simulator)

    def data_generation(self, num_samples, max_power, netB_power_mode="dependent"):
        self.power_corr_mat = np.random.uniform(
            -1, 1, (self.simulator.num_Rx_netB, self.simulator.num_Rx_netA)
        )
        self.power_corr_mat_test = np.random.uniform(
            -1, 1, (self.simulator.num_Rx_netB, self.simulator.num_Rx_netA)
        )
        # self.power_corr_mat_test = np.random.normal(
        #     0, 0.5, (self.simulator.num_Rx_netB, self.simulator.num_Rx_netA)
        # )

        self.observed_signal_and_interferences_list = []
        self.powers_list = []
        self.max_power = max_power

        powers_list_netA = []
        for i in range(num_samples):
            power_netA = np.random.uniform(
                0, self.max_power, self.simulator.num_Rx_netA
            )
            powers_list_netA.append(power_netA)

        for t, p_netA in enumerate(powers_list_netA):
            # self.simulator.update_gain_matrix()
            if netB_power_mode == "dependent":
                p_netB = self.power_corr_mat.dot(p_netA).clip(min=0, max=max_power)
            elif netB_power_mode == "random":
                p_netB = np.random.uniform(0, max_power, self.simulator.num_Rx_netB)
            elif netB_power_mode == "zero":
                p_netB = [0.00001 for i in range(self.simulator.num_Rx_netB)]
            p = np.append(p_netA, p_netB)
            self.powers_list.append(p)
            observed_signal_and_interference_noise = (
                np.array(self.simulator.Rx_signal_and_interference_AB_to_A(p))
                + self.simulator.noise_mW
            ) / np.sqrt(p_netA)
            self.observed_signal_and_interferences_list.append(
                observed_signal_and_interference_noise
            )

    def modeling(self, synthetic_constrained=True):
        self.lr_models = train_lr_models(
            powers_list=self.powers_list,
            observed_signal_and_interferences_list=self.observed_signal_and_interferences_list,
        )
        self.sc_models = train_sc_models(
            observed_signal_and_interferences_list=self.observed_signal_and_interferences_list,
            constrained=synthetic_constrained,
        )

    def run(self, num_rounds, netA_power_mode="wmmse", netB_power_mode="zero"):

        uniform_powers_mW = [50 for i in range(self.simulator.num_Rx)]
        if netA_power_mode == "wmmse":
            if not hasattr(self, "power_wmmse"):
                self.power_wmmse = wmmse(
                    simulator=self.simulator,
                    Rx_powers_mW=uniform_powers_mW,
                    Rx_max_powers_mW=[
                        self.max_power for i in range(self.simulator.num_Rx_netA)
                    ],
                    Rx_weights=np.ones(self.simulator.num_Rx),
                    max_iter=5000,
                )

        real_interf = []
        sc_interf = []
        lr_interf = []
        for i in range(num_rounds):
            if netA_power_mode == "wmmse":
                p_netA = self.power_wmmse[: self.simulator.num_Rx_netA]
            elif netA_power_mode == "random":
                p_netA = np.random.uniform(
                    0, self.max_power, self.simulator.num_Rx_netA
                )
            elif netA_power_mode == "uniform":
                p_netA = uniform_powers_mW[: self.simulator.num_Rx_netA]
            else:
                print("Unknown power pattern for netA")
                return

            if netB_power_mode == "dependent":
                p_netB = self.power_corr_mat.dot(p_netA).clip(min=0, max=self.max_power)
            elif netB_power_mode == "zero":
                p_netB = [0.00001 for i in range(self.simulator.num_Rx_netB)]
            elif netB_power_mode == "random":
                p_netB = np.random.uniform(
                    0, self.max_power, self.simulator.num_Rx_netB
                )
            elif netB_power_mode == "uniform":
                p_netB = uniform_powers_mW[self.simulator.num_Rx_netA :]
            else:
                print("Unknown power pattern for netB")
                return

            p = np.append(p_netA, p_netB)
            signal_and_interferences_A = (
                self.simulator.Rx_signal_and_interference_AB_to_A(p)
            )

            for i in range(self.simulator.num_Rx_netA):
                real_interf.append(signal_and_interferences_A[i])
                sc_interf.append(
                    self.sc_models[i].predict(
                        pd.DataFrame(signal_and_interferences_A).drop(i, axis=0)[0]
                    )
                )
                lr_interf.append(self.lr_models[i].predict([p_netA]))

        return pd.DataFrame(
            {
                "interference": [np.mean(real_interf)],
                "sc_bias": [
                    (np.mean(sc_interf) - np.mean(real_interf)) / np.mean(real_interf)
                ],
                "lr_bias": [
                    (np.mean(lr_interf) - np.mean(real_interf)) / np.mean(real_interf)
                ],
            }
        )

    @classmethod
    def _closest_power(cls, p, p_list):
        p_list = np.asarray(p_list)
        dist = np.sum((p_list - p) ** 2, axis=1)
        return np.argmin(dist)
