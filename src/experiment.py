from abc import ABCMeta, abstractmethod
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from src.sc import SyntheticControl
from src.algorithm import wmmse


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
        self.gain_netA = simulator.get_gain_mat(part="A", unit="mW")

    def data_generation(self, num_samples, max_power, netB_power_pattern="dependent"):
        self.power_corr_mat = np.random.uniform(
            -0.5, 0.5, (self.simulator.num_Rx_netB, self.simulator.num_Rx_netA)
        )

        self.observed_interferences_list = []
        self.powers_list = []
        self.max_power = max_power

        powers_netA = [
            np.random.uniform(0, self.max_power, self.simulator.num_Rx_netA)
            for i in range(num_samples)
        ]

        for t, p_netA in enumerate(powers_netA):
            p_netA = p_netA.clip(min=0, max=max_power)
            if netB_power_pattern == "dependent":
                p_netB = self.power_corr_mat.dot(p_netA).clip(min=0, max=max_power)
            elif netB_power_pattern == "random":
                p_netB = np.random.uniform(0, max_power, self.simulator.num_Rx_netB)
            p = np.append(p_netA, p_netB)
            self.powers_list.append(p)
            observed_interference = self.simulator.Rx_interference(p, part="A")
            self.observed_interferences_list.append(observed_interference)

    def modeling(self):
        self.lr_models = self._train_lr_models()
        self.sc_models = self._train_sc_models()

    def run(self, num_rounds, power_pattern_netA="wmmse", power_pattern_netB="zero"):

        uniform_powers_mW = [50 for i in range(self.simulator.num_Rx)]
        if power_pattern_netA == "wmmse":
            if not hasattr(self, "power_wmmse"):
                self.power_wmmse = wmmse(
                    simulator=self.simulator,
                    gain_mat_mW=self.gain_netA,
                    Rx_powers_mW=uniform_powers_mW,
                    Rx_max_powers_mW=[
                        self.max_power for i in range(self.simulator.num_Rx_netA)
                    ],
                    Rx_weights=np.ones(self.simulator.num_Rx),
                    Tx_idx_shift=0,
                    Rx_idx_shift=0,
                    max_iter=5000,
                )

        power_corr_mat = np.random.uniform(
            -0.5,
            0.5,
            (
                self.simulator.num_Tx_netB * self.simulator.num_Rx_per_Tx_netB,
                self.simulator.num_Tx_netA * self.simulator.num_Rx_per_Tx_netA,
            ),
        )

        real_interf = []
        sc_interf = []
        lr_interf = []
        for i in range(num_rounds):
            if power_pattern_netA == "wmmse":
                p_netA = self.power_wmmse[: self.simulator.num_Rx_netA]
            elif power_pattern_netA == "random":
                p_netA = np.random.uniform(
                    0, self.max_power, self.simulator.num_Rx_netA
                )
            elif power_pattern_netA == "uniform":
                p_netA = uniform_powers_mW[: self.simulator.num_Rx_netA]
            else:
                print("Unknown power pattern for netA")
                return

            if power_pattern_netB == "dependent":
                p_netB = power_corr_mat.dot(p_netA).clip(min=0, max=self.max_power)
            elif power_pattern_netB == "zero":
                p_netB = [0 for i in range(self.simulator.num_Rx_netB)]
            elif power_pattern_netB == "random":
                p_netB = np.random.uniform(
                    0, self.max_power, self.simulator.num_Rx_netB
                )
            elif power_pattern_netB == "uniform":
                p_netB = uniform_powers_mW[self.simulator.num_Rx_netA :]
            else:
                print("Unknown power pattern for netB")
                return

            p = np.append(p_netA, p_netB)
            interferences_netA = self.simulator.Rx_interference(p, part="A")

            for i in range(self.simulator.num_Rx_netA):
                real_interf.append(interferences_netA[i])
                sc_interf.append(
                    self.sc_models[i].predict(
                        pd.DataFrame(interferences_netA).drop(i, axis=0)[0]
                    )
                )
                lr_interf.append(self.lr_models[i].predict([p_netA]))

        return pd.DataFrame(
            {
                "real_interference": [np.mean(real_interf)],
                "sc_bias": [
                    (np.mean(sc_interf) - np.mean(real_interf)) / np.mean(real_interf)
                ],
                "lr_bias": [
                    (np.mean(lr_interf) - np.mean(real_interf)) / np.mean(real_interf)
                ],
            }
        )

    def _train_lr_models(self):
        lr_train_X = pd.DataFrame(
            np.array(self.powers_list)[:, : self.simulator.num_Rx_netA]
        )
        observed_interference_data = pd.DataFrame(self.observed_interferences_list)

        lr_models = []
        for i in range(self.simulator.num_Rx_netA):
            lr_train_y = observed_interference_data[i]
            lr_i = LinearRegression().fit(lr_train_X, lr_train_y)
            lr_models.append(lr_i)

        return lr_models

    def _train_sc_models(self):
        sc_models = []
        sc_data_ = pd.DataFrame(self.observed_interferences_list)
        scaling_factor = 100 / sc_data_.mean().mean()
        sc_data = sc_data_ * scaling_factor
        for i in range(self.simulator.num_Rx_netA):
            sc_train_X = sc_data.drop(i, axis=1)
            sc_train_y = sc_data[i]
            sc_i = SyntheticControl().fit(X=sc_train_X, y=sc_train_y)
            sc_models.append(sc_i)

        return sc_models

    @classmethod
    def _closest_power(cls, p, p_list):
        p_list = np.asarray(p_list)
        dist = np.sum((p_list - p) ** 2, axis=1)
        return np.argmin(dist)
