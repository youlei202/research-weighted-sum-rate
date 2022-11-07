import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.path_loss import PathLossInHShoppingMalls


class NetworkSimulator(object):
    def __init__(
        self,
        Tx_radius,
        Rx_radius,
        num_Tx_known,
        num_Rx_per_Tx_known,
        num_Tx_unknown,
        num_Rx_per_Tx_unknown,
        frequency_GHz=60,
        bandwidth_MHz=80,
        RU_bandwidth_KHz=180,
    ):
        self.Tx_radius = Tx_radius
        self.Rx_radius = Rx_radius
        self.num_Tx_known = num_Tx_known
        self.num_Rx_per_Tx_known = num_Rx_per_Tx_known
        self.num_Tx_unknown = num_Tx_unknown
        self.num_Rx_per_Tx_unknown = num_Rx_per_Tx_unknown
        self.frequency_Hz = frequency_GHz * 1e9
        self.bandwidth_Hz = bandwidth_MHz * 1e6
        self.RU_bandwidth_Hz = RU_bandwidth_KHz * 1e3

        self.path_loss_model = PathLossInHShoppingMalls()
        self.gain_mat_dBm = None
        self.gain_mat_mW = None

        # Generate x,y positions of known Tx and Rx
        (
            self.x_Tx_known,
            self.y_Tx_known,
        ) = NetworkSimulator._uniform_distribution_in_circle(
            0, 0, radius=Tx_radius, n=num_Tx_known
        )
        self.x_Rx_known_list = []
        self.y_Rx_known_list = []
        for x, y in zip(self.x_Tx_known, self.y_Tx_known):
            xx, yy = NetworkSimulator._uniform_distribution_in_circle(
                x, y, radius=self.Rx_radius, n=self.num_Rx_per_Tx_known
            )
            self.x_Rx_known_list.append(xx)
            self.y_Rx_known_list.append(yy)
        self.x_Rx_known = np.concatenate(self.x_Rx_known_list)
        self.y_Rx_known = np.concatenate(self.y_Rx_known_list)

        # Generate x,y positions of unknown Tx and Rx
        (
            self.x_Tx_unknown,
            self.y_Tx_unknown,
        ) = NetworkSimulator._uniform_distribution_in_circle(
            0, 0, radius=Tx_radius, n=num_Tx_unknown
        )
        self.x_Rx_unknown_list = []
        self.y_Rx_unknown_list = []
        for x, y in zip(self.x_Tx_unknown, self.y_Tx_unknown):
            xx, yy = NetworkSimulator._uniform_distribution_in_circle(
                x, y, radius=self.Rx_radius, n=self.num_Rx_per_Tx_unknown
            )
            self.x_Rx_unknown_list.append(xx)
            self.y_Rx_unknown_list.append(yy)
        self.x_Rx_unknown = np.concatenate(self.x_Rx_unknown_list)
        self.y_Rx_unknown = np.concatenate(self.y_Rx_unknown_list)

        self.x_Tx = np.append(self.x_Tx_known, self.x_Tx_unknown)
        self.y_Tx = np.append(self.y_Tx_known, self.y_Tx_unknown)
        self.x_Rx = np.append(self.x_Rx_known, self.x_Rx_unknown)
        self.y_Rx = np.append(self.y_Rx_known, self.y_Rx_unknown)

        self.Tx_of = self._generate_Rx_Tx_mapping()
        self.update_gain_matrix()

    def plot_network(self, figsize=(7, 7)):
        _, ax = plt.subplots(1, 1, figsize=figsize)
        sns.scatterplot(
            self.x_Tx_known, self.y_Tx_known, color="b", marker="s", s=100, ax=ax
        )
        sns.scatterplot(
            self.x_Tx_unknown,
            self.y_Tx_unknown,
            color="grey",
            marker="s",
            s=100,
            alpha=0.3,
            ax=ax,
        )

        sns.scatterplot(
            self.x_Rx_known, self.y_Rx_known, color="g", marker="o", s=50, ax=ax
        )
        sns.scatterplot(
            self.x_Rx_unknown,
            self.y_Rx_unknown,
            color="grey",
            marker="o",
            s=50,
            alpha=0.3,
            ax=ax,
        )
        for k in range(self.num_Tx_known):
            for j in range(self.num_Rx_per_Tx_known):
                x = self.x_Tx_known[k]
                y = self.y_Tx_known[k]
                u = self.x_Rx_known_list[k][j]
                v = self.y_Rx_known_list[k][j]
                ax.quiver(
                    x,
                    y,
                    u - x,
                    v - y,
                    scale_units="xy",
                    angles="xy",
                    scale=1,
                    width=0.005,
                    headlength=5,
                    headwidth=5,
                    color="black",
                    alpha=0.7,
                )

        for k in range(self.num_Tx_unknown):
            for j in range(self.num_Rx_per_Tx_unknown):
                x = self.x_Tx_unknown[k]
                y = self.y_Tx_unknown[k]
                u = self.x_Rx_unknown_list[k][j]
                v = self.y_Rx_unknown_list[k][j]
                ax.quiver(
                    x,
                    y,
                    u - x,
                    v - y,
                    scale_units="xy",
                    angles="xy",
                    scale=1,
                    width=0.005,
                    headlength=5,
                    headwidth=5,
                    color="grey",
                    alpha=0.7,
                )

        ax.set_xticks([])
        ax.set_yticks([])

    def update_gain_matrix(self):
        self.gain_mat_dBm = np.array(
            [
                [self._generate_gain_in_dBm(k, j) for j in range(len(self.x_Rx))]
                for k in range(len(self.x_Tx))
            ]
        )
        self.gain_mat_mW = np.vectorize(lambda x: 10 ** (x / 10))(self.gain_mat_dBm)

    def plot_gain_mat(self):
        plt.matshow(self.gain_mat_dBm)
        plt.colorbar()
        plt.show()

    def _generate_gain_in_dBm(self, Tx_uni_index, Rx_uni_index):
        x_tx = self.x_Tx[Tx_uni_index]
        y_tx = self.y_Tx[Tx_uni_index]
        x_rx = self.x_Rx[Rx_uni_index]
        y_rx = self.y_Rx[Rx_uni_index]
        distance_m = math.dist((x_tx, y_tx), (x_rx, y_rx))
        return self.path_loss_model.in_dBm(
            frequency_Hz=self.frequency_Hz, distance_m=distance_m
        )

    def _generate_Rx_Tx_mapping(self):
        # Mapping each Rx to its corresponding Tx
        Tx_of = []
        k = 0
        while k < self.num_Tx_known:
            j = 0
            while j < self.num_Rx_per_Tx_known:
                Tx_of.append(k)
                j += 1
            k += 1

        while k < self.num_Tx_known + self.num_Tx_unknown:
            j = 0
            while j < self.num_Rx_per_Tx_unknown:
                Tx_of.append(k)
                j += 1
            k += 1

        return Tx_of

    @classmethod
    def _uniform_distribution_in_circle(cls, center_x, center_y, radius, n):
        dist = np.sqrt(np.random.uniform(0, 1, n)) * radius
        angle = np.pi * np.random.uniform(0, 2, n)

        x = dist * np.cos(angle) + center_x
        y = dist * np.sin(angle) + center_y
        return x, y

    def get_Tx_uni_index(self, Rx_uni_index):
        return Rx_uni_index // len(self.x_Tx)

    def _noise_mW(self, dB):
        return (10 ** ((0 - 174) / 10)) * self.RU_bandwidth_Hz

    def weighted_sum_rate_Gbps(self, Tx_powers, Rx_weights):
        """Sum rate on 1 RU"""

        sinr_list = []
        rate_list = []
        for j in range(len(self.x_Rx)):
            k = self.Tx_of[j]
            signal = Tx_powers[k] * self.gain_mat_mW[k][j]
            interference = np.array(Tx_powers).dot(self.gain_mat_mW[:, j]) - signal
            sinr_j = signal / (interference + self._noise_mW(dB=3))
            sinr_list.append(sinr_j)

        for j in range(len(self.x_Rx)):
            rate_list.append(math.log(1 + sinr_list[j]))

        spectral_efficiency = 1.4426950408889 * 20 / 1e9  # Gnats / Hz
        scaler = self.bandwidth_Hz * spectral_efficiency
        return np.array(Rx_weights).dot(np.array(rate_list)) * scaler

    def get_gain_mat(self, part="full", unit="mW"):

        if unit == "dBm":
            gain_mat = self.gain_mat_dBm.copy()
        else:
            gain_mat = self.gain_mat_mW.copy()

        if part not in ["known", "unknown"]:
            return gain_mat.copy()
        elif part is "known":
            return gain_mat[
                : -self.num_Tx_unknown,
                : -self.num_Rx_per_Tx_unknown * self.num_Tx_unknown,
            ]
        else:
            return gain_mat[
                self.num_Tx_known :, self.num_Rx_per_Tx_known * self.num_Tx_known :
            ]
