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
        num_Tx_netA,
        num_Rx_per_Tx_netA,
        num_Tx_netB,
        num_Rx_per_Tx_netB,
        frequency_GHz=60,
        bandwidth_MHz=80,
        RU_bandwidth_KHz=180,
    ):
        self.Tx_radius = Tx_radius
        self.Rx_radius = Rx_radius
        self.num_Tx_netA = num_Tx_netA
        self.num_Rx_per_Tx_netA = num_Rx_per_Tx_netA
        self.num_Tx_netB = num_Tx_netB
        self.num_Rx_per_Tx_netB = num_Rx_per_Tx_netB

        self.num_Tx = num_Tx_netA + num_Tx_netB
        self.num_Rx_netA = num_Tx_netA * num_Rx_per_Tx_netA
        self.num_Rx_netB = num_Tx_netB * num_Rx_per_Tx_netB
        self.num_Rx = self.num_Rx_netA + self.num_Rx_netB

        self.frequency_Hz = frequency_GHz * 1e9
        self.bandwidth_Hz = bandwidth_MHz * 1e6
        self.RU_bandwidth_Hz = RU_bandwidth_KHz * 1e3

        self.noise_mW = (10 ** (0 - 174 / 10)) * self.RU_bandwidth_Hz

        self.path_loss_model = PathLossInHShoppingMalls()
        self.gain_mat_dBm = None
        self.gain_mat_mW = None

        # Generate x,y positions of known Tx and Rx
        (
            self.x_Tx_netA,
            self.y_Tx_netA,
        ) = NetworkSimulator._uniform_distribution_in_circle(
            0, 0, radius=Tx_radius, n=num_Tx_netA
        )
        self.x_Rx_netA_list = []
        self.y_Rx_netA_list = []
        for x, y in zip(self.x_Tx_netA, self.y_Tx_netA):
            xx, yy = NetworkSimulator._uniform_distribution_in_circle(
                x, y, radius=self.Rx_radius, n=self.num_Rx_per_Tx_netA
            )
            self.x_Rx_netA_list.append(xx)
            self.y_Rx_netA_list.append(yy)
        self.x_Rx_netA = np.concatenate(self.x_Rx_netA_list)
        self.y_Rx_netA = np.concatenate(self.y_Rx_netA_list)

        # Generate x,y positions of unknown Tx and Rx
        (
            self.x_Tx_netB,
            self.y_Tx_netB,
        ) = NetworkSimulator._uniform_distribution_in_circle(
            0, 0, radius=Tx_radius, n=num_Tx_netB
        )
        self.x_Rx_netB_list = []
        self.y_Rx_netB_list = []
        for x, y in zip(self.x_Tx_netB, self.y_Tx_netB):
            xx, yy = NetworkSimulator._uniform_distribution_in_circle(
                x, y, radius=self.Rx_radius, n=self.num_Rx_per_Tx_netB
            )
            self.x_Rx_netB_list.append(xx)
            self.y_Rx_netB_list.append(yy)
        self.x_Rx_netB = np.concatenate(self.x_Rx_netB_list)
        self.y_Rx_netB = np.concatenate(self.y_Rx_netB_list)

        self.x_Tx = np.append(self.x_Tx_netA, self.x_Tx_netB)
        self.y_Tx = np.append(self.y_Tx_netA, self.y_Tx_netB)
        self.x_Rx = np.append(self.x_Rx_netA, self.x_Rx_netB)
        self.y_Rx = np.append(self.y_Rx_netA, self.y_Rx_netB)

        self.Tx_of = self._generate_Rx_Tx_mapping()
        self.Rx_of = self._generate_Tx_Rx_mapping()
        self.update_gain_matrix()

    def plot_network(self, figsize=(7, 7)):
        _, ax = plt.subplots(1, 1, figsize=figsize)
        sns.scatterplot(
            self.x_Tx_netA, self.y_Tx_netA, color="b", marker="s", s=100, ax=ax
        )
        sns.scatterplot(
            self.x_Tx_netB,
            self.y_Tx_netB,
            color="grey",
            marker="s",
            s=100,
            alpha=0.3,
            ax=ax,
        )

        sns.scatterplot(
            self.x_Rx_netA, self.y_Rx_netA, color="g", marker="o", s=50, ax=ax
        )
        sns.scatterplot(
            self.x_Rx_netB,
            self.y_Rx_netB,
            color="grey",
            marker="o",
            s=50,
            alpha=0.3,
            ax=ax,
        )
        for k in range(self.num_Tx_netA):
            for j in range(self.num_Rx_per_Tx_netA):
                x = self.x_Tx_netA[k]
                y = self.y_Tx_netA[k]
                u = self.x_Rx_netA_list[k][j]
                v = self.y_Rx_netA_list[k][j]
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

        for k in range(self.num_Tx_netB):
            for j in range(self.num_Rx_per_Tx_netB):
                x = self.x_Tx_netB[k]
                y = self.y_Tx_netB[k]
                u = self.x_Rx_netB_list[k][j]
                v = self.y_Rx_netB_list[k][j]
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
        plt.matshow(self.gain_mat_dBm, cmap=plt.cm.Blues)
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
        while k < self.num_Tx_netA:
            j = 0
            while j < self.num_Rx_per_Tx_netA:
                Tx_of.append(k)
                j += 1
            k += 1

        while k < self.num_Tx_netA + self.num_Tx_netB:
            j = 0
            while j < self.num_Rx_per_Tx_netB:
                Tx_of.append(k)
                j += 1
            k += 1

        return Tx_of

    def _generate_Tx_Rx_mapping(self):
        Rx_of = []
        for k in range(len(self.x_Tx)):
            Rx_of.append(np.where(np.array(self.Tx_of) == k)[0])
        return Rx_of

    @classmethod
    def _uniform_distribution_in_circle(cls, center_x, center_y, radius, n):
        dist = np.sqrt(np.random.uniform(0, 1, n)) * radius
        angle = np.pi * np.random.uniform(0, 2, n)

        x = dist * np.cos(angle) + center_x
        y = dist * np.sin(angle) + center_y
        return x, y

    def get_Tx_uni_index(self, Rx_uni_index):
        return Rx_uni_index // len(self.x_Tx)

    def weighted_sum_rate_Gnats(self, Rx_powers_mW, Rx_weights, part="full"):
        sinr_list = []
        rate_list = []
        for i in range(len(self.x_Rx)):
            k = self.Tx_of[i]
            signal = (
                np.sum(np.take(Rx_powers_mW, self.Rx_of[k])) * self.gain_mat_mW[k][i]
            )
            interference = (
                np.array(
                    [
                        np.sum(np.take(Rx_powers_mW, self.Rx_of[j]))
                        for j in range(len(self.x_Tx))
                    ]
                ).dot(self.gain_mat_mW[:, i])
                - signal
            )
            sinr_j = signal / (interference + self.noise_mW)
            sinr_list.append(sinr_j)

        for i in range(len(self.x_Rx)):
            rate_list.append(math.log(1 + sinr_list[i]))

        spectral_efficiency = 1.4426950408889 * 20 / 1e9  # Gnats / Hz
        scaler = self.bandwidth_Hz * spectral_efficiency

        result = np.array(Rx_weights).dot(np.array(rate_list)) * scaler
        if part == "A":
            return result[: self.num_Rx_netA]
        if part == "B":
            return result[self.num_Rx_netA :]
        return result

    def get_gain_mat(
        self,
        unit="mW",
        part="full",
    ):

        if unit == "dBm":
            gain_mat = self.gain_mat_dBm.copy()
        else:
            gain_mat = self.gain_mat_mW.copy()

        if part not in ["A", "B"]:
            return gain_mat.copy()
        elif part is "A":
            return gain_mat[
                : -self.num_Tx_netB,
                : -self.num_Rx_per_Tx_netB * self.num_Tx_netB,
            ]
        else:
            return gain_mat[
                self.num_Tx_netA :, self.num_Rx_per_Tx_netA * self.num_Tx_netA :
            ]

    def Rx_interference(self, Rx_powers_mW, part="full"):

        interferences = []
        for i in range(len(self.x_Rx)):
            signal_and_interference = np.array(
                [
                    np.sum(np.take(Rx_powers_mW, self.Rx_of[j]))
                    for j in range(len(self.x_Tx))
                ]
            ).dot(self.gain_mat_mW[:, i])
            signal = (
                np.sum(np.take(Rx_powers_mW, self.Rx_of[self.Tx_of[i]]))
                * self.gain_mat_mW[self.Tx_of[i]][i]
            )
            interferences.append(signal_and_interference - signal)

        if part == "A":
            return interferences[: self.num_Rx_netA]
        if part == "B":
            return interferences[self.num_Rx_netA :]
        return interferences

    def abs_Rx_idx_netA(self, rele_Rx_idx):
        return rele_Rx_idx + 0

    def abs_Rx_idx_netB(self, rele_Rx_idx):
        return rele_Rx_idx + self.num_Rx_netA

    def abs_Tx_idx_netA(self, rele_Tx_idx):
        return rele_Tx_idx + 0

    def abs_Tx_idx_netB(self, rele_Tx_idx):
        return rele_Tx_idx + self.num_Tx_netA

    def rele_Rx_idx_netA(self, abs_Rx_idx):
        return abs_Rx_idx

    def rele_Rx_idx_netB(self, abs_Rx_idx):
        return abs_Rx_idx - self.num_Rx_netA

    def rele_Tx_idx_netA(self, abs_Tx_idx):
        return abs_Tx_idx

    def rele_Tx_idx_netB(self, abs_Tx_idx):
        return abs_Tx_idx - self.num_Tx_netA
