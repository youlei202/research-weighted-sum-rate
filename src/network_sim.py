import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class NetworkSimulator(object):
    def __init__(
        self,
        Tx_radius,
        Rx_radius,
        num_Tx_known,
        num_Rx_per_Tx_known,
        num_Tx_unknown,
        num_Rx_per_Tx_unknown,
    ):
        self.Tx_radius = Tx_radius
        self.Rx_radius = Rx_radius
        self.num_Tx_known = num_Tx_known
        self.num_Rx_per_Tx_known = num_Rx_per_Tx_known
        self.num_Tx_unknown = num_Tx_unknown
        self.num_Rx_per_Tx_unknown = num_Rx_per_Tx_unknown

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

    @classmethod
    def _uniform_distribution_in_circle(cls, center_x, center_y, radius, n):
        dist = np.sqrt(np.random.uniform(0, 1, n)) * radius
        angle = np.pi * np.random.uniform(0, 2, n)

        x = dist * np.cos(angle) + center_x
        y = dist * np.sin(angle) + center_y
        return x, y
