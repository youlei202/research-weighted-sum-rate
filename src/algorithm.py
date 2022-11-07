import math
import numpy as np


def weighted_minimum_mean_square_error(
    simulator,
    gain_mat_mW,
    Rx_powers_mW,
    Rx_max_powers_mW,
    Rx_weights,
    Tx_idx_shift=0,
    Rx_idx_shift=0,
):
    def channel_gain(i):
        k = simulator.Tx_of[i + Rx_idx_shift] - Tx_idx_shift
        gain = math.sqrt(gain_mat_mW[k][i])
        return gain

    Tx_num, Rx_num = gain_mat_mW.shape
    p = np.array(Rx_powers_mW[Rx_idx_shift : Rx_idx_shift + gain_mat_mW.shape[1]])
    alpha = Rx_weights[Rx_idx_shift : Rx_idx_shift + gain_mat_mW.shape[1]]
    v = np.vectorize(math.sqrt)(p)
    w = np.zeros(Rx_num)
    u = np.zeros(Rx_num)
    h = [channel_gain(i) for i in range(Rx_num)]

    t = 0
    while t < 5000:
        for i in range(Rx_num):
            u[i] = (
                h[i]
                * v[i]
                / (
                    np.array(
                        [
                            np.sum(
                                np.take(
                                    v**2,
                                    simulator.Rx_of[j + Tx_idx_shift] - Rx_idx_shift,
                                )
                            )
                            for j in range(Tx_num)
                        ]
                    ).dot(gain_mat_mW[:, i])
                    + simulator.noise_mW
                )
            )

        for i in range(Rx_num):
            w[i] = 1 / (1 - u[i] * h[i] * v[i])

        for i in range(Rx_num):
            k = simulator.Tx_of[i + Rx_idx_shift] - Tx_idx_shift
            v[i] = (
                alpha[i]
                * h[i]
                * u[i]
                * w[i]
                / gain_mat_mW[k, :].dot(alpha * w * (u**2))
            )
        v = v.clip(min=0, max=np.vectorize(math.sqrt)(Rx_max_powers_mW))
        p_prev = p
        p = v**2

        t += 1

    new_powers = Rx_powers_mW.copy()
    for i in range(Rx_num):
        new_powers[i + Rx_idx_shift] = p[i]

    print(new_powers)
    print(
        f"Weighted Sum Rate: {simulator.weighted_sum_rate_Gbps(new_powers, Rx_weights=Rx_weights)}"
    )
    print(f"Round {t}", np.linalg.norm(p - p_prev))
    print(f"\t Power: {new_powers}")

    return p
