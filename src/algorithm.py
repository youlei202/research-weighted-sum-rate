import math
import numpy as np


def wmmse_iteration(
    simulator, gain_mat_mW, h, u, w, p, p_max, alpha, Tx_idx_shift, Rx_idx_shift
):
    Tx_num, Rx_num = gain_mat_mW.shape
    v = np.vectorize(math.sqrt)(p)
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
            alpha[i] * h[i] * u[i] * w[i] / gain_mat_mW[k, :].dot(alpha * w * (u**2))
        )
    v = v.clip(min=0, max=np.vectorize(math.sqrt)(p_max))
    p = v**2
    return p, u, w


def wmmse(
    simulator,
    gain_mat_mW,
    Rx_powers_mW,
    Rx_max_powers_mW,
    Rx_weights,
    Tx_idx_shift=0,
    Rx_idx_shift=0,
    max_iter=5000,
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

    powers_list = []
    t = 0
    while t < max_iter:
        p_prev = p
        p, u, w = wmmse_iteration(
            simulator,
            gain_mat_mW,
            h=h,
            u=u,
            w=w,
            p=p,
            p_max=Rx_max_powers_mW,
            alpha=alpha,
            Tx_idx_shift=Tx_idx_shift,
            Rx_idx_shift=Rx_idx_shift,
        )

        t += 1

    p_final = Rx_powers_mW.copy()
    for i in range(Rx_num):
        p_final[i + Rx_idx_shift] = p[i]
    print(
        f"Weighted Sum Rate: {simulator.weighted_sum_rate_Gnats(p_final, Rx_weights=Rx_weights)}",
        f"Convergence Error at Round {t}",
        np.linalg.norm(p - p_prev),
    )
    return p_final
