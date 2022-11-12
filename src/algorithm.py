import math
import numpy as np
import pandas as pd

# from src.network_sim import NetworkSimulator
from src.experiment import train_sc_models


def wmmse_iteration(simulator, h, u, w, p, p_max, alpha, Tx_idx_shift, Rx_idx_shift):
    gain_netA = simulator.get_gain_mat(part="A", unit="mW")
    Tx_num, Rx_num = gain_netA.shape
    v = np.vectorize(math.sqrt)(p)
    for i in range(Rx_num):
        u[i] = (
            h[i]
            * v[i]
            / (simulator.Rx_signal_and_interference_A_to_A(p, i=i) + simulator.noise_mW)
        )

    for i in range(Rx_num):
        w[i] = 1 / (1 - u[i] * h[i] * v[i])

    for i in range(Rx_num):
        k = simulator.Tx_of[i + Rx_idx_shift] - Tx_idx_shift
        v[i] = alpha[i] * h[i] * u[i] * w[i] / gain_netA[k, :].dot(alpha * w * (u**2))
    v = v.clip(min=0, max=np.vectorize(math.sqrt)(p_max))
    p = v**2
    return p, u, w


def wmmse(
    simulator,
    Rx_powers_mW,
    Rx_max_powers_mW,
    Rx_weights,
    Tx_idx_shift=0,
    Rx_idx_shift=0,
    max_iter=5000,
):
    def channel_gain(i):
        gain_netA = simulator.get_gain_mat(part="A", unit="mW")
        k = simulator.Tx_of[i + Rx_idx_shift] - Tx_idx_shift
        channel_gain_netA = math.sqrt(gain_netA[k][i])
        return channel_gain_netA

    Tx_num, Rx_num = simulator.num_Tx_netA, simulator.num_Rx_netA
    p = np.array(Rx_powers_mW[:Rx_num])
    alpha = Rx_weights[:Rx_num]
    v = np.vectorize(math.sqrt)(p)
    w = np.zeros(Rx_num)
    u = np.zeros(Rx_num)

    powers_list = []
    t = 0
    while t < max_iter:
        p_prev = p
        h = [channel_gain(i) for i in range(Rx_num)]
        p, u, w = wmmse_iteration(
            simulator,
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
        f"Weighted Sum Rate: {simulator.weighted_sum_rate_Gnats(p_final, Rx_weights=Rx_weights, part='A')}",
        f"Convergence Error at Round {t}",
        np.linalg.norm(p - p_prev),
    )
    return p_final


def stochastic_wmmse_iteration(
    simulator,
    h,
    u,
    w,
    a,
    b,
    p,
    p_max,
    alpha,
    signal_plus_interferences_and_noise_A,
):
    v = np.vectorize(math.sqrt)(p)
    gain_netA = simulator.get_gain_mat(part="A", unit="mW")
    for i in range(simulator.num_Rx_netA):
        u[i] = h[i] * v[i] / signal_plus_interferences_and_noise_A[i]

    for i in range(simulator.num_Rx_netA):
        w[i] = 1 / (1 - u[i] * h[i] * v[i])

    for i in range(simulator.num_Rx_netA):
        k = simulator.Tx_of[i]
        a[i] = gain_netA[k, :].dot(alpha * w * (u**2))
        b[i] = alpha[i] * h[i] * u[i] * w[i]
        v[i] = b[i] / a[i]
    v = v.clip(min=0, max=np.vectorize(math.sqrt)(p_max))
    p = v**2
    return p, u, w, a, b


def stochastic_wmmse(
    simulator,
    Rx_powers_mW,
    Rx_max_powers_mW,
    Rx_weights,
    interference_mode="local",
    interference_models=[],
    netB_power_mode=None,
    power_corr_mat=None,
    max_iter=5000,
    sc_mode="sc",
):
    def channel_gain(i):
        k = simulator.Tx_of[i]
        gain = math.sqrt(simulator.gain_mat_mW[k][i])
        return gain

    if interference_mode not in ["local", "original", "sc_estimate", "lr_estimate"]:
        print("Unknown noise mode")
        return

    p = np.array(Rx_powers_mW[: simulator.num_Rx_netA])
    alpha = Rx_weights[: simulator.num_Rx_netA]
    v = np.vectorize(math.sqrt)(p)
    w = np.zeros(simulator.num_Rx_netA)
    u = np.zeros(simulator.num_Rx_netA)
    a = np.zeros(simulator.num_Rx_netA)
    b = np.zeros(simulator.num_Rx_netA)

    power_corr_mat

    t = 0
    rate_list = []
    while t < max_iter:
        p_prev = p
        h = [channel_gain(i) for i in range(simulator.num_Rx_netA)]

        # dice = np.uniform.randInt()
        if netB_power_mode == "dependent":
            p_netB = power_corr_mat.dot(p).clip(min=0, max=Rx_max_powers_mW)
        elif netB_power_mode == "zero":
            p_netB = [1e-20 for i in range(simulator.num_Rx_netB)]
        elif netB_power_mode == "random":
            p_netB = np.random.uniform(0, Rx_max_powers_mW, simulator.num_Rx_netB)
        elif netB_power_mode == "uniform":
            p_netB = [50 for i in range(simulator.num_Rx_netB)]
        else:
            p_netB = Rx_powers_mW[simulator.num_Rx_netA :]

        p_netAB = np.append(p, p_netB)
        if interference_mode == "local":
            signal_plus_interferences_and_noise_A = (
                np.array(simulator.Rx_signal_and_interference_A_to_A(p))
                + simulator.noise_mW
            )
        elif interference_mode == "original":
            signal_plus_interferences_and_noise_A = (
                np.array(simulator.Rx_signal_and_interference_AB_to_A(p_netAB))
                + simulator.noise_mW
            )
        elif interference_mode == "sc_estimate":
            signal_and_interferences_A = simulator.Rx_signal_and_interference_AB_to_A(
                p_netAB
            )
            result = []
            for i in range(simulator.num_Rx_netA):
                if sc_mode == "random":
                    synthetic_i = np.mean(
                        np.random.dirichlet(np.ones(simulator.num_Rx_netA - 1))
                        * pd.DataFrame(signal_and_interferences_A).drop(i, axis=0)[0]
                    )
                elif sc_mode == "center":
                    synthetic_i = (
                        pd.DataFrame(signal_and_interferences_A)
                        .drop(i, axis=0)[0]
                        .mean()
                    )
                else:
                    sc_X_test = pd.DataFrame(signal_and_interferences_A).drop(
                        i, axis=0
                    )[0]
                    synthetic_i = interference_models[i].predict(sc_X_test)
                result.append(synthetic_i + simulator.noise_mW)

            if np.random.rand() <= ((1 - t / max_iter) * 0.2) ** 2:
                signal_plus_interferences_and_noise_A = np.array(result)
            else:
                signal_plus_interferences_and_noise_A = (
                    np.array(simulator.Rx_signal_and_interference_AB_to_A(p_netAB))
                    + simulator.noise_mW
                )
        elif interference_mode == "lr_estimate":
            signal_plus_interferences_and_noise_A = [
                interference_models[i].predict([p]) + simulator.noise_mW
                for i in range(simulator.num_Rx_netA)
            ]
        p, u, w, a, b = stochastic_wmmse_iteration(
            simulator,
            h=h,
            u=u,
            w=w,
            a=a,
            b=b,
            p=p,
            p_max=Rx_max_powers_mW,
            alpha=alpha,
            signal_plus_interferences_and_noise_A=signal_plus_interferences_and_noise_A,
        )
        rate = simulator.weighted_sum_rate_Gnats(
            np.append(p, p_netB), Rx_weights=Rx_weights, part="A"
        )
        rate_list.append(rate)
        # simulator.update_gain_matrix()
        t += 1

    return rate_list
