import math
from abc import ABCMeta, abstractmethod

import numpy as np


class PathLoss(metaclass=ABCMeta):
    """Multi-frequency model
    Applying across 0.5-100 GHz band
    """

    SPEED_OF_LIGHT = 299792458  # m/s

    def in_dBm(self, frequency_Hz, distance_m):
        return 0 - self.in_dB(frequency_Hz=frequency_Hz, distance_m=distance_m)

    def in_mW(self, frequency_Hz, distance_m):
        return 10 ** (
            self.in_dBm(frequency_Hz=frequency_Hz, distance_m=distance_m) / 10
        )

    @classmethod
    def free_space_1m(cls, frequency_Hz):
        """Free space path loss at 1 meter
        Return path loss in dB
        """
        return 20 * math.log10(4 * math.pi * frequency_Hz / cls.SPEED_OF_LIGHT)

    @classmethod
    def shadow_fading(cls, std):
        return np.random.normal(loc=0, scale=std)

    @abstractmethod
    def in_dB(cls, frequency_Hz, distance_m):
        pass


class PathLossCloseInFreeSpace(PathLoss):
    def __init__(self, path_loss_exponent, shadow_fading_std):
        self.path_loss_exponent = path_loss_exponent
        self.shadow_fading_std = shadow_fading_std

    def in_dB(self, frequency_Hz, distance_m):
        return (
            self.free_space_1m(frequency_Hz)
            + 10 * self.path_loss_exponent * math.log10(distance_m)
            + self.shadow_fading(std=self.shadow_fading_std)
        )


class PathLossCloseInFreeSpaceFrequencyDependent(PathLossCloseInFreeSpace):
    def __init__(self, path_loss_exponent, slope, reference_freq_Hz, shadow_fading_std):
        super().__init__(
            path_loss_exponent=path_loss_exponent, shadow_fading_std=shadow_fading_std
        )
        self.slope = slope
        self.reference_freq_Hz = reference_freq_Hz

    def in_dB(self, frequency_Hz, distance_m):
        return (
            self.free_space_1m(frequency_Hz)
            + 10
            * self.path_loss_exponent
            * (1 + self.slope * (frequency_Hz / self.reference_freq_Hz - 1))
            * math.log10(distance_m)
            + self.shadow_fading(std=self.shadow_fading_std)
        )


class PathLossInHIndoorOfficeLOS(PathLossCloseInFreeSpace):
    def __init__(self, path_loss_exponent=1.73, shadow_fading_std=3.02):
        super().__init__(
            path_loss_exponent=path_loss_exponent, shadow_fading_std=shadow_fading_std
        )


class PathLossInHIndoorOfficeNLOSSingle(PathLossCloseInFreeSpaceFrequencyDependent):
    def __init__(
        self,
        path_loss_exponent=3.19,
        slope=0.06,
        reference_freq_Hz=24.2 * 1e9,
        shadow_fading_std=8.29,
    ):
        super().__init__(
            path_loss_exponent=path_loss_exponent,
            slope=slope,
            reference_freq_Hz=reference_freq_Hz,
            shadow_fading_std=shadow_fading_std,
        )


class PathLossInHIndoorOfficeNLOSDual(PathLossCloseInFreeSpaceFrequencyDependent):
    def __init__(
        self,
        path_loss_exponent=2.51,
        path_loss_exponent_2=4.25,
        slope=0.12,
        slope_2=0.04,
        reference_freq_Hz=24.1 * 1e9,
        distance_split=7.8,
        shadow_fading_std=7.65,
    ):
        super().__init__(
            path_loss_exponent=path_loss_exponent,
            slope=slope,
            reference_freq_Hz=reference_freq_Hz,
            shadow_fading_std=shadow_fading_std,
        )
        self.path_loss_exponent_2 = path_loss_exponent_2
        self.slope_2 = slope_2
        self.distance_split = distance_split

    def in_dB(self, frequency_Hz, distance_m):
        if distance_m <= 1:
            raise ValueError
        elif distance_m <= self.distance_split:
            return super().in_dB(frequency_Hz=frequency_Hz, distance_m=distance_m)
        else:
            return super().in_dB(
                frequency_Hz=frequency_Hz, distance_m=self.distance_split
            ) + 10 * self.path_loss_exponent_2 * (
                1 + self.slope_2 * (frequency_Hz / self.reference_freq_Hz - 1)
            ) * math.log10(
                distance_m / self.distance_split
            )


class PathLossInHShoppingMallsLOS(PathLossCloseInFreeSpace):
    def __init__(self, pathloss_component=1.73, shadow_fading_std=2.01):
        super().__init__(
            path_loss_exponent=pathloss_component, shadow_fading_std=shadow_fading_std
        )


class PathLossInHShoppingMallsNLOSSingle(PathLossCloseInFreeSpaceFrequencyDependent):
    def __init__(
        self,
        path_loss_component=2.59,
        slope=0.01,
        reference_freq_Hz=39.5 * 1e9,
        shadow_fading_std=7.4,
    ):
        super().__init__(
            path_loss_exponent=path_loss_component,
            slope=slope,
            reference_freq_Hz=reference_freq_Hz,
            shadow_fading_std=shadow_fading_std,
        )


class PathLossInHShoppingMallsNLOSDual(PathLossCloseInFreeSpaceFrequencyDependent):
    def __init__(
        self,
        path_loss_exponent=2.43,
        path_loss_exponent_2=8.36,
        slope=0.01,
        slope_2=0.39,
        reference_freq_Hz=39.5 * 1e9,
        distance_split=110,
        shadow_fading_std=6.26,
    ):
        super().__init__(
            path_loss_exponent=path_loss_exponent,
            slope=slope,
            reference_freq_Hz=reference_freq_Hz,
            shadow_fading_std=shadow_fading_std,
        )
        self.path_loss_exponent_2 = path_loss_exponent_2
        self.slope_2 = slope_2
        self.distance_split = distance_split

    def in_dB(self, frequency_Hz, distance_m):
        if distance_m <= 1:
            raise ValueError
        elif distance_m <= self.distance_split:
            return super().in_dB(frequency_Hz=frequency_Hz, distance_m=distance_m)
        else:
            return super().in_dB(
                frequency_Hz=frequency_Hz, distance_m=self.distance_split
            ) + 10 * self.path_loss_exponent_2 * (
                1 + self.slope_2 * (frequency_Hz / self.reference_freq_Hz - 1)
            ) * math.log10(
                distance_m / self.distance_split
            )
