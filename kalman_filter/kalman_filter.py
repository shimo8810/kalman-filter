from typing import Tuple

import numpy as np
import numpy.linalg as LA
from numpy import ndarray


class KalmanFilter:
    """
    Kalman Filter Class
    """
    def __init__(self, F: ndarray, H: ndarray, Q: ndarray, R: ndarray):
        """
        Kalman Filter Class Constructor
        Args:
            - F(ndarray): the state-transition model
            - H(ndarray): the observation model
            - Q(ndarray): the covariance of the process noise
            - R(ndarray): the covariance of the observation noise
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R

        self.dimx = F.shape[0]
        self.dimz = H.shape[0]

    def predict(self, x: ndarray, P: ndarray) -> Tuple[ndarray, ndarray]:
        """predict a priori state estimate

        Args:
            - `x(ndarray)`: the a posteriori state estimate at the previous timestep
            - `P(ndarray)`: the a posteriori estimate covariance at from the previous timestep

        Returns:
            - `ndarray`: the a priori state estimate at the current timestep
            - `ndarray`: the a priori estimate covariance at the current timestep
        """
        x_predicted = self.F @ x
        P_predicted = self.F @ P @ self.F.T + self.Q

        return x_predicted, P_predicted

    def update(self, x: ndarray, P: ndarray,
               z: ndarray) -> Tuple[ndarray, ndarray]:
        """update a posteriori state estimate

        Args:
            - `x(ndarray)`: the a priori state estimate at the current timestep
            - `P(ndarray)`: the a priori estimate covariance at the current timestep
            - `z(ndarray)`: the observation of the true state at the current timestep
        Returns:
            - `ndarray`: the a posteriori state estimate at the current timestep
            - `ndarray`: the a posteriori estimate covariance at from the current timestep
        """
        # pre-fit residual
        y = z - self.H @ x
        # pre-fit residual covariance
        S = self.H @ P @ self.H.T + self.R
        # optimal Kalman gain
        K = P @ self.H.T @ LA.inv(S)

        x_updated = x + K @ y
        P_updated = (np.identity(self.dimx) - K @ self.H) @ P

        return x_updated, P_updated
