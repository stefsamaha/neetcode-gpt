import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        y_true: float
    ) -> Tuple[NDArray[np.float64], float]:

        # ----- Forward pass -----
        z = np.dot(x, w) + b
        y_hat = 1.0 / (1.0 + np.exp(-z))

        # ----- Backward pass -----

        # dL / dy_hat
        dL_dyhat = y_hat - y_true

        # dy_hat / dz
        dyhat_dz = y_hat * (1.0 - y_hat)

        # dL / dz via chain rule
        dL_dz = dL_dyhat * dyhat_dz

        # dz / dw = x
        dL_dw = dL_dz * x

        # dz / db = 1
        dL_db = dL_dz

        return np.round(dL_dw, 5), round(dL_db, 5)


