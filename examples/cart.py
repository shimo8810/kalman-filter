import numpy as np
import matplotlib.pyplot as plt

from kalman_filter import KalmanFilter

s1 = 1
s2 = 10


class Cart:
    def __init__(self):
        self.x = 0
        self.v = 0

    def __call__(self):
        a = np.random.normal(0, s1)
        n = np.random.normal(0, s2)
        self.x += self.v + a / 2.0
        self.v += a

        # true x, ovserved x, true v
        return self.x, self.x + n, self.v


if __name__ == "__main__":
    # definitions
    F = np.array([[1, 1], [0, 1]])
    H = np.array([[1, 0]])
    Q = np.array([[1 / 4, 1 / 2], [1 / 2, 1]]) * (s1**2)
    R = np.array([[s2**2]])

    # init
    x = np.array([0, 0])
    P = np.array([[0, 0], [0, 0]])

    kf = KalmanFilter(F, H, Q, R)

    cart = Cart()
    vs = [0]  # filterd v
    ws = [0]  # true v
    xs = [0]  # filterd x
    ys = [0]  # true x
    zs = [0]  # observed x
    for _ in range(100):
        y, z, w = cart()
        x, P = kf.predict(x, P)
        x, P = kf.update(x, P, z)

        vs.append(x[1])
        ws.append(w)
        xs.append(x[0])
        ys.append(y)
        zs.append(z)

    # plot graph
    figs, axes = plt.subplots(1, 2)
    axes[1].plot(vs, label='filterd v', marker='.')
    axes[1].plot(ws, label='true v', marker='x')
    axes[0].plot(xs, label='filterd x', marker='.')
    axes[0].plot(ys, label='true x', marker='x')
    axes[0].plot(zs, label='observed x', marker='+')
    axes[0].set_ylabel('x')
    axes[1].set_ylabel('v')
    axes[0].set_xlabel('step')
    axes[1].set_xlabel('step')
    axes[0].legend()
    axes[1].legend()
    axes[0].grid()
    axes[1].grid()
    figs.tight_layout()
    plt.show()