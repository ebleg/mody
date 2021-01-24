import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt

import lagrange

t = np.linspace(0, 1)
x0 = np.array([0, 0, np.pi/12, 0, 0, 0])
y = odeint(lagrange.f, x0, t)

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, y[:, 3])
ax[0].set_title("Cart position")
ax[1].plot(t, y[:, 4])
ax[1].set_title("Pendulum angle")
ax[2].plot(t, y[:, 5])
ax[2].set_title("Link angle")
fig.show()

input("Press <ENTER> key to quit")
