import dill

import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt

f = dill.load(open("f_dyn", "rb"))

t = np.linspace(0, 4, num=300)
x0 = np.array([0, np.pi/6, np.pi/6, 0, 0, 0])
y = odeint(f, x0, t)

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, y[:, 0])
ax[0].set_title("Cart position")
ax[1].plot(t, y[:, 1])
ax[1].set_title("Pendulum angle")
ax[2].plot(t, y[:, 2])
ax[2].set_title("Link angle")
fig.tight_layout()
fig.show()

input("Press <ENTER> to quit")
