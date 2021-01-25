import dill

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

import parameters as par
# from lagrange import sym_pars

f_nopars = dill.load(open("f_dyn", "rb"))

num_pars = [*par.m_links, *par.d_links, *par.m_point, par.b_cart, par.b_joint,
            par.g, par.k, par.l0]


def f_inp(t, F, qdq): return f_nopars(qdq, F, *num_pars)

# With electric motor
def f(t, qdq, F):

x0 = np.array([0, 0, np.pi/6, 0, 0, 0])
sol = solve_ivp(f, (0, 4), x0, method="RK45")
t = sol.t
y = sol.y

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, y[0, :])
ax[0].set_title("Cart position")
ax[1].plot(t, y[1, :])
ax[1].set_title("Pendulum angle")
ax[2].plot(t, y[2, :])
ax[2].set_title("Link angle")
fig.tight_layout()
fig.show()

input("Press <ENTER> to quit")
