import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc

plt.rc("xtick", labelsize="x-small")
plt.rc("ytick", labelsize="x-small")
plt.rc("xtick.minor", visible=True)
plt.rc("ytick.minor", visible=True)
plt.rc("axes", grid=True)
plt.rc("axes", xmargin=0)
plt.rc("text", usetex=True)

t = np.linspace(0, 10, 200)

fig, ax = plt.subplots()
ax.plot(t, np.sin(t), color="k")
ax.set_title("A nice example", fontweight="bold")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (-)")
fig.show()

input("Press <ENTER> to quit")