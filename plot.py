import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc

plt.style.use("seaborn-bright")
plt.rc("font", family="serif")
# plt.rc("font", serif="STIXGeneral")
plt.rc("xtick", labelsize="x-small")
plt.rc("ytick", labelsize="x-small")
plt.rc("xtick.minor", visible=True)
plt.rc("ytick.minor", visible=True)
plt.rc("axes", grid=True)
plt.rc("axes", xmargin=0)
# plt.rc("text", usetex=True)
plt.rc('axes', axisbelow=True)

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#2ecc71", "#3498db",
                                                    "#e74c3c", "#f1c40f",
                                                    "#e67e22", "#9b59b6"])

if __name__ == "__main__":
    t = np.linspace(0, 10, 200)

    fig, ax = plt.subplots()
    ax.plot(t, np.sin(t), color="k")
    ax.set_title("A nice example", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (-)")
    fig.show()

    input("Press <ENTER> to quit")
