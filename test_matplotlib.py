import numpy as np
import matplotlib.pyplot as plt


def neon_plot(x, y, ax=None):
    if ax is None:
        ax = plt.gca()
    (line,) = ax.plot(x, y, lw=1, zorder=6)
    for cont in range(6, 1, -1):
        ax.plot(x, y, lw=cont, color=line.get_color(), zorder=5, alpha=0.05)
    return ax


# Styling
repo = "https://raw.githubusercontent.com/nicoguaro/matplotlib_styles/master"
style = repo + "/styles/neon.mplstyle"
plt.style.use(style)

# Plotting
x = np.linspace(0, 4, 100)
y = np.sin(np.pi * x + 1e-6) / (np.pi * x + 1e-6)
plt.figure(figsize=(6, 4))
for cont in range(5):
    neon_plot(x, y / (cont + 1))


plt.xlabel("One axis")
plt.ylabel("The other axis")
plt.grid(zorder=3, alpha=0.2)
plt.savefig("neon_example.png", dpi=300, bbox_inches="tight")
plt.show()
