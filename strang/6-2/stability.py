import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_stability_regions():
    # 図の設定
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # 共通定義
    # z は単位円 e^(i*theta) を表す
    # MATLAB: z = exp(1i*pi*(0:200)/100);
    theta = np.linspace(0, 2, 201) * np.pi
    z = np.exp(1j * theta)
    r = z - 1

    # 軸描画の補助関数
    def plot_axes(ax, xlim, ylim):
        ax.plot(xlim, [0, 0], "k-", lw=0.5)  # x軸
        ax.plot([0, 0], ylim, "k-", lw=0.5)  # y軸
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
        ax.grid(True, linestyle=":", alpha=0.6)

    # ---------------------------------------------------------
    # 1. Adams-Bashforth（左上）
    # ---------------------------------------------------------
    ax = axs[0, 0]
    plot_axes(ax, [-2.5, 0.5], [-1.5, 1.5])
    ax.set_title("Adams-Bashforth")

    # 次数 1
    s = 1.0
    curve = r / s
    ax.plot(np.real(curve), np.imag(curve), label="p=1")

    # 次数 2
    s = (3 - 1 / z) / 2
    curve = r / s
    ax.plot(np.real(curve), np.imag(curve), label="p=2")

    # 次数 3
    s = (23 - 16 / z + 5 / (z**2)) / 12
    curve = r / s
    ax.plot(np.real(curve), np.imag(curve), label="p=3")
    ax.legend(loc="best", fontsize="small")

    # ---------------------------------------------------------
    # 2. Adams-Moulton（右上）
    # ---------------------------------------------------------
    ax = axs[0, 1]
    plot_axes(ax, [-7, 1], [-4, 4])
    ax.set_title("Adams-Moulton")

    # 次数 3
    s = (5 * z + 8 - 1 / z) / 12
    curve = r / s
    ax.plot(np.real(curve), np.imag(curve), label="p=3")

    # 次数 4
    s = (9 * z + 19 - 5 / z + 1 / (z**2)) / 24
    curve = r / s
    ax.plot(np.real(curve), np.imag(curve), label="p=4")

    # 次数 5
    s = (251 * z + 646 - 264 / z + 106 / (z**2) - 19 / (z**3)) / 720
    curve = r / s
    ax.plot(np.real(curve), np.imag(curve), label="p=5")

    # 次数 6（式は d を使う）
    d = 1 - 1 / z
    s = 1 - d / 2 - d**2 / 12 - d**3 / 24 - 19 * (d**4) / 720 - 3 * (d**5) / 160
    curve = d / s
    ax.plot(np.real(curve), np.imag(curve), label="p=6")
    ax.legend(loc="best", fontsize="small")

    # ---------------------------------------------------------
    # 3. Backward Differentiation（左下）
    # ---------------------------------------------------------
    ax = axs[1, 0]
    plot_axes(ax, [-15, 35], [-25, 25])
    ax.set_title("Backward Differentiation")

    # 次数 1〜5
    # MATLAB: r = 0; for i = 1:5, r = r+(d.^i)/i; plot(r), end
    d = 1 - 1 / z
    r_bdf = 0
    for i in range(1, 6):
        r_bdf = r_bdf + (d**i) / i
        ax.plot(np.real(r_bdf), np.imag(r_bdf), label=f"p={i}")
    ax.legend(loc="best", fontsize="small")

    # ---------------------------------------------------------
    # 4. Runge-Kutta（右下）
    # ---------------------------------------------------------
    ax = axs[1, 1]
    plot_axes(ax, [-5, 2], [-3.5, 3.5])
    ax.set_title("Runge-Kutta")

    # Newton反復で境界を追跡
    # 次数 1
    w = 0 + 0j
    W = [w]
    for i in range(1, len(z)):
        # w = w-(1+w-z(i));
        w = w - (1 + w - z[i])
        W.append(w)
    W = np.array(W)
    ax.plot(np.real(W), np.imag(W), label="p=1")

    # 次数 2
    w = 0 + 0j
    W = [w]
    for i in range(1, len(z)):
        # w = w-(1+w+.5*w^2-z(i)^2)/(1+w);
        numerator = 1 + w + 0.5 * (w**2) - (z[i] ** 2)
        denominator = 1 + w
        w = w - numerator / denominator
        W.append(w)
    W = np.array(W)
    ax.plot(np.real(W), np.imag(W), label="p=2")

    # 次数 3
    w = 0 + 0j
    W = [w]
    for i in range(1, len(z)):
        # w = w-(1+w+.5*w^2+w^3/6-z(i)^3)/(1+w+w^2/2);
        numerator = 1 + w + 0.5 * (w**2) + (w**3) / 6 - (z[i] ** 3)
        denominator = 1 + w + 0.5 * (w**2)
        w = w - numerator / denominator
        W.append(w)
    W = np.array(W)
    ax.plot(np.real(W), np.imag(W), label="p=3")

    # 次数 4
    w = 0 + 0j
    W = [w]
    for i in range(1, len(z)):
        # w = w-(1+w+.5*w^2+w^3/6+w.^4/24-z(i)^4)/(1+w+w^2/2+w.^3/6);
        numerator = 1 + w + 0.5 * (w**2) + (w**3) / 6 + (w**4) / 24 - (z[i] ** 4)
        denominator = 1 + w + 0.5 * (w**2) + (w**3) / 6
        w = w - numerator / denominator
        W.append(w)
    W = np.array(W)
    ax.plot(np.real(W), np.imag(W), label="p=4")
    ax.legend(loc="best", fontsize="small")

    fig.tight_layout()
    output_path = Path(__file__).with_name("stability.png")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.show()


if __name__ == "__main__":
    plot_stability_regions()
