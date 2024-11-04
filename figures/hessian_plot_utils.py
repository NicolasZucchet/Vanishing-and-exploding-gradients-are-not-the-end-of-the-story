from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import jax.numpy as jnp


def extend_hessian(H, sizes, pads):
    cum_sums = [0] + list(np.cumsum(sizes))
    cs_pads = [int(0)] + list(np.cumsum(pads))
    extended_H = (
        np.ones((cum_sums[-1] + np.sum(pads).astype(int), cum_sums[-1] + np.sum(pads).astype(int)))
        * np.nan
    )
    positions = []
    for i, size in enumerate(sizes):
        positions.append((cum_sums[i] + cum_sums[i + 1]) / 2 + cs_pads[i] - 0.5)
        for j, size_ in enumerate(sizes):
            extended_H[
                cum_sums[i] + cs_pads[i] : cum_sums[i + 1] + cs_pads[i],
                cum_sums[j] + cs_pads[j] : cum_sums[j + 1] + cs_pads[j],
            ] = H[cum_sums[i] : cum_sums[i + 1], cum_sums[j] : cum_sums[j + 1]]

    eigvals, eigvects = np.linalg.eigh(H)
    eigvals = eigvals[::-1]
    eigvects = eigvects[:, ::-1]
    extended_eigvects = (
        np.ones((cum_sums[-1] + np.sum(pads).astype(int), eigvects.shape[1])) * np.nan
    )
    for i, size in enumerate(sizes):
        extended_eigvects[cum_sums[i] + cs_pads[i] : cum_sums[i + 1] + cs_pads[i], :] = eigvects[
            cum_sums[i] : cum_sums[i + 1], :
        ]

    return extended_H, positions, eigvals, extended_eigvects


def plot(ax, positions, legends, extended_H, extended_eigvects, eigvals, i):
    # take max of the non nan values
    max_val = np.nanmax(np.abs(extended_H))
    im = ax[0, i].imshow(
        extended_H,
        cmap="RdBu",
        norm=matplotlib.colors.SymLogNorm(linthresh=1, linscale=1.75, vmin=-max_val, vmax=max_val),
    )
    # xaxis on top
    ax[0, i].xaxis.tick_top()
    # remove lines and ticks but not labels
    ax[0, i].spines["bottom"].set_visible(False)
    ax[0, i].spines["left"].set_visible(False)
    # remove line in ticks
    ax[0, i].xaxis.set_ticks_position("none")
    ax[0, i].yaxis.set_ticks_position("none")

    ax[0, i].set_xticks(positions, legends)
    ax[0, i].set_yticks(positions, legends)
    ax[0, i].tick_params(axis="y", pad=0)
    ax[0, i].tick_params(axis="x", pad=-1)

    divider = make_axes_locatable(ax[0, i])
    cax = divider.append_axes("right", size=0.05, pad=0.12)
    if max_val < 1e6:
        cb = plt.colorbar(im, cax=cax, ticks=[-10000, -100, -1, 0, 1, 100, 10000])
    elif max_val < 1e8:
        cb = plt.colorbar(im, cax=cax, ticks=[-1e6, -1e3, -1, 0, 1, 1e3, 1e6])
    else:
        cb = plt.colorbar(im, cax=cax, ticks=[-1e8, -10000, -1, 0, 1, 1e4, 1e8])
    cb.ax.yaxis.set_tick_params(which="minor", length=0)
    cb.ax.yaxis.set_tick_params(which="major", length=2, pad=2, width=0.5)
    cb.outline.set_linewidth(0.5)

    im = ax[1, i].imshow(
        extended_eigvects[:, :10].T,
        norm=matplotlib.colors.Normalize(vmin=-1, vmax=1),
        cmap="PRGn",
    )
    ax[1, i].spines["bottom"].set_visible(False)
    ax[1, i].spines["left"].set_visible(False)
    ax[1, i].xaxis.set_ticks_position("none")
    ax[1, i].yaxis.set_ticks_position("none")
    ax[1, i].set_yticks(
        [0], ["D"], color="white"
    )  # fake lable to make sur that sizes are consistent
    ax[1, i].set_xticks([])
    # cb = plt.colorbar(im, ax=ax[1, i],
    # set width of the cb to be the same as the one above
    divider = make_axes_locatable(ax[1, i])
    cax = divider.append_axes("right", size=0.05, pad=0.12)
    cb = plt.colorbar(im, cax=cax, ticks=[-1, 0, 1])
    cb.ax.yaxis.set_tick_params(which="minor", length=0)
    cb.ax.yaxis.set_tick_params(which="major", length=2, pad=2, width=0.5)
    cb.outline.set_linewidth(0.5)

    ax[2, i].plot(eigvals)
    ax[2, i].set_yscale("log")
    ax[2, i].set_xlabel("Index eigenvalue")
    if i == 0:
        ax[2, i].set_ylabel("Eigenvalue")
    ax[2, i].minorticks_off()
