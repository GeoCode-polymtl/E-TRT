import numpy as np
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
from matplotlib import gridspec


def plot_sum_across_slices(ax, phi, Rtype, Ttype, kk, CC, mm,
                           plane='Ck', vmin=1e0, vmax=2e4,
                           iso_levels=None, cmap='turbo', iso_style='--',
                           sumcost=True, levels=None, log=False, boundingbox=False):
    # Build cubes
    Rcube = np.stack(phi[Rtype], axis=-1)
    Tcube = np.stack(phi[Ttype], axis=-1)
    Zcube = Rcube + Tcube if sumcost else Rcube

    # Plane selection
    if plane == 'Ck':
        data = np.min(Zcube, axis=2)
        X, Y = np.meshgrid(kk, CC)
    elif plane == 'Ck_T':
        data = np.min(Tcube, axis=2)
        X, Y = np.meshgrid(kk, CC)
    elif plane == 'km':
        data = np.min(Zcube, axis=0).T
        X, Y = np.meshgrid(kk, mm)
    elif plane == 'Cm':
        data = np.min(Zcube, axis=1).T
        X, Y = np.meshgrid(CC, mm)
    else:
        raise ValueError(f"Unknown plane: {plane}")

    if boundingbox:
        # Find indices where data is below the iso threshold
        inside = data <= iso_levels[0]
        yy, xx = np.where(inside)

        if yy.size > 0 and xx.size > 0:
            # Extract the coordinates of the inside region
            x_coords = X[yy, xx]
            y_coords = Y[yy, xx]

            # Bounding box limits
            Xmin, Xmax = x_coords.min(), x_coords.max()
            Ymin, Ymax = y_coords.min(), y_coords.max()

            box = np.array([Xmin, Xmax, Ymin, Ymax])
            return box
        else:
            return None

    # Levels
    if levels is None:
        if log:
            levels = np.geomspace(vmin, vmax, 30)
        else:
            levels = np.linspace(vmin, vmax, 30)

    # Plot
    if log:
        cf = ax.contourf(X, Y, data,
                         levels=levels,
                         cmap=cmap,
                         norm=LogNorm(vmin=vmin, vmax=vmax), interpolation='nearest')
    else:
        cf = ax.contourf(X, Y, data,
                         levels=levels,
                         cmap=cmap,
                         vmin=vmin, vmax=vmax, interpolation='nearest')

    if iso_levels is not None:
        ax.contour(X, Y, data, levels=iso_levels, colors='red', linewidths=1, linestyles=iso_style)

    return cf


def plot_prob(ax, phi, Rtype, Ttype, kk, CC, mm,
              plane='Ck', vmin=1e-20, vmax=1.0,
              iso_levels=None, cmap='turbo', levels=None,
              q=0.025):
    """
    Plot posterior slices and compute MAP, mean, marginals, and credible intervals.
    :param ax: matplotlib axis to plot on
    :param phi: dictionary of cost arrays
    :param Rtype: list of keys for the reconstruction cost components
    :param Ttype: list of keys for the temperature cost components
    :param kk: array of k values
    :param CC: array of C values
    :param mm: array of m values
    :param plane: which plane to plot ('Ck', 'Ck_T', 'km', 'Cm')
    :param vmin: minimum value for color scale
    :param vmax: maximum value for color scale
    :param iso_levels: levels for iso-contours
    :param cmap: colormap
    :param levels: contour levels
    :param q: quantile for credible intervals
    :return: contourf object and statistics dictionary
    """

    # Build cubes
    Zcube = np.stack(phi[Rtype], axis=-1)
    Tcube = np.stack(phi[Ttype], axis=-1)

    # Stats for E-TRT
    def compute_stats_3d(cube):
        idx = np.unravel_index(np.argmax(cube), cube.shape)
        MAP = (CC[idx[0]], kk[idx[1]], mm[idx[2]])
        mean_C = np.sum(CC[:, None, None] * cube) / cube.sum()
        mean_k = np.sum(kk[None, :, None] * cube) / cube.sum()
        mean_m = np.sum(mm[None, None, :] * cube) / cube.sum()
        marg_C = cube.sum(axis=(1, 2))
        marg_k = cube.sum(axis=(0, 2))
        marg_m = cube.sum(axis=(0, 1))

        def CI(grid, marginal):
            cdf = np.cumsum(marginal);
            cdf /= cdf[-1]
            return grid[np.searchsorted(cdf, q)], grid[np.searchsorted(cdf, 1 - q)]

        return {"MAP": MAP,
                "mean": (mean_C, mean_k, mean_m),
                "confidence_intervals": {"C": CI(CC, marg_C),
                                         "k": CI(kk, marg_k),
                                         "m": CI(mm, marg_m)}}

    stats_Z = compute_stats_3d(Zcube)

    # Stats for temperature
    Tslice = Tcube[:, :, 0]
    idx = np.unravel_index(np.argmax(Tslice), Tslice.shape)
    MAP_T = (CC[idx[0]], kk[idx[1]])
    mean_C_T = np.sum(CC[:, None] * Tslice) / Tslice.sum()
    mean_k_T = np.sum(kk[None, :] * Tslice) / Tslice.sum()
    marg_C_T = Tslice.sum(axis=1)
    marg_k_T = Tslice.sum(axis=0)

    def CI(grid, marginal):
        cdf = np.cumsum(marginal);
        cdf /= cdf[-1]
        return grid[np.searchsorted(cdf, q)], grid[np.searchsorted(cdf, 1 - q)]

    stats_T = {"MAP": MAP_T,
               "mean": (mean_C_T, mean_k_T),
               "confidence_intervals": {"C": CI(CC, marg_C_T),
                                        "k": CI(kk, marg_k_T)}}

    # Plane selection
    if plane == 'Ck':
        data = Zcube.sum(axis=2)
        X, Y = np.meshgrid(kk, CC)
    elif plane == 'Ck_T':
        data = Tcube.sum(axis=2)
        X, Y = np.meshgrid(kk, CC)
    elif plane == 'km':
        data = Zcube.sum(axis=0).T
        X, Y = np.meshgrid(kk, mm)
    elif plane == 'Cm':
        data = Zcube.sum(axis=1).T
        X, Y = np.meshgrid(CC, mm)
    else:
        raise ValueError(f"Unknown plane: {plane}")
    data /= data.sum()

    # Plot
    levels = levels or np.geomspace(vmin, vmax, 30)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cf = ax.contourf(X, Y, data, levels=levels, cmap=cmap, norm=norm)
    if iso_levels is not None:
        ax.contour(X, Y, data, levels=iso_levels, colors='red', linewidths=1)

    return cf, {"E-TRT": stats_Z, "TRT": stats_T}


n = 33
rows, cols = 3, 11


def plot_grid_field(phi, Rtype, Ttype, title, CC, kk, mm, showERT=True, show=True):
    xticks = np.arange(len(kk))[::4]
    yticks = np.arange(len(CC))[::2]

    fig = plt.figure(figsize=(2.6 * cols + 1, 5))
    gs = gridspec.GridSpec(rows, cols, wspace=0.1, hspace=0.25)

    # Global vmin/vmax for color scaling
    vmin = min((phi[Rtype][i] + phi[Ttype][i]).min() for i in range(n))
    vmax = max((phi[Rtype][i] + phi[Ttype][i]).max() for i in range(n))

    min_coords = []
    min_values = []

    for i in range(n):
        combined = phi[Rtype][i] + phi[Ttype][i]
        coord = np.unravel_index(np.argmin(combined), combined.shape)
        min_coords.append(coord)
        min_values.append(combined[coord])

    # Find global minimum index
    global_min_index = np.argmin(min_values)

    if show:

        for i in range(n):
            row = i // cols
            col = i % cols
            ax = fig.add_subplot(gs[row, col])

            img = phi[Rtype][i] + phi[Ttype][i]
            im = ax.imshow(img, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='turbo')

            min_y, min_x = min_coords[i]
            ax.plot(min_x, min_y, 'rx', markersize=5,
                    label=f'k={np.round(kk[min_x], 3)} W/mK\nCs= {np.round(CC[min_y], 3)} MJ/m$^3$')  # Red dot at minimum location
            ax.text(min_x + 0.1, min_y + 0.1, f"{min_values[i]:.4f}", color='white', fontsize=8, ha='left', va='bottom')
            ax.legend(fontsize=8)
            ax.set_title(f"m = {np.round(mm[i], 4)}", fontsize=9)

            if row == rows - 1:
                ax.set_xticks(xticks)
                ax.set_xticklabels([f"{val:.1f}" for val in kk[::4]], rotation=45)
                ax.set_xlabel('k (W/m/K)', fontsize=10)
            else:
                ax.set_xticks([])

            if col == 0:
                ax.set_yticks(yticks)
                ax.set_yticklabels([f"{val / 1e6:.2f}" for val in CC[::2]])
                ax.set_ylabel('$C_s$ (MJ/m³/K)', fontsize=10)
            else:
                ax.set_yticks([])

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.set_label(r'$\phi_{\rho}+\phi_T$', fontsize=12)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

        plt.show()

    if showERT:

        fig = plt.figure(figsize=(2.6 * cols + 1, 5))
        gs = gridspec.GridSpec(rows, cols, wspace=0.1, hspace=0.25)

        vmin = min((phi[Rtype][i]).min() for i in range(n))
        vmax = max((phi[Rtype][i]).max() for i in range(n))

        for i in range(n):
            row = i // cols
            col = i % cols
            ax = fig.add_subplot(gs[row, col])

            img = phi[Rtype][i]
            im = ax.imshow(img, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='turbo')

            # Plot a red dot at the location of the minimum value in each subplot
            min_y, min_x = min_coords[i]
            ax.plot(min_x, min_y, 'rx', markersize=5)
            ax.set_title(f"m = {np.round(mm[i], 3)}", fontsize=9)

            if row == rows - 1:
                ax.set_xticks(xticks)
                ax.set_xticklabels([f"{val:.1f}" for val in kk[::4]], rotation=45)
                ax.set_xlabel('k (W/m/K)', fontsize=10)
            else:
                ax.set_xticks([])

            if col == 0:
                ax.set_yticks(yticks)
                ax.set_yticklabels([f"{val / 1e6:.2f}" for val in CC[::2]])
                ax.set_ylabel('$C_s$ (MJ/m³/K)', fontsize=10)
            else:
                ax.set_yticks([])

        # Vertical colorbar on the right
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.set_label(r'$\phi_R$', fontsize=12)

        plt.show()

    return min_coords, global_min_index
