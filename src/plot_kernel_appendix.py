import pickle as pkl
import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from utils import *


def load_model(label):
    print("Loading data for model '{}'".format(label))
    with open(os.path.join(os.path.dirname(__file__), "..", "data", label+".pkl"), "rb") as file:
        model = pkl.load(file)

    spline_basis = model["spline_model"]["spline_bases"]
    wavelength_start = model["spline_model"]["wavelength_start"]
    wavelength_stop = model["spline_model"]["wavelength_stop"]
    wavelengths = np.arange(wavelength_start, wavelength_stop+1)

    trace = model["glm_model"]["trace"]
    return trace, wavelengths, spline_basis



dataset_labels = ["Field A", "Field B", "Field C", "Field D"]
naive_dataset_names = ["naive_field_A", "naive_field_B", "naive_field_C", "naive_field_D"]

################################################################################
# naive models
trace, wavelengths, spline_basis = load_model("naive_pooled")
w_matrix = trace.posterior.w.stack(sample=("chain", "draw")).values
all_kernels = np.dot(spline_basis, w_matrix)

fig2 = plt.figure(figsize=(6.5, 5))
gs = GridSpec(3, 2, height_ratios=[2,1,1], width_ratios=[1, 1], figure=fig2, left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.4, hspace=0.4)
ax0 = fig2.add_subplot(gs[0,:])
ax1 = fig2.add_subplot(gs[1,0], sharex=ax0, sharey=ax0)
ax2 = fig2.add_subplot(gs[2,0], sharex=ax0, sharey=ax0)
ax3 = fig2.add_subplot(gs[1,1], sharex=ax0, sharey=ax0)
ax4 = fig2.add_subplot(gs[2,1], sharex=ax0, sharey=ax0)

# main plot
ax0.axhline(0, color="black", linestyle="dashed")
quantile_plot(ax0, wavelengths, all_kernels, cmap="BlueToGray", antialiased=True)
ax0.plot(wavelengths, all_kernels.mean(axis=1), linewidth=2, color="black")
ax0.set_title("Inferred kernel for baseline model (pooled data)")
plt.text(x=0.0, y=1.0, va="bottom", ha="right", s=r"$\textbf{A}$", fontsize=14, fontweight='bold', transform=ax0.transAxes)


# plots for individual fields
for i,ax in enumerate([ax1, ax2, ax3, ax4]):
    trace, wavelengths, spline_basis = load_model(naive_dataset_names[i])
    w_matrix = trace.posterior.w.stack(sample=("chain", "draw")).values
    specific_kernels = np.dot(spline_basis, w_matrix)
    ax.set_title(dataset_labels[i])

    ax.axhline(0, color="black", linestyle="dashed")
    quantile_plot(ax, wavelengths, specific_kernels, cmap="BlueToGray", antialiased=True)
    ax.plot(wavelengths, specific_kernels.mean(axis=1), linewidth=2, color="black")
    if i%2 == 0:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel('Wavelength [nm]')

    if i >= 2:
        ax.tick_params(labelleft=False)
    else:
        ax.set_ylabel('Contribution')
    plt.text(x=0.0, y=1.0, va="bottom", ha="right", s="$\\textbf{{{}}}$".format("BDCE"[i]), fontsize=14, fontweight='bold', transform=ax.transAxes)

ax0.set_ylabel('Contribution')
ax0.set_ylim([-0.15,0.15])
ax0.xaxis.set_major_locator(plt.MaxNLocator(5))
ax0.yaxis.set_major_locator(plt.MaxNLocator(3))

ax0.set_rasterized(True)
ax1.set_rasterized(True)
ax2.set_rasterized(True)
ax3.set_rasterized(True)
ax4.set_rasterized(True)
################################################################################
# full hierarchical model
trace, wavelengths, spline_basis = load_model("hierarchical_full")
w_shared_matrix = trace.posterior.w_shared.stack(sample=("chain", "draw")).values
all_kernels = np.dot(spline_basis, w_shared_matrix)

fig1 = plt.figure(figsize=(6.5, 5))
gs = GridSpec(3, 2, height_ratios=[2,1,1], width_ratios=[1, 1], figure=fig1, left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.4, hspace=0.4)
ax0 = fig1.add_subplot(gs[0,:])
ax1 = fig1.add_subplot(gs[1,0], sharex=ax0, sharey=ax0)
ax2 = fig1.add_subplot(gs[2,0], sharex=ax0, sharey=ax0)
ax3 = fig1.add_subplot(gs[1,1], sharex=ax0, sharey=ax0)
ax4 = fig1.add_subplot(gs[2,1], sharex=ax0, sharey=ax0)

# main plot
ax0.axhline(0, color="black", linestyle="dashed")
quantile_plot(ax0, wavelengths, all_kernels, cmap="BlueToGray", antialiased=True)
ax0.plot(wavelengths, all_kernels.mean(axis=1), linewidth=2, color="black")
ax0.set_title("Inferred kernel for full hierarchical model")
plt.text(x=0.0, y=1.0, va="bottom", ha="right", s=r"$\textbf{A}$", fontsize=14, fontweight='bold', transform=ax0.transAxes)

# plots for individual fields
for i,ax in enumerate([ax1, ax2, ax3, ax4]):
    ax.set_title(dataset_labels[i])
    w_specific_matrix = trace.posterior.w.isel(w_dim_0=i).stack(sample=("chain", "draw")).values
    specific_kernels = np.dot(spline_basis, w_shared_matrix+w_specific_matrix)

    ax.axhline(0, color="black", linestyle="dashed")
    quantile_plot(ax, wavelengths, specific_kernels, cmap="BlueToGray", antialiased=True)
    ax.plot(wavelengths, specific_kernels.mean(axis=1), linewidth=2, color="black")
    if i%2 == 0:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel('Wavelength [nm]')
    if i >= 2:
        ax.tick_params(labelleft=False)
    else:
        ax.set_ylabel('Contribution')
    plt.text(x=0.0, y=1.0, va="bottom", ha="right", s="$\\textbf{{{}}}$".format("BDCE"[i]), fontsize=14, fontweight='bold', transform=ax.transAxes)

ax0.set_ylabel('Contribution')
ax0.set_ylim([-0.15,0.15])
ax0.xaxis.set_major_locator(plt.MaxNLocator(5))
ax0.yaxis.set_major_locator(plt.MaxNLocator(3))

ax0.set_rasterized(True)
ax1.set_rasterized(True)
ax2.set_rasterized(True)
ax3.set_rasterized(True)
ax4.set_rasterized(True)

fig1.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "kernel_hierarchical_full.pdf"), pad_inches=0.4, dpi=300)
fig2.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "kernel_naive.pdf"), pad_inches=0.4, dpi=300)



#plt.show()
