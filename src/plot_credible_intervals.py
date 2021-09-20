import pickle as pkl
import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from utils import *


## Plot naive model(s)
naive_models = [
    ("naive_pooled", "Marginal parameter distribution of baseline model"), # uses all data
]

plot_args = {
    "w": {"color": "C1", "label": "$\\textrm{weights}~w_{k}, k\in\{1,\dots,11\}$", "markersize": 4, "interquartile_linewidth": 2, "credible_linewidth": 1},
    "sd": {"color": "C1", "label": "$\\textrm{deviation}~\\sigma$", "markersize": 4, "interquartile_linewidth": 2, "credible_linewidth": 1},
    "bias": {"color": "C1", "label": "$\\textrm{bias}~b$", "markersize": 4, "interquartile_linewidth": 2, "credible_linewidth": 1},
}

for label,name in naive_models:
    with open(os.path.join(os.path.dirname(__file__), "..", "data", label+".pkl"), "rb") as file:
        model = pkl.load(file)

    fig = plt.figure(figsize=(6.5, 6.5))
    grid = GridSpec(1, 2, top=0.85, bottom=0.1, left=0.07, right=0.97, hspace=0.25, wspace=0.15)

    ax1,g1 = forestplot(model["glm_model"]["trace"].posterior, label_position="title", var_labels=["w"], var_args=plot_args, fig=fig, sp=grid[:,0], one_based=True)
    ax2,g2 = forestplot(model["glm_model"]["trace"].posterior, label_position="title", var_labels=["bias", "sd"], var_args=plot_args, fig=fig, sp=grid[:,1], one_based=True)

    ax1[0].set_xlim([-3.01, 3.01])
    ax1[0].set_ylabel("$\\mathrm{index}~k$")
    ax2[0].set_xlim([0.5, 1.5])
    ax2[1].set_xlim([0.0, 0.5])

    plt.text(x=0.0, y=1.0, va="bottom", ha="right", s=r"$\textbf{A}$", fontsize=14, fontweight='bold', transform=ax1[0].transAxes)
    plt.text(x=0.0, y=1.0, va="bottom", ha="right", s=r"$\textbf{B}$", fontsize=14, fontweight='bold', transform=ax2[0].transAxes)
    plt.text(x=0.0, y=1.0, va="bottom", ha="right", s=r"$\textbf{C}$", fontsize=14, fontweight='bold', transform=ax2[1].transAxes)

    fig.subplots_adjust(hspace=2)
    fig.suptitle(name)
    fig.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "credible_intervals_"+label+".pdf"), pad_inches=0.4)

## Plot full hierarchical model
hierarchical_model_full = ("hierarchical_full", "Marginal parameter distribution of full hierarchical model")

plot_args = {
    "w": {"color": "C1", "markersize": 4, "interquartile_linewidth": 2, "credible_linewidth": 1},
    "sd": {"color": "C1", "label": "$\\textrm{deviation}~\\sigma$", "markersize": 4, "interquartile_linewidth": 2, "credible_linewidth": 1},
    "bias": {"color": "C1", "markersize": 4, "interquartile_linewidth": 2, "credible_linewidth": 1},
    "w_shared": {"color": "C1", "label": "$\\textrm{shared weights}~w^*_{k}$", "markersize": 4, "interquartile_linewidth": 2, "credible_linewidth": 1},
    "bias_shared": {"color": "C1", "label": "$\\textrm{shared bias}~b^*$", "markersize": 4, "interquartile_linewidth": 2, "credible_linewidth": 1},
}

label,name = hierarchical_model_full
with open(os.path.join(os.path.dirname(__file__), "..", "data", label+".pkl"), "rb") as file:
    model = pkl.load(file)

fig = plt.figure(figsize=(6.5, 6.5))
grid = GridSpec(3, 5, width_ratios=[2,1,1,1,1], top=0.85, bottom=0.1, left=0.07, right=0.97, hspace=0.4, wspace=0.25, height_ratios=[8,1,1])

ax0,g0 = forestplot(model["glm_model"]["trace"].posterior, label_position="title", var_labels=["w_shared"], var_args=plot_args, fig=fig, sp=grid[0,0], one_based=True)
ax0[0].set_xlim([-3.01, 3.01])
ax0[0].set_ylabel("$\\mathrm{index}~k$")
for (i,m) in enumerate(model["glm_model"]["trace"].posterior.w_dim_0):
    plot_args["w"]["label"] = "$\\Delta w^{}_{{k}}$".format(i+1)    
    plot_args["bias"]["label"] = "$\\Delta b^{}$".format(i+1)    
    ax1,g1 = forestplot(model["glm_model"]["trace"].posterior.isel(w_dim_0=m), label_position="title", var_labels=["w"], var_args=plot_args, fig=fig, sp=grid[0,m+1], one_based=True)
    ax2,g2 = forestplot(model["glm_model"]["trace"].posterior.isel(bias_dim_0=int(m)), label_position="title", var_labels=["bias"], var_args=plot_args, fig=fig, sp=grid[1,m+1], one_based=True)
    ax1[0].set_xlim([-0.3, 0.3])
    ax1[0].get_yaxis().set_ticklabels([])
    ax2[0].set_xlim([-3.01, 3.01])
    plt.text(x=0.0, y=1.0, va="bottom", ha="right", s="$\\textbf{{{}}}$".format("BCDE"[i]), fontsize=14, fontweight='bold', transform=ax1[0].transAxes)
    plt.text(x=0.0, y=1.0, va="bottom", ha="right", s="$\\textbf{{{}}}$".format("GHIJ"[i]), fontsize=14, fontweight='bold', transform=ax2[0].transAxes)


ax3,g3=forestplot(model["glm_model"]["trace"].posterior, label_position="title", var_labels=["bias_shared"], var_args=plot_args, fig=fig, sp=grid[1,0], one_based=True)
ax4,g4=forestplot(model["glm_model"]["trace"].posterior, label_position="title", var_labels=["sd"], var_args=plot_args, fig=fig, sp=grid[2,0], one_based=True)

ax3[0].set_xlim([-3.01, 3.01])
ax4[0].set_xlim([-0.05, 0.55])

plt.text(x=0.0, y=1.0, va="bottom", ha="right", s=r"$\textbf{A}$", fontsize=14, fontweight='bold', transform=ax0[0].transAxes)
plt.text(x=0.0, y=1.0, va="bottom", ha="right", s=r"$\textbf{F}$", fontsize=14, fontweight='bold', transform=ax3[0].transAxes)
plt.text(x=0.0, y=1.0, va="bottom", ha="right", s=r"$\textbf{K}$", fontsize=14, fontweight='bold', transform=ax4[0].transAxes)


fig.suptitle(name)
fig.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "credible_intervals_"+label+".pdf"), pad_inches=0.4)

# Plot bias-only hierarchical model
hierarchical_model_only_bias = ("hierarchical_only_bias", "Marginal parameter distribution of hierarchical bias model")

plot_args = {
    "w": {"color": "C1", "label": "$\\textrm{weights}~w_{k}, k\in\{1,\dots,11\}$", "markersize": 4, "interquartile_linewidth": 2, "credible_linewidth": 1},
    "sd": {"color": "C1", "label": "$\\textrm{deviation}~\\sigma$", "markersize": 4, "interquartile_linewidth": 2, "credible_linewidth": 1},
    "bias_shared": {"color": "C1", "label": "$\\textrm{shared~bias}~b^*$", "markersize": 4, "interquartile_linewidth": 2, "credible_linewidth": 1},
    "bias": {"color": "C1", "label": "$b$", "markersize": 4, "interquartile_linewidth": 2, "credible_linewidth": 1},
}

label,name = hierarchical_model_only_bias
with open(os.path.join(os.path.dirname(__file__), "..", "data", label+".pkl"), "rb") as file:
    model = pkl.load(file)

fig = plt.figure(figsize=(6.5, 6.5))
grid = GridSpec(6, 2, top=0.85, bottom=0.1, left=0.07, right=0.97, hspace=1, wspace=0.25)

ax0,g0 = forestplot(model["glm_model"]["trace"].posterior, label_position="title", var_labels=["w"], var_args=plot_args, fig=fig, sp=grid[:,0], one_based=True)
ax0[0].set_xlim([-3.01, 3.01])
ax0[0].set_ylabel("$\\mathrm{index}~k$")
ax1,g1 = forestplot(model["glm_model"]["trace"].posterior, label_position="title", var_labels=["bias_shared"], var_args=plot_args, fig=fig, sp=grid[0,1], one_based=True)
ax1[0].set_xlim([-3.01, 3.01])
for (i,m) in enumerate(model["glm_model"]["trace"].posterior.bias_dim_0):
    plot_args["bias"]["label"] = "$\\Delta b^{} = b^{}-b^*$".format(i+1,i+1)    
    ax3,g3 = forestplot(model["glm_model"]["trace"].posterior.isel(bias_dim_0=int(m)), label_position="title", var_labels=["bias"], var_args=plot_args, fig=fig, sp=grid[1+i,1], one_based=True)
    ax3[0].set_xlim([-3.01, 3.01])
    plt.text(x=0.0, y=1.0, va="bottom", ha="right", s="$\\textbf{{{}}}$".format("CDEF"[i]), fontsize=14, fontweight='bold', transform=ax3[0].transAxes)

ax4,g4 = forestplot(model["glm_model"]["trace"].posterior, label_position="title", var_labels=["sd"], var_args=plot_args, fig=fig, sp=grid[5,1], one_based=True)
ax4[0].set_xlim([-0.05, 0.55])


plt.text(x=0.0, y=1.0, va="bottom", ha="right", s=r"$\textbf{A}$", fontsize=14, fontweight='bold', transform=ax0[0].transAxes)
plt.text(x=0.0, y=1.0, va="bottom", ha="right", s=r"$\textbf{B}$", fontsize=14, fontweight='bold', transform=ax1[0].transAxes)
plt.text(x=0.0, y=1.0, va="bottom", ha="right", s=r"$\textbf{G}$", fontsize=14, fontweight='bold', transform=ax4[0].transAxes)



fig.suptitle(name)
fig.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "credible_intervals_"+label+".pdf"), pad_inches=0.4)


plt.show()