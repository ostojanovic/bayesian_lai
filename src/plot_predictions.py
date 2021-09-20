import pickle as pkl
import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from utils import *
import arviz


# create figure and subplots
fig,axes = plt.subplots(3, 2, gridspec_kw={"width_ratios": [2,1], "hspace": 0.4}, figsize=(6.5, 6.5), sharex='col', sharey='col')


# load LAI ground truth values
full_data = load_data()
full_LAI = full_data["LAI"]
m = full_LAI.max()

models = [
    ("naive_pooled", "Baseline model", (slice(None),slice(None))),
    ("hierarchical_only_bias", "Hierarchical bias model", (slice(None),slice(None))), # uses hiearchical model with different bias terms, only
    ("hierarchical_full", "Full hierarchical model", (slice(None),slice(None))), # uses hiearchical model
]

for ((ax1,ax2),(label,name,idx)) in zip(axes, models):
    print("Loading data for model '{}'".format(label))
    with open(os.path.join(os.path.dirname(__file__), "..", "data", label+".pkl"), "rb") as file:
        model = pkl.load(file)

    # sort by ascending ground truth LAI
    LAI_ground_truth_train  = full_LAI.loc[(False,*idx)]
    LAI_ground_truth_test  = full_LAI.loc[(True,*idx)]

    LAI_predicted_train  = model["glm_model"]["pred_train"].mean(axis=0)
    LAI_predicted_test  = model["glm_model"]["pred_test"].mean(axis=0)

    LAI_predicted_quantiles_train  = np.quantile(model["glm_model"]["pred_train"], (0.25,0.75), axis=0)
    LAI_predicted_quantiles_test  = np.quantile(model["glm_model"]["pred_test"], (0.25,0.75), axis=0)

    LAI_residual_test = (LAI_ground_truth_test - LAI_predicted_test) / LAI_ground_truth_test
    LAI_residual_train = (LAI_ground_truth_train - LAI_predicted_train) / LAI_ground_truth_train

    LAI_residual_test_loc, LAI_residual_test_counts = np.unique(LAI_residual_test, return_counts=True)
    LAI_residual_test_cdf = np.cumsum(LAI_residual_test_counts)
    LAI_residual_test_cdf = LAI_residual_test_cdf / LAI_residual_test_cdf[-1]

    LAI_residual_train_loc, LAI_residual_train_counts = np.unique(LAI_residual_train, return_counts=True)
    LAI_residual_train_cdf = np.cumsum(LAI_residual_train_counts)
    LAI_residual_train_cdf = LAI_residual_train_cdf / LAI_residual_train_cdf[-1]

    q1,q2 = (0.25,0.75)
    LAI_quantiles_test = np.quantile(LAI_residual_test, [q1,q2])
    LAI_quantiles_train = np.quantile(LAI_residual_train, [q1,q2])

    ax1.set_title(name)
    ax1.plot([0, m], [0, m], color="black", alpha=0.5)
    ax1.errorbar(x=LAI_ground_truth_train, y=LAI_predicted_train, yerr=np.abs((LAI_predicted_quantiles_train-LAI_predicted_train)), linestyle="", markersize=3, linewidth=1, marker=".", color="C1")
    ax1.errorbar(x=LAI_ground_truth_test, y=LAI_predicted_test, yerr=np.abs((LAI_predicted_quantiles_test-LAI_predicted_test)), linestyle="", markersize=3, linewidth=1, marker=".", color="C5")
    
    ax1.fill_between([0,m], [0, m+m*LAI_quantiles_train[0]], [0, m+m*LAI_quantiles_train[1]], color="gray", alpha=0.25)

    ax2.fill_betweenx([0,1], *LAI_quantiles_train, color="gray", alpha=0.25)
    ax2.plot(LAI_residual_train_loc, LAI_residual_train_cdf, color="C1")
    ax2.plot(LAI_residual_test_loc, LAI_residual_test_cdf, color="C5")
    # arviz.plot_kde(LAI_residual_train, ax=ax2, adaptive=True, plot_kwargs={"color": "C1"})
    # arviz.plot_kde(LAI_residual_test, ax=ax2, adaptive=True, plot_kwargs={"color": "C5", "linestyle": "dashed"})
    ax2.set_xlim([-0.5,0.5])
    ax2.set_ylim([0, 1.05])
    ax2.set_yticks([0, q1, 0.5, q2, 1.0])
    ax2.set_xticks([-0.5, 0, 0.5])
    ax2.set_title("Residual CDF")
    
# set axis decoration
axes[0,0].set_ylabel("predicted LAI")
axes[0,0].tick_params(axis="both")
axes[1,0].set_ylabel("predicted LAI")
axes[1,0].tick_params(axis="both")
axes[2,0].set_ylabel("predicted LAI")
axes[2,0].set_xlabel("measured LAI")
axes[2,0].tick_params(axis="both")
axes[2,1].set_xlabel("relative residual")
plt.text(x=0.0, y=1.01, va="bottom", ha="right", s=r"$\textbf{A}$", fontsize=14, fontweight='bold', transform=axes[0,0].transAxes)
plt.text(x=0.0, y=1.01, va="bottom", ha="right", s=r"$\textbf{B}$", fontsize=14, fontweight='bold', transform=axes[0,1].transAxes)
plt.text(x=0.0, y=1.01, va="bottom", ha="right", s=r"$\textbf{C}$", fontsize=14, fontweight='bold', transform=axes[1,0].transAxes)
plt.text(x=0.0, y=1.01, va="bottom", ha="right", s=r"$\textbf{D}$", fontsize=14, fontweight='bold', transform=axes[1,1].transAxes)
plt.text(x=0.0, y=1.01, va="bottom", ha="right", s=r"$\textbf{E}$", fontsize=14, fontweight='bold', transform=axes[2,0].transAxes)
plt.text(x=0.0, y=1.01, va="bottom", ha="right", s=r"$\textbf{F}$", fontsize=14, fontweight='bold', transform=axes[2,1].transAxes)

 
fig.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "predictions.pdf"), pad_inches=0.4)
# plt.show()
