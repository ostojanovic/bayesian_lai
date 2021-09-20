import pickle as pkl
import numpy as np
import os
from utils import *


# load results from hierarchical model for illustration
label = "hierarchical_full"
print("Loading data for model '{}'".format(label))
with open(os.path.join(os.path.dirname(__file__), "..", "data", label+".pkl"), "rb") as file:
    model = pkl.load(file)

# load preprocessed data
data = load_data()
spectrum = data.loc[:,wavelength_start:wavelength_stop].values
freqs = np.arange(wavelength_start,wavelength_stop+1)
spline_basis = model["spline_model"]["spline_bases"]

# plot one measurement, cumulative derivative & spline basis functions
fig1 = plt.figure(figsize=(6.5,8))
fig1.subplots_adjust(hspace=0.3)
# plt.grid(axis="x",color="grey", alpha=0.5)

# plot spline basis functions
ax4 = plt.subplot(414)
ax4.set_prop_cycle(color=basis_colors)
ax4.tick_params(axis='both')
lines=ax4.plot(freqs, spline_basis)

ax4.set_yticks([0])
ax4.set_xlabel('Wavelength [nm]')
ax4.set_ylim([-0.01,0.25])

peak_idxs = np.argmax(spline_basis, axis=0)
peaks = freqs[peak_idxs]
for (i,(x,idx)) in enumerate(zip(peaks, peak_idxs)):
    y = spline_basis[idx,i]
    ax4.annotate(
        i+1,
        xy=(x,y),
        xytext=(0,5),
        xycoords="data",
        textcoords="offset points",
        ha="center"
    )
ax4.set_title(r"Spline basis functions $b_k(\lambda)$", rotation=0, va="bottom", loc="center", ha="center")
ax4.set_xlabel(r'Wavelength $\lambda$ [nm]')

# plot raw reflectance spectrum
ax1 = plt.subplot(411, sharex=ax4)
ax1.tick_params(axis='both')
ax1.tick_params(axis="x", labelbottom=False)
ax1.tick_params(axis="y")
num_examples = 10
idx = np.random.randint(0,spectrum.shape[0], num_examples)
plt.plot(freqs, spectrum[idx,:].T, alpha=0.5, color="cornflowerblue", linewidth=1)
plt.plot(freqs, spectrum[idx,:].mean(axis=0), color="black", linestyle="dashed", linewidth=2)
plt.title(r"Representative and average reflectance spectra $R_i(\lambda), \bar R(\lambda)$", rotation=0, va="bottom", loc="center", ha="center")
plt.yticks([0])

ax2 = plt.subplot(412, sharex=ax4)
ax2.tick_params(axis='both')
ax2.tick_params(axis="y")
ax2.tick_params(axis="x", labelbottom=False)
plt.plot(freqs, np.diff(model["spline_model"]["cum_abs_curvature"],prepend=[0]), "-", color="cornflowerblue")
plt.title(r"Absolute curvature $q(\lambda)$ of $\bar R(\lambda)$", rotation=0, va="bottom", loc="center", ha="center")
plt.yticks([0])

ax3 = plt.subplot(413, sharex=ax4)
ax3.tick_params(axis='both')
ax3.tick_params(axis="y")
ax3.tick_params(axis='x', labelbottom=False)
plt.plot(freqs, model["spline_model"]["cum_abs_curvature"], color="cornflowerblue", linewidth=2)
for px,py in zip(model["spline_model"]["adaptive_rate_knots"], model["spline_model"]["percentiles"]):
    plt.plot([400,px+400,px+400],[py,py,0], '.-', color="black", alpha=0.4)
plt.yticks([0])
plt.title(r"Cumulative curvature $Q(\lambda)$, knots $\kappa_k$", rotation=0, va="bottom", loc="center", ha="center")

plt.text(x=0.0, y=1.02, va="bottom", ha="right", s=r"$\textbf{A}$", fontsize=14, fontweight='bold', transform=ax1.transAxes)
plt.text(x=0.0, y=1.02, va="bottom", ha="right", s=r"$\textbf{B}$", fontsize=14, fontweight='bold', transform=ax2.transAxes)
plt.text(x=0.0, y=1.02, va="bottom", ha="right", s=r"$\textbf{C}$", fontsize=14, fontweight='bold', transform=ax3.transAxes)
plt.text(x=0.0, y=1.02, va="bottom", ha="right", s=r"$\textbf{D}$", fontsize=14, fontweight='bold', transform=ax4.transAxes)

fig1.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "spline_schema.pdf"), pad_inches=0.4)
# plt.show()
