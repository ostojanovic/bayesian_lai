import pickle as pkl
import numpy as np
import os
from utils import *
from matplotlib.gridspec import GridSpec


# load preprocessed data
data = load_data()
spectrum = data.loc[:,wavelength_start:wavelength_stop]
freqs = np.arange(wavelength_start,wavelength_stop+1)

num_examples = 5

fig, axes = plt.subplots(figsize=(6.5,6.5), nrows=2, ncols=2, sharex=True, sharey=True, gridspec_kw={"hspace": 0.3})

for i,field in enumerate(["A","B","C","D"]):
    ax = axes.ravel()[i]
    ax.set_title("Field {}".format(field))

    ax.tick_params(axis='both', size=8)
    ax.locator_params(axis='x', nbins=6)

    spectrum_slice = spectrum.loc[:, field, :]
    idx_row = np.random.choice(spectrum_slice.shape[0], num_examples)

    # quantile_plot(ax, freqs, spectrum_slice.values.T, cmap="BlueToGray", antialiased=True)

    ax.fill_between(freqs, spectrum_slice.values.min(axis=0), spectrum_slice.values.max(axis=0), color="black", alpha=0.1)
    ax.plot(freqs, spectrum_slice.iloc[idx_row,:].values.T, color="cornflowerblue", alpha=0.5)
    ax.plot(freqs, spectrum_slice.iloc[idx_row,:].values.mean(axis=0), color="black", linestyle="dashed", linewidth=2)

axes[0,0].set_ylabel("Relative power")
axes[1,0].set_ylabel("Relative power")
axes[1,0].set_xlabel("Wavelength [nm]")
axes[1,1].set_xlabel("Wavelength [nm]")
plt.text(x=0.0, y=1.01, va="bottom", ha="right", s=r"$\textbf{A}$", fontsize=14, fontweight='bold', transform=axes[0,0].transAxes)
plt.text(x=0.0, y=1.01, va="bottom", ha="right", s=r"$\textbf{B}$", fontsize=14, fontweight='bold', transform=axes[0,1].transAxes)
plt.text(x=0.0, y=1.01, va="bottom", ha="right", s=r"$\textbf{C}$", fontsize=14, fontweight='bold', transform=axes[1,0].transAxes)
plt.text(x=0.0, y=1.01, va="bottom", ha="right", s=r"$\textbf{D}$", fontsize=14, fontweight='bold', transform=axes[1,1].transAxes)

fig.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "spectrograms_appendix.pdf"), pad_inches=0.4)
# plt.show()