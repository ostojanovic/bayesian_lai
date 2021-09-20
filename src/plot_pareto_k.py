import arviz
import pickle as pkl
import os
from utils import *

# Identify critical points
(label,name) = ("hierarchical_full","Hier. Full")
with open(os.path.join(os.path.dirname(__file__), "..", "data", label+".pkl"), "rb") as file:
    model = pkl.load(file)
psis_loo = model["glm_model"]["psis-loo"]
critical_points = psis_loo.pareto_k.where(lambda k: k>0.7, drop=True)

# Load spectrograms
data = load_data()
freqs = np.arange(wavelength_start,wavelength_stop+1)
spectrum = data.iloc[critical_points.Y_dim_0].loc[:,wavelength_start:wavelength_stop]
spectrum = spectrum.droplevel(0).reset_index()
spectrum["Measurement"] = "Measurement for Field " + spectrum["group"] + " on " + spectrum["date and time"].apply(str)
spectrum.drop(["group","date and time"], axis="columns", inplace=True)
spectrum.set_index("Measurement", inplace=True)

# Plot
fig,ax = plt.subplots(1,2, gridspec_kw={"width_ratios": [1,2], "wspace": 0.3, "bottom": 0.5}, figsize=(6.5, 4))

colors = np.array(["silver" for i in range(len(data))], dtype=object)
colors[critical_points.Y_dim_0.values] = np.array(plt.rcParams["axes.prop_cycle"].by_key()["color"])[critical_points.argsort().values]
arviz.plot_khat(psis_loo.pareto_k, ax=ax[0], color=colors, marker="o", hlines_kwargs={"color": "dimgray"})
ax[0].set_xlabel("Measurement ID")
ax[0].set_title("Pareto k")


mean1 = data.loc[:,"B",:].mean().loc[wavelength_start:wavelength_stop]
mean1.name = "Mean reflectance spectrum for Field B"
mean2 = data.loc[:,"D",:].mean().loc[wavelength_start:wavelength_stop]
mean2.name = "Mean reflectance spectrum for Field D"
mean1.plot(ax=ax[1], color="black", linestyle="dashed", linewidth=2)
mean2.plot(ax=ax[1], color="black", linestyle="dotted", linewidth=2)
spectrum.T.plot(ax=ax[1], xlabel="Wavelength [nm]", ylabel="Rel. power", title="Reflectance spectra")
ax[1].legend(loc='upper right', bbox_to_anchor=(1.0,-0.3), fancybox=False, frameon=False)

plt.text(x=0.0, y=1.01, va="bottom", ha="right", s=r"$\textbf{A}$", fontsize=14, fontweight='bold', transform=ax[0].transAxes)
plt.text(x=0.0, y=1.01, va="bottom", ha="right", s=r"$\textbf{B}$", fontsize=14, fontweight='bold', transform=ax[1].transAxes)

fig.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "pareto_k.pdf"), pad_inches=0.4)
