import pickle as pkl
import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from utils import *
import matplotlib.transforms as transforms


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
naive_dataset_names = ["naive_field_one_2011", "naive_field_one_2012", "naive_field_two_2015", "naive_field_two_2016"]


# leaf pigment content 
# photosynthetic capacity (i.e.  chlorophyll content) 495–680nm 
# 690–720nm is the red edge region 
# 750–800nm; brown pigment content of senescent leaves and canopies 
# 1150–1260nm - canopy water content [34]
# near-infrared, from 750 to 2500nm (we use spectrum till 1350nm); relates to the leaf cell structure 

# 670nm chlorophyll absorption feature
# 1200nm - water absorption 


intervals=[
    # (x1, x2, y, label, marker, kwargs); x1 and x2 in data-coords, y is the "level" in y-direction
    (400, 750, 2, "visible light", "<|-|>", {"ha": "center"}),
    (750, 1350, 2, "near-infrared spectrum", "-|>", {"ha": "center"}),
    (400, 700, 1, "leaf pigment", "<|-|>", {"ha": "center"}),
    (750, 800, 1, "brown pigment", "<|-|>", {"ha": "left"}),
    (495, 680, 0, "photosyn.", "<|-|>", {"ha": "center"}),
    (690, 720, 0, "red edge", "<|-|>", {"ha": "left"}),
    (1150,1260, 0, "canopy water content", "<|-|>", {"ha": "center"}),
]
points = [
    # (x,y,label, marker)
    (670, 0, "", "k|"),
    (1200, 0, "", "k|"),
]

relevant_wavelengths = [670, 750, 1200] #np.unique(np.array([[x1,x2] for (x1,x2,y) in intervals]))

label = "hierarchical_only_bias"

################################################################################
# Plot
trace, wavelengths, spline_basis = load_model(label)
w_matrix = trace.posterior.w.stack(sample=("chain", "draw")).values
all_kernels = np.dot(spline_basis, w_matrix)

fig = plt.figure(figsize=(6.5, 5.0))
gs = GridSpec(2, 1, height_ratios=[2,1], figure=fig, left=0.12, bottom=0.1, right=0.95, top=0.9, wspace=0.25, hspace=0.2)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[1,0], sharex=ax0)
plt.setp(ax0.get_xticklabels(), visible=False)

# main plot
quantile_plot(ax0, wavelengths, all_kernels, cmap="BlueToGray", antialiased=True)

# plot annotations
ax0.set_ylim([-0.1,0.2])
yspace = 0.4
ylevels = np.max([y for (x1,x2,y,l,marker,kwargs) in intervals])+1
for x1,x2,y,l,marker,kwargs in intervals:
    yy = 1-yspace + yspace/ylevels * y
    ax0.annotate(
        "",
        xy=(x1,yy),
        xytext=(x2,yy),
        xycoords=("data","axes fraction"),
        textcoords=("data","axes fraction"),
        arrowprops=dict(arrowstyle=marker, facecolor='black', edgecolor='black'),
        **kwargs
    )
    ax0.annotate(
        l,
        xy=((x1+x2)/2,yy),
        xytext=(0,5),
        xycoords=("data","axes fraction"),
        textcoords="offset points",
        **kwargs
    )

for x,y,l,marker in points:
    yy = 1-yspace + yspace/ylevels * y
    ax0.plot([x],[yy], marker, ms=5, transform=transforms.blended_transform_factory(ax0.transData, ax0.transAxes))
    ax0.annotate(
        l,
        xy=(x,yy),
        xytext=(0,5),
        ha="center",
        xycoords=("data","axes fraction"),
        textcoords="offset points",
    )
 
ax0.plot(wavelengths, all_kernels.mean(axis=1), linewidth=2, color="black")
ax0.axhline(0, color="black", linestyle="dashed")
ax0.set_title("Inferred kernel function (shared between all datasets)")
# ax0.set_ylabel("Contribution")
ax0.xaxis.grid(True)
ax0.yaxis.grid(False)
ax0.set_yticks([-0.1,0,0.1])
ax0.set_rasterized(True)

with open(os.path.join(os.path.dirname(__file__), "..", "data", label+"_feature_importance.pkl"), "rb") as file:
    feature_importance=np.array(pkl.load(file))
    feature_importance /= feature_importance.mean()

importance = np.dot(spline_basis, feature_importance)
importance /= importance.mean()

ax1.set_title("Inferred importance of individual features")
ax1.set_prop_cycle(color=basis_colors)
# ax1.plot(wavelengths, importance, color="gray", linestyle="dashed")
peak_idxs = np.argmax(spline_basis, axis=0)
peaks = wavelengths[peak_idxs]
for (i,(x,y)) in enumerate(zip(peaks, feature_importance)):
    markerline, stemlines, baseline = ax1.stem([x],[y])
    markerline.set_markerfacecolor(basis_colors[i])
    markerline.set_markeredgecolor(basis_colors[i])
    stemlines.set_edgecolor(basis_colors[i])
    ax1.annotate(
        i+1,
        xy=(x,y),
        xytext=(0,5),
        xycoords="data",
        textcoords="offset points",
        ha="center"
    )
ax1.set_xlabel(r'Wavelength $\lambda$ [nm]')
# ax1.set_ylabel("Rel. importance")
ax1.set_ylim([-0.1,2.1])

plt.text(x=0.0, y=1.0, va="bottom", ha="right", s=r"$\textbf{A}$", fontsize=14, fontweight='bold', transform=ax0.transAxes)
plt.text(x=0.0, y=1.0, va="bottom", ha="right", s=r"$\textbf{B}$", fontsize=14, fontweight='bold', transform=ax1.transAxes)

# plt.setp(markerline, "color", [l.get_color() for l in lines])

fig.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "kernel_hierarchical_only_bias.pdf"), pad_inches=0.4, dpi=300)



#plt.show()
