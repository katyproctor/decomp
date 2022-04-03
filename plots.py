## Plots to check component classification is doing sensible things
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plot_ihl_classification(max_ncomp, models, m_bulge, m_ihl, allocs_arr,
                        output_folder, fname):
    ns = np.arange(2,max_ncomp+1)

    fig, ax = plt.subplots(1,3, figsize = (6.5*3, 5))
    colors = {'ihl':'#ffb703', 'bulge':'#b7094c', np.nan:'grey'}


    for n in ns:
        model = models[n-2]

        ebind_y = model.means_.T[1]
        comp = pd.Series(allocs_arr[n-2])

        ax[0].scatter(np.repeat(n, len(ebind_y)), ebind_y, c =comp.map(colors))
        ax[0].set_xlabel("$\mathrm{n_{clusters}}$", fontsize = 13)
        ax[0].set_ylabel(r"${\mathrm{\langle e/e_{min}\rangle}}$", fontsize = 13)

    # bulge
    ax[1].scatter(ns, m_bulge, c = '#b7094c')
    ax[1].axhline(np.median(m_bulge), c = 'grey')
    ax[1].axhline(np.median(m_bulge)*1.05, c = 'grey', linestyle = "--")
    ax[1].axhline(np.median(m_bulge)*0.95, c = 'grey', linestyle = "--")

    # highlights the value chosen for final classification
    med_ind = np.where(m_bulge == np.median(m_bulge))
    ax[1].scatter(ns[med_ind], m_bulge[med_ind], c= "k")
    ax[1].set_xlabel("$\mathrm{n_{clusters}}$", fontsize = 13)
    ax[1].set_ylabel("$\mathrm{M_{bulge}\, (M_\odot)}$", fontsize = 13)


    # IHL 
    ax[2].scatter(ns, m_ihl, c = '#ffb703')
    ax[2].axhline(np.median(m_ihl), c = 'grey')
    ax[2].axhline(np.median(m_ihl)*1.05, c = 'grey', linestyle = "--")
    ax[2].axhline(np.median(m_ihl)*0.95, c = 'grey', linestyle = "--")
    ax[2].set_xlabel("$\mathrm{n_{clusters}}$", fontsize = 13)
    ax[2].set_ylabel("$\mathrm{M_{IHL}\, (M_\odot)}$", fontsize = 13)

    med_ind = np.where(m_ihl == np.median(m_ihl))
    ax[2].scatter(ns[med_ind], m_ihl[med_ind], c= "k")
    
    plt.tight_layout()
    plt.savefig(output_folder + fname + "_bulge_ihl_decomp.png",
    facecolor="white", dpi=150)
    plt.close()

