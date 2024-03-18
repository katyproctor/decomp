### Rules for assigning GMM clusters to galaxy components
import numpy as np
import pandas as pd


def disk_allocation(dat, clus_str, model, probs):
    '''Assign clusters to disk, bulge or IHL based on separation of:
    jzjc means for disk, ebind means for bulge/IHL. 
    Sum cluster probabilities to disk, bulge, IHL.'''

    jzjc_means = model.means_.T[0]
    ebind_means = model.means_.T[1]
    
    # Gaussian allocation
    clus_col = dat[clus_str]
    allocs_disk = np.where(jzjc_means >= 0.5, "disk", "sph")
    allocs_sph = pd.cut(ebind_means, bins=2, labels=['IHL', 'bulge'], include_lowest=True)
    allocs = np.where(jzjc_means < 0.5, allocs_sph, "disk")

    disk_inds = np.where((allocs_disk == 'disk') | (allocs == 'disk'))[0]
    ihl_inds = np.where(allocs == 'IHL')[0]
    bulge_inds = np.where(allocs == 'bulge')[0]

    comps = np.where(clus_col.isin(disk_inds), "disk",
                    np.where(clus_col.isin(bulge_inds), "bulge",
                    np.where(clus_col.isin(ihl_inds), "IHL", "check")))

    disk_probs = np.sum(probs[:,disk_inds], axis=1)
    bulge_probs = np.sum(probs[:,bulge_inds], axis=1)
    ihl_probs = np.sum(probs[:,ihl_inds], axis=1)
    
    # value to split bulge and IHL gaussians
    ecut = (np.max(ebind_means) + np.min(ebind_means))/2
    
    return comps, allocs, ecut


def sph_allocation(dat, clus_str, model, probs):
    '''Assign clusters to bulge or IHL based on separation of ebind means'''

    ebind_means = model.means_.T[1]
    clus_col = dat[clus_str]

    allocs = pd.cut(ebind_means, bins=2, labels=['IHL', 'bulge'],
    include_lowest=True)
    bulge_inds = np.where(allocs == 'bulge')[0]
    ihl_inds = np.where(allocs == 'IHL')[0]
    comps = np.where(clus_col.isin(ihl_inds), "IHL", "bulge")

    bulge_probs = np.sum(probs[:,bulge_inds], axis=1)
    ihl_probs = np.sum(probs[:,ihl_inds], axis=1)

    # value to split bulge and IHL gaussians
    ecut = (np.max(ebind_means) + np.min(ebind_means))/2

    return comps, allocs, ecut

