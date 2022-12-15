### Rules for assigning GMM clusters to galaxy components
import numpy as np
import pandas as pd


def simultaneous_allocation(dat, clus_str, model):
    '''Assign clusters to disk, bulge or IHL based on separation of:
    jzjc means for disk, ebind means for bulge/IHL'''

    jzjc_means = model.means_.T[0]
    ebind_means = model.means_.T[1]
    
    # Gaussian allocation
    clus_col = dat[clus_str]
    allocs_disk = np.where(jzjc_means >= 0.5, "disk", "sph")
    allocs_sph = pd.cut(ebind_means, bins=2, labels=['IHL', 'bulge'], include_lowest=True)
    allocs = np.where(jzjc_means < 0.5, allocs_sph, "disk")
    
    disk_inds = np.where(allocs_disk == 'disk')[0]
    ihl_inds = np.where(allocs == 'IHL')[0]
    bulge_inds = np.where(allocs == 'bulge')[0]

    comps = np.where(clus_col.isin(disk_inds), "disk",
                    np.where(clus_col.isin(bulge_inds), "bulge",
                    np.where(clus_col.isin(ihl_inds), "IHL", "check")))

    return comps, allocs


def sph_allocation(dat, clus_str, model):
    '''Assign clusters to bulge or IHL based on separation of ebind means'''

    ebind_means = model.means_.T[1]
    clus_col = dat[clus_str]

    allocs = pd.cut(ebind_means, bins=2, labels=['IHL', 'bulge'],
    include_lowest=True)
    bulge_inds = np.where(allocs == 'bulge')[0]
    ihl_inds = np.where(allocs == 'IHL')[0]
    comps = np.where(clus_col.isin(ihl_inds), "IHL", "bulge")

    return comps, allocs


def select_model(iter_dat, min_ncomp, sph, m_disk, m_bulge, m_ihl):
    '''Select the model to use to allocate particles to components'''
    med_disk = np.median(m_disk)
    med_bulge = np.median(m_bulge)
    med_ihl = np.median(m_ihl)
 
    if sph == False:
        disk_ind = np.where(m_disk == med_disk)[0][0]
    else:
        disk_ind = -1

    if med_ihl == 0:
        ihl_ind = -1
    else:
        ihl_ind = np.where(m_ihl == med_ihl)[0][0]
    
    bulge_ind = np.where(m_bulge == med_bulge)[0][0]

    # number of components of final alloation model
    ind = np.max([disk_ind, bulge_ind, ihl_ind])
    comp_no = ind + min_ncomp

    if sph == False:
        disk = iter_dat[iter_dat["comp_" + str(comp_no)] == "disk"].copy()
        disk_mad = np.median(abs(m_disk - disk['Mass'].sum()))
    else:
        disk = None
        disk_mad = np.NaN

    if med_ihl > 0:
        ihl = iter_dat[iter_dat["comp_" + str(comp_no)] == "IHL"].copy()
        ihl_mad = np.median(abs(m_ihl - ihl['Mass'].sum()))
        bulge = iter_dat[iter_dat["comp_" + str(comp_no)] == "bulge"].copy()
    else:
        ihl = None
        ihl_mad = np.NaN
        # in case final model inclues IHL
        bulge = iter_dat[(iter_dat["comp_" + str(comp_no)] == "bulge") |
        (iter_dat["comp_" + str(comp_no)] == "IHL")].copy()

    bulge_mad = np.median(abs(m_bulge - bulge['Mass'].sum()))

    return  disk, bulge, ihl, disk_mad, bulge_mad, ihl_mad, comp_no 
