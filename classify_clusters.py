### Rules for assigning GMM clusters to galaxy components
### classify_comps also dictates which model to use as final allocation model
import numpy as np
import pandas as pd


def simultaneous_allocation(clus_col, model):
    '''Assign clusters to disk, bulge or IHL based on separation of:
    jzjc means for disk, ebind means for bulge/IHL'''

    jzjc_means = model.means_.T[0]
    ebind_means = model.means_.T[1]

    # split spheroid into bulge + ihl
    allocs_sph = pd.cut(ebind_means, bins=2, labels=['IHL', 'bulge'],
     include_lowest=True)
    allocs = np.where(jzjc_means < 0.5, allocs_sph, "disk")

    disk_inds = np.where(allocs == 'disk')[0]
    ihl_inds = np.where(allocs == 'IHL')[0]
    bulge_inds = np.where(allocs == 'bulge')[0]

    comps = np.where(clus_col.isin(disk_inds), "disk",
                    np.where(clus_col.isin(bulge_inds), "bulge",
                    np.where(clus_col.isin(ihl_inds), "IHL", "check")))

    return comps, allocs


def sph_allocation(clus_col, model):
    '''Assign clusters to bulge or IHL based on separation of ebind means'''
    ebind_means = model.means_.T[1]

    allocs = pd.cut(ebind_means, bins=2, labels=['IHL', 'bulge'],
    include_lowest=True)
    ihl_inds = np.where(allocs == 'IHL')[0]
    comps = np.where(clus_col.isin(ihl_inds), "IHL", "bulge")

    return comps, allocs


def classify_comps(iter_dat, min_ncomp, m_disk, m_bulge, m_ihl, sph):
    '''Choose which n to use to allocate particles to components'''
    med_disk = np.median(m_disk)
    med_bulge = np.median(m_bulge)
    med_ihl = np.median(m_ihl)
 
    disk_ind = np.where(m_disk == med_disk)[0][0]
    bulge_ind = np.where(m_bulge == med_bulge)[0][0]
    ihl_ind = np.where(m_ihl == med_ihl)[0][0]

    # IHL non-detection criteria
    if m_ihl[1] == 0 or med_ihl ==0:
        print("No IHL")
        ihl_bool = False
        ind = np.max([disk_ind, bulge_ind])

    else:
        ihl_bool = True
        ind = np.max([disk_ind, bulge_ind, ihl_ind])

    # number of components of final alloation model
    comp_no = ind + min_ncomp

    if sph == False:
        disk = iter_dat[iter_dat["comp_" + str(comp_no)] == "disk"].copy()
        disk_mad = np.median(abs(m_disk - disk['Mass'].sum()))
    else:
        disk = None
        disk_mad = np.NaN

    if m_ihl[1] > 0:
        bulge = iter_dat[iter_dat["comp_" + str(comp_no)] == "bulge"].copy()
        ihl = iter_dat[iter_dat["comp_" + str(comp_no)] == "IHL"].copy()
        ihl_mad = np.median(abs(m_ihl - ihl['Mass'].sum()))

    else:
        bulge = iter_dat[(iter_dat["comp_" + str(comp_no)] == "bulge") |
        (iter_dat["comp_" + str(comp_no)] == "IHL")].copy()
        ihl = None
        ihl_mad = np.NaN

    bulge_mad = np.median(abs(m_bulge - bulge['Mass'].sum()))

    return  disk, bulge, ihl, disk_mad, bulge_mad, ihl_mad, ihl_bool
