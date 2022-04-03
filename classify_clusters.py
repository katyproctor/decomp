### Functions to assign GMM clusters to galaxy components
import numpy as np
import pandas as pd

def get_disk_comp(model):
    '''Return index of highest jz/jc cluster provided that it has mean > 0.5'''

    jzjc_mean = model.means_.T[0]
    print(jzjc_mean)

    if(any(jzjc_mean >= 0.5)):
        disk_ind = np.argmax(jzjc_mean)

    else:
        disk_ind = None

    return disk_ind


def blind_ihl_allocation(clus_col, model):
    '''Assign clusters to bulge or IHL based on separation of ebind means'''
    ebind_means = model.means_.T[1] # order changes after disk removal
    # creates two bins of equal width
    allocs = pd.cut(ebind_means, bins=2, labels=['ihl', 'bulge'], include_lowest=True)
    ihl_inds = np.where(allocs == 'ihl')[0]
    comps = np.where(clus_col.isin(ihl_inds), "ihl", "bulge")

    return comps, allocs


def classify_ihl(iter_dat, m_bulge, m_ihl):
    if (m_bulge[1] > 0): # if any bulge comp detected after ncomp=3
        # convergence criterion is 5% variation from median
        median = np.median(m_ihl)
        ind = np.where(m_ihl == median)[0][0]
        comp_no = ind + 2 # accounts for the fact n=2 is minimum no of comps

        bulge = iter_dat[iter_dat["comp_" + str(comp_no)] == "bulge"].copy()
        ihl = iter_dat[iter_dat["comp_" + str(comp_no)] != "bulge"].copy()

    else:
        print("No significant bulge component detected")
        ihl = iter_dat
        bulge = None
        comp_no = 0

    return  bulge, ihl, comp_no
