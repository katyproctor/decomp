## Plots to check component classification is doing sensible things
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import splotch
import matplotlib as mpl
import math

import properties

def plot_classification(dat, min_ncomp, max_ncomp, models, m_disk, m_bulge, m_ihl, comp_no, 
                        disk, bulge, ihl, allocs_arr, sph, output_folder, fname):

    ns = np.arange(min_ncomp,max_ncomp+1)

    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use('~/code/paper.mplstyle')
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8

    fig, ax = plt.subplots(2,3, figsize = (3.321*3, 2*3.0))
    colors = {'disk':'#2a9d8f', 'IHL':'#ffb703', 'bulge':'#b7094c', np.nan:'grey'}
 
    mtot = dat['Mass'].sum()
    comps = dat['gmm_pred'].unique()
    bulge_mad = round(np.median(abs(m_bulge - bulge['Mass'].sum()))/mtot,3)

    if "IHL" in comps:
        ihl_mad = round(np.median(abs(m_ihl - ihl['Mass'].sum()))/mtot, 3)
    else:
        ihl_mad = np.NaN

    for n in ns:
        model = models[n-min_ncomp]
        jzjc_y = model.means_.T[0]
        ebind_y = model.means_.T[1]
        comp = pd.Series(allocs_arr[n-min_ncomp])
       
        ax[0,0].scatter(np.repeat(n, len(jzjc_y)), jzjc_y, c=comp.map(colors), s = 20)
        ax[0,0].set_xlabel("$\mathrm{n}$", fontsize = 10)
        ax[0,0].set_ylabel(r"${\mathrm{\langle\, j_{z}/j_{circ}\rangle}}$", fontsize = 10)
        ax[0,0].xaxis.get_major_locator().set_params(integer=True)

        ax[0,1].scatter(np.repeat(n, len(ebind_y)), ebind_y, c=comp.map(colors), s = 20)
        ax[0,1].set_xlabel("$\mathrm{n}$", fontsize = 10)
        ax[0,1].set_ylabel(r"${\mathrm{\langle\, e/e_{min}\rangle}}$", fontsize = 10)
        ax[0,1].xaxis.get_major_locator().set_params(integer=True)

        ax[0,2].set_visible(False)

    if sph == True:
        ax[1,0].set_visible(False)

    else:
        disk_mad = round(np.median(abs(m_disk - disk['Mass'].sum()))/mtot, 3)
        ax[1,0].scatter(ns, m_disk, c = '#2a9d8f', s = 18)
        ax[1,0].axhline(m_disk[comp_no-min_ncomp], c = 'grey', lw = 0.9)
        ax[1,0].axhline(m_disk[comp_no-min_ncomp]*1.1, c = 'grey', linestyle = "--", lw = 0.9)
        ax[1,0].axhline(m_disk[comp_no-min_ncomp]*0.9, c = 'grey', linestyle = "--", lw = 0.9)
        ax[1,0].text(0.7, 0.9, r"$\sigma_{\rm{disk}}\,=\,$"+ str(disk_mad),
         transform=ax[1,0].transAxes)
        ax[1,0].xaxis.get_major_locator().set_params(integer=True)
        med_ind = np.where(m_disk == np.median(m_disk))
        ax[1,0].scatter(ns[med_ind], m_disk[med_ind], c= "k")
        ax[1,0].set_xlabel("$\mathrm{n}$", fontsize = 10)
        ax[1,0].set_ylabel("$\mathrm{M_{disk}\, (M_\odot)}$", fontsize = 10)

    # bulge
    ax[1,1].scatter(ns, m_bulge, c = '#b7094c', s = 18)
    ax[1,1].axhline(m_bulge[comp_no-min_ncomp], c = 'grey', lw = 0.9)
    ax[1,1].axhline(m_bulge[comp_no-min_ncomp]*1.1, c = 'grey', linestyle = "--", lw = 0.9)
    ax[1,1].axhline(m_bulge[comp_no-min_ncomp]*0.9, c = 'grey', linestyle = "--", lw = 0.9)
    ax[1,1].text(0.7, 0.9, r"$\sigma_{\rm{bulge}}\,=\,$"+ str(bulge_mad),
    transform=ax[1,1].transAxes)
    ax[1,1].set_xlabel("$\mathrm{n}$", fontsize = 10)
    ax[1,1].set_ylabel("$\mathrm{M_{bulge}\, (M_\odot)}$", fontsize = 10)
    ax[1,1].xaxis.get_major_locator().set_params(integer=True)
    med_ind = np.where(m_bulge == np.median(m_bulge))
    ax[1,1].scatter(ns[med_ind], m_bulge[med_ind], c= "k")

    # ihl
    ax[1,2].scatter(ns, m_ihl, c = '#ffb703', s = 18)
    ax[1,2].axhline(m_ihl[comp_no-min_ncomp], c = 'grey', lw = 0.9)
    ax[1,2].axhline(m_ihl[comp_no-min_ncomp]*1.1, c = 'grey', linestyle = "--", lw = 0.9)
    ax[1,2].axhline(m_ihl[comp_no-min_ncomp]*0.9, c = 'grey', linestyle = "--", lw = 0.9)
    ax[1,2].text(0.7, 0.9, r"$\sigma_{\rm{IHL}}\,=\,$"+ str(ihl_mad),
    transform=ax[1,2].transAxes)
    ax[1,2].set_xlabel("$\mathrm{n}$", fontsize = 10)
    ax[1,2].set_ylabel("$\mathrm{M_{IHL}\, (M_\odot)}$", fontsize = 10)
    ax[1,2].xaxis.get_major_locator().set_params(integer=True)

    med_ind = np.where(m_ihl == np.median(m_ihl))
    ax[1,2].scatter(ns[med_ind], m_ihl[med_ind], c= "k")

    plt.tight_layout()
    plt.savefig(output_folder + fname + "_decomp.png",
    facecolor="white", dpi=300)
    plt.close()


def plot_proj(dat, output_folder, fname):

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2,3, figsize = (3.1*3, 3*2))

    dat['Formation_z'] = 1/(dat['StellarFormationTime']) - 1
    comps = dat['gmm_pred'].unique()
    bulge = dat[dat['gmm_pred'] == "bulge"].copy()

    fbulge = round(bulge['Mass'].sum()/dat['Mass'].sum(), 3)
    fsph = round(2*dat['Mass'][dat['jz/jcirc'] < 0].sum()/dat['Mass'].sum(), 3)
    fbulge = round(bulge['Mass'].sum()/dat['Mass'].sum(), 3)
    _, kco_bulge = properties.calc_kappa_co(bulge, bulge['rad'].max())
    kco_bulge = round(kco_bulge, 3)

    if "IHL" in comps:
        ihl = dat[dat['gmm_pred'] == "IHL"].copy()
        if ihl.shape[0] < 50:
            print("no plots")
            return
        fihl = round(ihl['Mass'].sum()/dat['Mass'].sum(), 3)
        _, kco_ihl = properties.calc_kappa_co(ihl, ihl['rad'].max())
        kco_ihl = round(kco_ihl, 3)
    
    if "disk" in comps:
        disk = dat[dat['gmm_pred'] == "disk"].copy()
        _, kco_disk = properties.calc_kappa_co(disk, disk['rad'].max())
        kco_disk = round(kco_disk, 3)
        fdisk = round(disk['Mass'].sum()/dat['Mass'].sum(), 3)
        
    if len(comps) == 3:
        dat_list = [disk, bulge, ihl]
        fcomp_list = [fdisk, fbulge, fihl]
        title_list = ["disk", "bulge", "IHL"]
        kappa_list=[kco_disk, kco_bulge, kco_ihl]

        # limits if disk component exists
        vlim =  np.abs(np.quantile(disk['vy'], 0.975)) # vel limit
        nmin_list = [2,2,1]
        bin_list = [math.ceil(50/fdisk), math.ceil(50/fbulge), math.ceil(50/fihl)]


    elif set(['bulge', 'disk']).issubset(comps):
        dat_list = [disk, bulge, None]
        fcomp_list = [fdisk, fbulge, None]
        title_list = ["disk", "bulge", None]
        kappa_list=[kco_disk, kco_bulge, None]

        vlim =  np.abs(np.quantile(dat['vy'], 0.95)) # vel limit
        nmin_list = [2, 2,None]
        bin_list = [math.ceil(50/fdisk), math.ceil(50/fbulge), None]

    else:
        dat_list = [None, bulge, ihl]
        fcomp_list = [None, fbulge, fihl]
        title_list = [None, "bulge", "IHL"]
        kappa_list=[None, kco_bulge, kco_ihl]
        
        vlim =  np.abs(np.quantile(dat['vy'], 0.95)) # vel limit
        nmin_list = [None, 2, 1]
        bin_list = [None, math.ceil(50/fbulge), math.ceil(50/fihl)]

    # limits - top panel
    lim = dat['rad'].quantile(0.99) # x/y limit
    cmap = "coolwarm"

    # limits - lower panel
    var_list = ['jz/jcirc', 'ebindrel', 'Formation_z']
    jzjc_bins = np.arange(-1,1,0.025)
    ebindrel_bins = np.arange(0,1,0.025)
    formz_bins = np.arange(0,dat['Formation_z'].quantile(0.96),0.1)
    bin_list_vars = [jzjc_bins, ebindrel_bins, formz_bins]
    label_list = [r"$j_{z}/j_{\rm{circ}}$", r"$e_{\rm{bind,\,rel}}$",
     r"$z_{\rm{form}}$"]

    for i,ax in enumerate(axes.flat):
        # plot component profiles
        if i < 3:
            # skip first panel if no disk comp
            if i == 0 and "disk" not in comps:
                ax.set_visible(False)
             
            elif i ==2 and "IHL" not in comps:
                ax.set_visible(False)

            else:
                tmp = dat_list[i]
                splotch.plots_2d.hist2D(tmp['x'], tmp['z'], c = tmp['vy'],
                cstat = 'median', cmap = cmap,
                                        clim = [-vlim-1e-3,vlim+1e-3],
                                        xlim = (-lim,lim), ylim = (-lim,lim),
                                        bins = bin_list[i],
                                        nmin = nmin_list[i], ax = ax)
                ax.set_aspect("equal")
                ax.set_title(title_list[i])

                # fcomp
                ax.text(0.05, 0.9, r"$f_{\rm{comp}}=$" + str(fcomp_list[i]),
                transform=ax.transAxes)
                # kappa_co
                ax.text(0.05, 0.82, r"$\kappa_{\rm{co}}=$" + str(kappa_list[i]),
                transform=ax.transAxes)

                ax.set_xlim((-lim,lim))
                ax.set_ylim((-lim,lim))
                ax.set_xlabel('x (kpc)')

        else:
            j = i - 3
            if "disk" in comps:
                ax.hist(disk[var_list[j]], bins = bin_list_vars[j], histtype = "step",
                align = 'left', color = "#2a9d8f", label ="disk")
            ax.hist(bulge[var_list[j]], bins = bin_list_vars[j], histtype = "step",
             align = 'left', color = "#b7094c", label = "bulge")
            if "IHL" in comps:
                ax.hist(ihl[var_list[j]], bins = bin_list_vars[j], histtype = "step",
                 align = 'left', color = "#ffb703", label = "IHL")
            
            ax.hist(dat[var_list[j]], bins = bin_list_vars[j], color = "grey",
            alpha =0.5, align = 'left')
            ax.set_xlabel(label_list[j])

            if j == 0: # jz/jc
                ax.text(0.05, 0.9, r"$2\,f(j_z/j_{\rm{circ}}<0)=$" + str(fsph),
             transform=ax.transAxes,  fontsize = 8)

            if j == 2:
                ax.legend(prop = {'size':8})

        if i == 1 or i==2:
            ax.set_yticklabels([])
            ax.set_yticks([])


    axes[0,0].set_ylabel('z (kpc)')
    axes[1,0].set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_folder + fname + "_proj.png", dpi=300)
    plt.close()


def plot_jzjc(dat, output_folder, fname):
    mpl.rcParams.update(mpl.rcParamsDefault) # dark bg messes things up otherwise
    plt.style.use('~/code/paper.mplstyle')
    fig, ax = plt.subplots(1,1, figsize = (3.2, 3))

    jzjc_bins = np.arange(-1,1,0.02)
    ax.hist(dat['jz/jcirc'], bins = jzjc_bins)
    ax.axvline(np.median(dat['jz/jcirc']), color="orange")

    plt.tight_layout()
    plt.savefig(output_folder + fname + "_jzjc.png", dpi=300)
    plt.close()

