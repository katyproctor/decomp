import numpy as np
import h5py
import os
import pandas as pd

g_to_msol = 5.02785e-34
cm_to_kpc = 3.24078e-22

def get_nfiles(gfpath, zstring = "028_z000p000"):
    gf = h5py.File(gfpath + 'eagle_subfind_tab_' + zstring + '.0.hdf5', 'r')
    return gf['FOF'].attrs.get('NTask')


def read_group_mass(nfiles, gpfpath, zstring = "028_z000p000"):
    '''Read M200 for all groups for subsetting'''

    # Output array.
    dat_list = []

    # Loop over each file and extract the data.
    for i in range(nfiles):
        f = h5py.File(gpfpath + 'eagle_subfind_tab_' + zstring + '.%i.hdf5'%i, 'r')
        M200 = f['FOF/Group_M_Crit200'][...]

        # Get conversion factors.
        cgs_mass     = f['FOF/Group_M_Crit200'].attrs.get('CGSConversionFactor')
        aexp_mass    = f['FOF/Group_M_Crit200'].attrs.get('aexp-scale-exponent')

        # hexp only depends on redshift so use this for all
        hexp    = f['FOF/Group_M_Crit200'].attrs.get('h-scale-exponent')

        # Get expansion factor and Hubble parameter from the header.
        a       = f['Header'].attrs.get('Time')
        h       = f['Header'].attrs.get('HubbleParam')

        # Convert to physical units - kpc / Msol
        M200 = np.multiply(M200, g_to_msol * cgs_mass * a**aexp_mass * h**hexp, dtype='f8')
        dat_list.append(pd.DataFrame(M200.T,
                                     columns = ["m200"]))

        f.close()

    data = pd.concat(dat_list, ignore_index=True)
    data['GroupNumber'] = np.arange(data.shape[0]) + 1

    return data


def read_groups(nfiles, gpfpath, zstring = "028_z000p000"):
    '''Read information on the halo for each central identified by FoF at z=0'''

    # Output array.
    dat_list = []

    # Loop over each file and extract the data.
    for i in range(nfiles):
        f = h5py.File(gpfpath + 'eagle_subfind_tab_' + zstring + '.%i.hdf5'%i, 'r')

        # data to read
        cop = f['FOF/GroupCentreOfPotential'][...]

        M200 = f['FOF/Group_M_Crit200'][...]
        R200 = f['FOF/Group_R_Crit200'][...]
        Nsub = np.array(f['FOF/NumOfSubhalos'][...])

        # Get conversion factors.
        cgs_length     = f['FOF/GroupCentreOfPotential'].attrs.get('CGSConversionFactor')
        aexp_length    = f['FOF/GroupCentreOfPotential'].attrs.get('aexp-scale-exponent')

        cgs_mass     = f['FOF/Group_M_Crit200'].attrs.get('CGSConversionFactor')
        aexp_mass    = f['FOF/Group_M_Crit200'].attrs.get('aexp-scale-exponent')

        # hexp only depends on redshift so use this for all
        hexp    = f['FOF/Group_M_Crit200'].attrs.get('h-scale-exponent')

        # Get expansion factor and Hubble parameter from the header.
        a       = f['Header'].attrs.get('Time')
        h       = f['Header'].attrs.get('HubbleParam')
        boxsize = f['Header'].attrs.get('BoxSize')      # L [Mph/h].

        # Convert to physical units - kpc / Msol
        cop = np.multiply(cop, cm_to_kpc * cgs_length * a**aexp_length * h**hexp, dtype='f8')
        M200 = np.multiply(M200, g_to_msol * cgs_mass * a**aexp_mass * h**hexp, dtype='f8')
        R200 = np.multiply(R200, cm_to_kpc * cgs_length * a**aexp_length * h**hexp, dtype='f8')

        tmp = np.vstack([cop.T, M200.T, R200.T, Nsub.T])
        dat_list.append(pd.DataFrame(tmp.T,
                                     columns = ["cop_x", "cop_y", "cop_z",
                                     "m200", "r200", "Nsub"]))

        f.close()

    data = pd.concat(dat_list, ignore_index=True)
    data['GroupNumber'] = np.arange(data.shape[0]) + 1

    return data


def get_subgroups(nfiles, gpfpath, zstring):

    # Output array.
    dat_list = []
 
    # Loop over each file and extract the data.
    for i in range(nfiles):
        f = h5py.File(gpfpath + 'eagle_subfind_tab_' + zstring + '.%i.hdf5'%i, 'r')
        
        dset = f['Subhalo']
        subgpns = np.array(dset['SubGroupNumber'])
        gpn = np.array(dset["GroupNumber"])
        cop = dset['CentreOfPotential']
        mstar = dset['MassType'][:,4]
        mdm = dset['MassType'][:,1]
  
        # Get conversion factors.
        cgs_length     = dset['CentreOfMass'].attrs.get('CGSConversionFactor')
        aexp_length    = dset['CentreOfMass'].attrs.get('aexp-scale-exponent')
        hexp    = dset['CentreOfMass'].attrs.get('h-scale-exponent')

        cgs_mass     = f['Subhalo/Stars']["Mass"].attrs.get('CGSConversionFactor')
        aexp_mass = f['Subhalo/Stars']["Mass"].attrs.get('aexp-scale-exponent')
        hexp_mass = f['Subhalo/Stars']["Mass"].attrs.get('h-scale-exponent')

        # Get expansion factor and Hubble parameter from the header.
        a       = f['Header'].attrs.get('Time')
        h       = f['Header'].attrs.get('HubbleParam')
        boxsize = f['Header'].attrs.get('BoxSize')      # L [Mph/h].

        # Convert to physical units - kpc / Msol
        cop = np.multiply(cop, cm_to_kpc * cgs_length * a**aexp_length * h**hexp, dtype='f8')
        mstar = np.multiply(mstar, g_to_msol * cgs_mass * a**aexp_mass * h**hexp_mass, dtype='f8')
        mdm = np.multiply(mdm, g_to_msol * cgs_mass * a**aexp_mass * h**hexp_mass, dtype='f8')
        tmp = np.vstack([gpn.T, subgpns.T, cop.T, mstar.T, mdm.T])
        dat_list.append(pd.DataFrame(tmp.T,
                                     columns = ["GroupNumber", "SubGroupNumber",
                                                "cop_x", "cop_y", "cop_z", "mstar", "mdm"]))

        f.close()

    data = pd.concat(dat_list, ignore_index=True)

    return data    



def read_com_cop(nfiles, gpfpath, zstring = "028_z000p000"):
    '''Read CoP and CoM for each central identified by FoF at z=0'''
    
    # have confirmed that FoF CoP = Subhalo=0 CoP

    # Output array.
    dat_list = []

    # Loop over each file and extract the data.
    for i in range(nfiles):
        f = h5py.File(gpfpath + 'eagle_subfind_tab_' + zstring + '.%i.hdf5'%i, 'r')
        
        dset = f['Subhalo']
        subgpns = np.array(dset['SubGroupNumber'])
        central_ind = np.where(subgpns == 0)

        # data to read
        gpn = np.array(dset["GroupNumber"][central_ind])
        cop = dset['CentreOfPotential'][central_ind]
        com = dset['CentreOfMass'][central_ind]
        mstar = f['Subhalo/Stars']["Mass"][central_ind]
        rmax = f['Subhalo']["VmaxRadius"][central_ind]

        # Get conversion factors.
        cgs_length     = dset['CentreOfMass'].attrs.get('CGSConversionFactor')
        aexp_length    = dset['CentreOfMass'].attrs.get('aexp-scale-exponent')
        hexp    = dset['CentreOfMass'].attrs.get('h-scale-exponent')

        cgs_mass     = f['Subhalo/Stars']["Mass"].attrs.get('CGSConversionFactor')
        aexp_mass = f['Subhalo/Stars']["Mass"].attrs.get('aexp-scale-exponent')
        hexp_mass = f['Subhalo/Stars']["Mass"].attrs.get('h-scale-exponent')

        # Get expansion factor and Hubble parameter from the header.
        a       = f['Header'].attrs.get('Time')
        h       = f['Header'].attrs.get('HubbleParam')
        boxsize = f['Header'].attrs.get('BoxSize')      # L [Mph/h].

        # Convert to physical units - kpc / Msol
        cop = np.multiply(cop, cm_to_kpc * cgs_length * a**aexp_length * h**hexp, dtype='f8')
        com = np.multiply(com, cm_to_kpc * cgs_length * a**aexp_length * h**hexp, dtype='f8')
        rmax = np.multiply(rmax, cm_to_kpc * cgs_length * a**aexp_length * h**hexp, dtype='f8')
        mstar = np.multiply(mstar, g_to_msol * cgs_mass * a**aexp_mass * h**hexp_mass, dtype='f8')

        tmp = np.vstack([gpn.T, com.T, mstar.T, rmax.T])
        dat_list.append(pd.DataFrame(tmp.T,
                                     columns = ["GroupNumber", 
                                                "com_x", "com_y", "com_z", "mstar", "rmax"]))

        f.close()

    data = pd.concat(dat_list, ignore_index=True)

    return data
