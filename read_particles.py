import numpy as np
import h5py
import os
import pandas as pd

# cgs to useful units
g_to_msol = 5.02785e-34
cm_to_kpc = 3.24078e-22
s_to_Gyr = 1/(86400*365.25*1e9)
cm_to_km = 1e-5


def read_stars(fnames, keep_group, var_list, central = True):
    """ Read star data, fnames is list of particle data file names,
     keep_group is the GroupNumber to subset the data to.
     var_list is the list of variables to save
     central: Boolean indicating whether to subset to stars bound to the central galaxy only."""

    # Output array.
    data_list = []

    # Column headers (replace Coordinates with x,y,z)
    expanded_cols = var_list.copy()
    expanded_cols.remove("Coordinates")
    expanded_cols.remove("Velocity")
    new_cols = ["x", "y", "z", "vx", "vy", "vz"]

    for i in range(6):
        expanded_cols.insert(i, new_cols[i])

    # Loop over each file and extract relevant data
    for ff in fnames:
        f = h5py.File(ff, 'r')
        tmp_gn = f['PartType4/GroupNumber'][...]
        tmp_sgn = f['PartType4/SubGroupNumber'][...]
       
        # keep particles in FoF group
        if central == False:
            keep_ind = tmp_gn == keep_group

        else: # keep particles in FoF group and centrals only
            keep_ind = (tmp_gn == keep_group) & (tmp_sgn == 0)

        all_vars = []

        if(any(keep_ind)):

            for tmp_var in var_list:

                if(tmp_var in ["Coordinates", "Velocity"]):
                    tmp = f['PartType4/' + str(tmp_var)][:][keep_ind].T # extract x,y,z comps

                else:
                    tmp = f['PartType4/' + str(tmp_var)][keep_ind]

                # Get conversion factors.
                cgs     = f['PartType4/' + str(tmp_var)].attrs.get('CGSConversionFactor')
                aexp    = f['PartType4/' + str(tmp_var)].attrs.get('aexp-scale-exponent')
                hexp    = f['PartType4/' + str(tmp_var)].attrs.get('h-scale-exponent')

                # Get expansion factor and Hubble parameter from the header.
                a       = f['Header'].attrs.get('Time')
                h       = f['Header'].attrs.get('HubbleParam')
                boxsize = f['Header'].attrs.get('BoxSize')      # L [Mph/h].

                # Convert to physical.
                tmp = np.multiply(tmp, cgs * a**aexp * h**hexp , dtype='f8')

                # Combine with other variables (for 1 file).
                all_vars.append(tmp)
            f.close()

            # Combine data from different files.
            data_list.append(pd.DataFrame(np.vstack(all_vars).T, columns = expanded_cols))

    # combine to Pandas df
    if data_list:
        data = pd.concat(data_list, ignore_index = True)

    else:
        data = pd.DataFrame([])

    return data


def read_stars_subgroups(fnames, keep_group, keep_subgroups, var_list):
    """ Read star data, fnames is list of particle data file names,
     keep_group is the GroupNumber to subset the data to.
     var_list is the list of variables to save
     central: Boolean indicating whether to subset to stars bound to the central galaxy only."""

    # Output array.
    data_list = []

    # Column headers (replace Coordinates with x,y,z)
    expanded_cols = var_list.copy()
    expanded_cols.remove("Coordinates")
    expanded_cols.remove("Velocity")
    new_cols = ["x", "y", "z", "vx", "vy", "vz"]

    for i in range(6):
        expanded_cols.insert(i, new_cols[i])

    for ff in fnames:
        f = h5py.File(ff, 'r')

        # only read data fro relevant group and subgroups
        tmp_gn = f['PartType4/GroupNumber'][...]
        tmp_sgn = f['PartType4/SubGroupNumber'][...]
        keep_ind = (tmp_gn == keep_group) & (np.isin(tmp_sgn, np.array(keep_subgroups)))

        all_vars = []

        if(any(keep_ind)):

            for tmp_var in var_list:

                if(tmp_var in ["Coordinates", "Velocity"]):
                    tmp = f['PartType4/' + str(tmp_var)][:][keep_ind].T # extract x,y,z comps

                else:
                    tmp = f['PartType4/' + str(tmp_var)][keep_ind]

                # Get conversion factors.
                cgs     = f['PartType4/' + str(tmp_var)].attrs.get('CGSConversionFactor')
                aexp    = f['PartType4/' + str(tmp_var)].attrs.get('aexp-scale-exponent')
                hexp    = f['PartType4/' + str(tmp_var)].attrs.get('h-scale-exponent')

                # Get expansion factor and Hubble parameter from the header.
                a       = f['Header'].attrs.get('Time')
                h       = f['Header'].attrs.get('HubbleParam')
                boxsize = f['Header'].attrs.get('BoxSize')      # L [Mph/h].

                # Convert to physical.
                tmp = np.multiply(tmp, cgs * a**aexp * h**hexp , dtype='f8')

                # Combine with other variables (for 1 file).
                all_vars.append(tmp)
            f.close()

            # Combine data from different files.
            data_list.append(pd.DataFrame(np.vstack(all_vars).T, columns = expanded_cols))

    # combine to Pandas df
    if data_list:
        data = pd.concat(data_list, ignore_index = True)

    else:
        data = pd.DataFrame([])

    return data


def read_stars_subset(fnames, subset_var="GroupNumber", subset_ids = [100], var_list= ["Coordinates", "Velocity", "Mass"]):
    """ Read star data, fnames is list of particle data file names,
     subset_var: variable used to subset the particle data on, default to GroupNumber
     ids is list of subset_var to read. e.g. List of GroupNumbers in default case.
     var_list is the list of stellar properties to save. """

    # Output array.
    data_list = []

    # Column headers (replace Coordinates with x,y,z)
    expanded_cols = var_list.copy()
    expanded_cols.remove("Coordinates")
    expanded_cols.remove("Velocity")
    new_cols = ["x", "y", "z", "vx", "vy", "vz"]

    for i in range(6):
        expanded_cols.insert(i, new_cols[i])

    if type(subset_ids) == int:
        subset_ids = [subset_ids]

    # Loop over each file and extract relevant data
    for ff in fnames:
        f = h5py.File(ff, 'r')
        all_vars = []
        tmp_ids = f['PartType4/'+ subset_var][...]

        keep_ind = np.isin(tmp_ids, np.array(subset_ids))

        if(any(keep_ind)):

            for tmp_var in var_list:

                if(tmp_var in ["Coordinates", "Velocity"]):
                    tmp = f['PartType4/' + str(tmp_var)][:][keep_ind].T # extract x,y,z comps

                else:
                    tmp = f['PartType4/' + str(tmp_var)][keep_ind]

                # Get conversion factors.
                cgs     = f['PartType4/' + str(tmp_var)].attrs.get('CGSConversionFactor')
                aexp    = f['PartType4/' + str(tmp_var)].attrs.get('aexp-scale-exponent')
                hexp    = f['PartType4/' + str(tmp_var)].attrs.get('h-scale-exponent')

                # Get expansion factor and Hubble parameter from the header.
                a       = f['Header'].attrs.get('Time')
                h       = f['Header'].attrs.get('HubbleParam')
                boxsize = f['Header'].attrs.get('BoxSize')      # L [Mph/h].

                # Convert to physical.
                tmp = np.multiply(tmp, cgs * a**aexp * h**hexp , dtype='f8')

                # Combine with other variables (for 1 file).
                all_vars.append(tmp)
            f.close()

            # Combine data from different files.
            data_list.append(pd.DataFrame(np.vstack(all_vars).T, columns = expanded_cols))

    # combine to Pandas df
    if data_list:
        data = pd.concat(data_list, ignore_index = True)

    else:
        data = pd.DataFrame([])

    return data




def read_dm(fnames, keep_group):
    """ Read dark matter particle data, fnames is list of particle data file names, keep_group is the GroupNumber to subset the data to. """

    # Output arrays.
    data_coords = []
    data_vels = []
    data_ids = []
    data_energy = []

    data = []

    # Loop over each file and extract relevant data
    for ff in fnames:
        f = h5py.File(ff, 'r')
        tmp_gn = f['PartType1/GroupNumber'][...]
        tmp_sgn = f['PartType1/SubGroupNumber'][...]

        # keep particles in group and centrals only
        keep_ind = (tmp_gn == keep_group) & (tmp_sgn == 0)
        if(any(keep_ind)):
            tmp_coord = f['PartType1/Coordinates'][:][keep_ind]
            tmp_vels = f['PartType1/Velocity'][:][keep_ind]
            tmp_ids = f['PartType1/ParticleIDs'][keep_ind]
            tmp_energy = f['PartType1/ParticleBindingEnergy'][keep_ind]

            data_coords.append(tmp_coord)
            data_vels.append(tmp_vels)
            data_ids.append(tmp_ids)
            data_energy.append(tmp_energy)

            # Get conversion factors.
            cgs     = f['PartType1/Coordinates'].attrs.get('CGSConversionFactor')
            aexp    = f['PartType1/Coordinates'].attrs.get('aexp-scale-exponent')
            hexp    = f['PartType1/Coordinates'].attrs.get('h-scale-exponent')

            cgs_vel     = f['PartType1/Velocity'].attrs.get('CGSConversionFactor')
            aexp_vel    = f['PartType1/Velocity'].attrs.get('aexp-scale-exponent')
            hexp_vel    = f['PartType1/Velocity'].attrs.get('h-scale-exponent')

            cgs_e     = f['PartType1/ParticleBindingEnergy'].attrs.get('CGSConversionFactor')
            aexp_e    = f['PartType1/ParticleBindingEnergy'].attrs.get('aexp-scale-exponent')
            hexp_e    = f['PartType1/ParticleBindingEnergy'].attrs.get('h-scale-exponent')

            # Get expansion factor and Hubble parameter from the header.
            a       = f['Header'].attrs.get('Time')
            h       = f['Header'].attrs.get('HubbleParam')
            boxsize = f['Header'].attrs.get('BoxSize')      # L [Mph/h].

            f.close()

    # Combine to a single array.
    data_coords = np.vstack(data_coords)
    data_vels = np.vstack(data_vels)
    data_ids = np.concatenate(data_ids)
    data_energy = np.concatenate(data_energy)

    # Convert to physical.
    data_coords = np.multiply(data_coords, cgs * a**aexp * h**hexp, dtype='f8')
    data_vels = np.multiply(data_vels, cgs_vel * a**aexp_vel * h**hexp_vel, dtype='f8')
    data_energy = np.multiply(data_energy, cgs_e * a**aexp_e * h**hexp_e, dtype='f8')

    # append and create df
    data.append(pd.DataFrame(data_coords, columns = ['x','y','z']))
    data.append(pd.DataFrame(data_vels, columns = ['vx','vy','vz']))
    data.append(pd.DataFrame(data_ids, columns = ['ParticleIDs']))
    data.append(pd.DataFrame(data_energy, columns = ['ParticleBindingEnergy']))

    # put it all together
    data = pd.concat(data, axis=1)
    data['type'] = "dm"

    return data


def read_dataset_dm_mass(base):
    """ Special case for the mass of dark matter particles. """
    f           = h5py.File(base+'/snapshot_028_z000p000/snap_028_z000p000.0.hdf5', 'r')
    h           = f['Header'].attrs.get('HubbleParam')
    a           = f['Header'].attrs.get('Time')
    dm_mass     = f['Header'].attrs.get('MassTable')[1]

    # Use the conversion factors from the mass entry in the gas particles.
    cgs  = f['PartType0/Mass'].attrs.get('CGSConversionFactor')
    aexp = f['PartType0/Mass'].attrs.get('aexp-scale-exponent')
    hexp = f['PartType0/Mass'].attrs.get('h-scale-exponent')
    f.close()

    # Convert to physical.
    m = dm_mass * cgs * a**aexp * h**hexp

    return m


def unit_conversion(dat):
    '''
    Converts from cgs to useful units.
    [r] : cm -> kpc
    [v] : cm/s -> km/s
    [Mass] : g -> Msol
    '''

    dat['x'], dat['y'], dat['z'] = dat['x']* cm_to_kpc, dat['y']* cm_to_kpc, dat['z']* cm_to_kpc
    dat['vx'], dat['vy'], dat['vz'] = dat['vx']* cm_to_km, dat['vy']* cm_to_km, dat['vz']* cm_to_km
    dat['Mass'] = dat['Mass']*g_to_msol

    return dat
