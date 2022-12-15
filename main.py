import numpy as np
import h5py
import glob
import os
import pandas as pd
import argparse
import time

import read_particles
import read_groups
import properties
import model

# cgs to useful units
g_to_msol = 5.02785e-34
cm_to_kpc = 3.24078e-22
s_to_Gyr = 1/(86400*365.25*1e9)
cm_to_km = 1e-5

def parse_args():
    """
    Input is list of GroupNumbers for which the central galaxy star particles will be processed and decomposed. Defaults to three groups for testing purposes.
    """
    parser = argparse.ArgumentParser(description='Process Eagle data.')
    parser.add_argument("-gl", "--group_list", help="List of Eagle GroupNumbers to process.", nargs='*', default = [567, 914,1000])
    parser.add_argument("-b", "--base_dir", help="Base directory for the Eagle data. Enter as string.")
    parser.add_argument("-o", "--out_dir", help="Where to store the processed star data. Enter as string.", default = "model_output")
    parser.add_argument("-i", "--job_ind", help="index of job array")
  
    # default list of variables to store for each central - first four are required for modelling purposes
    var_list = ["Coordinates", "Velocity", "ParticleBindingEnergy",
                "Mass", "StellarFormationTime",
                "Metallicity", "ParticleIDs"
                ]
    parser.add_argument("-vl", "--var_list", help="List of variables to store (for stellar particles)", nargs='*', default = var_list)
    args = parser.parse_args()
    
    base_dir = args.base_dir
    out_dir = args.out_dir
    keep_groups = args.group_list
    keep_vars = args.var_list
    job_ind = args.job_ind

    return base_dir, out_dir, keep_groups, keep_vars, job_ind



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


def run_processing(file_names, var_list, group_dat, gpn, base):
    '''
    Extracts relevant group and particle quantities for centrals
    file_names: string, directory where the Eagle particle data is stored
    group_dat: pd.DataFrame, group level data'''

    dat = read_particles.read_stars(file_names, gpn, var_list)
    
    # convert to useful units
    dat = unit_conversion(dat)
    dat = properties.calc_ebindrel(dat)
    
    # merge group data
    dat['GroupNumber'] = gpn
    dat = pd.merge(dat, group_dat, on = "GroupNumber", how='inner')
    return dat


def main():

    # set particle and group directories
    base, output_fpath, keep_groups, var_list, job_ind = parse_args()  
    fpath = base + 'particledata_028_z000p000/'
    gpfpath = base + 'groups_028_z000p000/'
    
    # get group data
    nfiles_gp = read_groups.get_nfiles(gpfpath)
    group_dat = read_groups.read_groups(nfiles_gp, gpfpath)
   
    # list all z=0 files
    file_names = glob.glob(fpath + "*.hdf5")

    # store global properties
    global_list = []

    startTime = time.time()
    print("Processing groups: ", keep_groups) 
    for gpn in keep_groups:
        gpn = int(gpn)
        print("Reading group number:", gpn)
        
        ## Process star partiles in central
        dat = run_processing(file_names, var_list, group_dat, gpn, base)
    
        ## Calculate various properties for modelling
        dat = properties.run(dat)

        ## Decomposition model and component properties calc
        plot_folder = output_fpath + "plots/"
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        dat, disk_mad, bulge_mad, ihl_mad, comp_no  = model.run(dat, plot_folder, gpn)
        summary = properties.calc_comp_properties(dat, disk_mad, bulge_mad, ihl_mad, comp_no, gpn)
        global_list.append(summary)
     
        ## Save output for central
        if not os.path.exists(output_fpath):
            os.makedirs(output_fpath) 
        output_fname = "central_" + str(gpn) + ".pkl"

        vartmp= ["x","y","z","vx","vy","vz","ParticleBindingEnergy",
                "Mass","Metallicity", "ParticleIDs"] 
        extra_vars = ['m200', 'r200', 'jcirc', 'jz/jcirc', 'ebindrel', 'gmm_pred', 'Formation_z']
        save_vars = vartmp + extra_vars
        dat[save_vars].to_pickle(output_fpath+output_fname)

    # save global data
    executionTime = (time.time() - startTime)
    print('Time to run all galaxies: ' + str(round(executionTime/60, 3)) + ' minutes', flush = True)
    global_dat = pd.concat(global_list, ignore_index = True)
    global_dat.to_pickle(output_fpath+"summary"+str(job_ind)+".pkl")

if __name__ == '__main__':
    main()

