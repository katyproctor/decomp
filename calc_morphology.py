import numpy as np
import pandas as pd
from scipy import interpolate,linalg

# Code adapted from Thob+2019

def morphological_diagnostics(dat, aperture=30,CoMvelocity=True,reduced_structure=True):
    """
    Compute the morphological diagnostics through the (reduced or not) inertia tensor.
    Returns the morphological diagnostics for the input particles.
    Parameters
    ----------
    ----------
    XYZ : array_like of dtype float, shape (n, 3)
        Particles coordinates (in unit of length L) such that XYZ[:,0] = X,
        XYZ[:,1] = Y & XYZ[:,2] = Z
    mass : array_like of dtype float, shape (n, )
        Particles masses (in unit of mass M)
    Vxyz : array_like of dtype float, shape (n, 3)
        Particles coordinates (in unit of velocity V) such that Vxyz[:,0] = Vx,
        Vxyz[:,1] = Vy & Vxyz[:,2] = Vz
    aperture : float, optional
        Aperture (in unit of length L) for the computation. Default is 0.03 L
    CoMvelocity : bool, optional
        Boolean to allow the centering of velocities by the considered particles
        centre-of-mass velocity. Default to True
    reduced_structure : bool, optional
        Boolean to allow the computation to adopt the iterative reduced form of the
        inertia tensor. Default to True
    Returns
    -------
    ellip : float
        The ellipticity parameter 1-c/a.
    triax : float
        The triaxiality parameter (a^2-b^2)/(a^2-c^2).
    Transform : array of dtype float, shape (3, 3)
        The orthogonal matrix representing the 3 axes as unit vectors: in real-world
        coordinates, Transform[0] = major, Transform[1] = inter, Transform[2] = minor. 
    abc : array of dtype float, shape (3, )
        The corresponding (a,b,c) lengths (in unit of length L).
    """
    XYZ = np.array(dat[['x', 'y', 'z']], dtype=float)
    mass = np.array(dat['Mass'], dtype=float)
    Vxyz = np.array(dat[['vx', 'vy', 'vz']], dtype=float)
 
    particlesall = np.vstack([XYZ.T,mass,Vxyz.T]).T
    # Compute distances
    distancesall = np.linalg.norm(particlesall[:,:3],axis=1)
    # Restrict particles
    extract = (distancesall<aperture)
    particles = particlesall[extract].copy()
    distances = distancesall[extract].copy()
    Mass = np.sum(particles[:,3])
    # Compute kinematic diagnostics
    if CoMvelocity:
        # Compute CoM velocty, correct
        dvVmass = np.nan_to_num(np.sum(particles[:,3][:,np.newaxis]*particles[:,4:7],axis=0)/Mass)
        particlesall[:,4:7]-=dvVmass
        particles[:,4:7]-=dvVmass
    # Compute momentum
    smomentums = np.cross(particlesall[:,:3],particlesall[:,4:7])
    momentum = np.sum(particles[:,3][:,np.newaxis]*smomentums[extract],axis=0)
    # Compute morphological diagnostics
    s = 1; q = 1; Rsphall = 1+reduced_structure*(distancesall-1); stop = False
    while not('structure' in locals()) or (reduced_structure and not(stop)):
        particles = particlesall[extract].copy()
        Rsph = Rsphall[extract]; Rsph/=np.median(Rsph)
        # Compute structure tensor
        structure = np.sum((particles[:,3]/Rsph**2)[:,np.newaxis,np.newaxis]*(np.matmul(particles[:,:3,np.newaxis],particles[:,np.newaxis,:3])),axis=0)/np.sum(particles[:,3]/Rsph**2)
        # Diagonalise structure tensor
        eigval,eigvec = linalg.eigh(structure)
        # Get structure direct oriented orthonormal base
        eigvec[:,2]*=np.round(np.sum(np.cross(eigvec[:,0],eigvec[:,1])*eigvec[:,2]))
        # Return minor axe
        structmainaxe = eigvec[:,np.argmin(eigval)].copy()
        # Permute base and align Y axis with minor axis in momentum direction
        sign = int(np.sign(np.sum(momentum*structmainaxe)+np.finfo(float).tiny))
        structmainaxe *= sign
        temp =  np.array([1,sign,1])*(eigvec[:,((np.argmin(eigval)+np.array([(3+sign)/2,0,(3-sign)/2]))%3).astype(int)])
        eigval = eigval[((np.argmin(eigval)+np.array([(3+sign)/2,0,(3-sign)/2]))%3).astype(int)]
        # Permute base to align Z axis with major axis
        foo = (np.argmax(eigval)/2)*2
        temp = np.array([(-1)**(1+foo/2),1,1])*(temp[:,np.array([2-foo,1,foo]).astype(int)])
        eigval = eigval[np.array([2-foo,1,foo]).astype(int)]
        # Compute change of basis matrix
        transform = linalg.inv(temp)
        stop = (np.max((1-np.sqrt(eigval[:2]/eigval[2])/np.array([q,s]))**2)<1e-4)
        if (reduced_structure and not(stop)):
            q,s = np.sqrt(eigval[:2]/eigval[2])
            Rsphall = linalg.norm(np.matmul(transform,particlesall[:,:3,np.newaxis])[:,:,0]/np.array([q,s,1]),axis=1)
            extract = (Rsphall<aperture/(q*s)**(1/3.))
    Transform = transform.copy()
    ellip = 1-np.sqrt(eigval[1]/eigval[2])
    triax = (1-eigval[0]/eigval[2])/(1-eigval[1]/eigval[2])
    Transform = Transform[...,[2,0,1],:]#so that transform[0] = major, transform[1] = inter, transform[2] = minor
    abc = np.sqrt(eigval[[2,0,1]])
    # Return
    return ellip,triax,Transform,abc


