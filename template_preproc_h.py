# -*- coding: utf-8 -*-
"""
Nipype stroke resting state preprocessing script

Created on Wed Mar  2 08:59:09 2016

Peter Goodin

This script takes the HEALTHY data from the Connect studies and runs 
a resting state analysis cleaning and normalisation regime specialised for stroke
data. 

The only difference between this and the stroke scripts is the removal of FLAIR + mask from
epi coreg + normalisation. 

v3. 
*Made seperate mask calcs for WM and CSF. CSF didn't survive after erosion for
some subs. Upped threshold for csf to 1
*Full preproc script.

v4.
*Removed "manual" SVD of noise vars in favour of using code from sklearn
(validated)

v5.
*Outputs global and noglobal signal files, makes filtered and non-filtered
versions (filtered for connectivity, non-filtered for ALFF / FALFF)
NOTE - ReHo images can be collected from either the warped EPIs or the warped 
non-filtered EPIs.

Readded manual SVD. Quicker, results identical, better control. 

v6.
*Changed order from segment > coregister > make masks to segment > make masks > coregister
Helped remove problems with participants with small ventricles having 0 voxels
for CSF after thresh + ero. Changed thresh to .99 + added 2nd erosion

v7. 

Added 1% STD signal regressor as an option...

v8.
Added FFT filter (code from nipype resting state script)
Changed erosion from FSL to custom function using scipy.ndimage
(faster, more control) 

v9.
Added new erosion algorithm (scipy.ndimage) - faster than FSL erosion + more
control on erosion properties. 

Split WM + CSF into two compcor calls with 5 components each (same as CONN toolbox).
Thresh @ .99 with 2 erosions each (6mm)

Outputs correlation matrices from the AAL atlas for further
analysis.


v9.1.
Refactored code of compcor to make less monolithic.
Added graph lasso regularised AAL outputs.
"""


####Import####
from __future__ import division
from nipype.interfaces import spm, dcmstack, ants, afni
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.algorithms import confounds
from nipype.pipeline.engine import Workflow, Node, MapNode
import os
import time


start = time.time()
#Set up directories and info

raw_dir = 'path/to/raw/data'
write_dir = 'path/to/write/directory'
work_dir = write_dir + 'working/'
crash_dir = write_dir + 'crash/'
out_dir = write_dir + 'output/'

try:
    os.mkdir(out_dir)
except:
    print('Outdir: {} already exists. Not creating new folder'.format(out_dir)) 

try:
    os.mkdir(crash_dir)
except:
    print('Crashdir: {} already exists. Not creating new folder'.format(crash_dir)) 
    

os.chdir(write_dir)
print(os.getcwd())



###SETTINGS###

#Smoothing kernal
fwhm = 6

#subject_list = os.listdir(raw_dir)

subject_list_raw = os.listdir(raw_dir)
subject_list = [x for x in subject_list_raw if 'D_H' in x]
subject_list.sort()

#Select files
template={'anat': raw_dir + '{subject_id}/anat/*.dcm',
          'epi': raw_dir + '{subject_id}/epi/*.dcm',
          'mni_template':'/path/to/warp/templates/template_3mm_brain.nii'}




#Place custom functions here

def metaread(nifti):
    """
    Combines metadata read from the header, populates the SPM slice timing
    correction inputs and outputs the time corrected epi image.
    Uses dcmstack.lookup to get TR and slice times, and NiftiWrapper to get
    image dimensions (number of slices is the z [2]).
    """

    from nipype.interfaces import dcmstack
    from dcmstack.dcmmeta import NiftiWrapper
    nii = NiftiWrapper.from_filename(nifti)
    imdims = nii.meta_ext.shape
    sliceno = imdims[2]
    mid_slice = int(sliceno / 2)
    lookup = dcmstack.LookupMeta()
    lookup.inputs.meta_keys = {'RepetitionTime':'TR','CsaImage.MosaicRefAcqTimes':'ST'}
    lookup.inputs.in_file = nifti
    lookup.run()
    slicetimes = [int(lookup.result['ST'][0][x]) for x in range(0,imdims[2])] #Converts slice times to ints. 
    tr = lookup.result['TR']/1000 #Converts tr to seconds.
    ta = tr -(tr / sliceno)
    return (sliceno, slicetimes, tr, ta, mid_slice)
           
metadata = Node(Function(function = metaread, input_names = ['nifti'], output_names = ['sliceno', 'slicetimes', 'tr', 'ta', 'mid_slice']), name = 'metadata')
#Outputs: tr, slicetimes, imdims

   
def voldrop(epi_list):
    """
    Drops volumes > nvols.
    """
    import numpy as np
    import os
    nvols = 140 #<--------See if there's a way to call a variable outside of a function as input for the function (globals)
    vols = len(epi_list)
    if vols > nvols:
        epi_list = epi_list[0: nvols]
    volsdropped = vols - nvols
    print('Dropped {} volumes'.format(volsdropped))
    volsdropped_fn = os.path.join(os.getcwd(), 'volsdropped.txt')
    np.savetxt(volsdropped_fn, np.atleast_2d(volsdropped))
    
    return (epi_list,volsdropped_fn)        

dropvols = Node(Function(function = voldrop, input_names = ['epi_list'], output_names = ['epi_list','volsdropped_fn']),name='dropvols')
#Outputs: epi_list 

def get_mask_files(seg):
    """
    Makes a list  of outputs from the segmentation of the T1 to be passed to
    masking functions
    """
    wm = seg[1][0]
    csf = seg[2][0]
    return (wm,csf)

mask_list=Node(Function(function=get_mask_files,input_names=['seg'],output_names=['wm','csf']),name='mask_list')

def make_noise_masks(wm_in,csf_in):
    """
    Creates noises masks to be used for compcor. Thresholds and erodes masks by 1 voxel per iteration.
    Note: For stroke participants, the lesion mask is applied to decrease the amount of voxels for erosion.
    This is mostly to do with participants who due to lesioning have enlarged ventricles. These enlarged ventricles
    voxels may throw up an error if too many are entered into the SVD when doing compcor. 
    """
    import numpy as np
    import nibabel as nb
    from scipy.ndimage import binary_erosion as be
    import os
	
    csf_info = nb.load(csf_in)
    wm_info = nb.load(wm_in)

    csf_data = csf_info.get_data() > 0.99
    wm_data = wm_info.get_data() > 0.99

    csf_data[np.isnan(csf_data) == 1] = 0
    wm_data[np.isnan(wm_data) == 1] = 0

    #Erosion structure
    #s=np.ones([2,2,2])   
    s = np.ones([3,3,3])

    csf_erode = be(csf_data, iterations = 2,structure = s)
    wm_erode = be(wm_data,iterations = 2,structure = s)
    
    wm_img = nb.Nifti1Image(wm_erode, header = wm_info.header, affine = wm_info.affine)
    csf_img = nb.Nifti1Image(csf_erode, header = csf_info.header, affine = csf_info.affine)
    wm_mask_fn = os.path.join(os.getcwd(), 'wm_erode.nii')
    csf_mask_fn = os.path.join(os.getcwd(), 'csf_erode.nii')
    wm_img.to_filename(wm_mask_fn)
    csf_img.to_filename(csf_mask_fn)
    
    return(wm_mask_fn, csf_mask_fn)

noisemask = Node(Function(function = make_noise_masks, input_names = ['wm_in','csf_in'], output_names=['wm_mask_fn', 'csf_mask_fn']), name = 'noisemask')
    
def get_seg_files(seg, wmnoise, csfnoise):
    """
    Makes a list (coreg_list) of outputs from the segmentation of the T1 to be passed to
    coregistration from T1 to EPI dims. 
    """
    seg = [x[0] for x in seg] #unlist items
    gm = seg[0]
    wm = seg[1]
    csf = seg[2]
    wmnoise = wmnoise
    csfnoise = csfnoise
    coreg_list = [gm, wm, csf, wmnoise, csfnoise]
    return (coreg_list)

seg_list = Node(Function(function = get_seg_files, input_names = ['seg','wmnoise','csfnoise'], output_names = ['coreg_list']), name = 'seg_list')


def get_anat_2_epi_files(coreg_files):
    """
    Makes a list of outputs from the T1 to EPI coregistration. 
    """
    gm = coreg_files[0]
    wm = coreg_files[1]
    csf = coreg_files[2]
    wmnoise = coreg_files[3]
    csfnoise = coreg_files[4]
    return (gm, wm, csf, wmnoise, csfnoise)

anat2epi_list = Node(Function(function = get_anat_2_epi_files, input_names = ['coreg_files'], output_names = ['gm', 'wm', 'csf', 'wmnoise', 'csfnoise']), name='anat2epi_list')
#Outputs: gm, wm, csf, coregistered source image


def calc_mmask(gm, wm, csf, anat):
    """
    Calculates participant specific brain mask using gm, wm and csf co-registered output.
    """
    import numpy as np
    import nibabel as nb
    import os
    from scipy.ndimage import binary_fill_holes as bfh    
    
    anat = nb.load(anat).get_data()

    gm_mask = nb.load(gm).get_data()
    wm_mask = nb.load(wm).get_data()
    csf_mask = nb.load(csf).get_data()
    m_mask = np.sum([gm_mask, wm_mask, csf_mask], axis = 0)
    m_mask[m_mask > 0.1] = 1
    m_mask = bfh(m_mask) #Fills holes in mask.
    
    #brain_thresh = np.sum([gm_mask, wm_mask], axis = 0) > 0
    #brain_ss = anat * brain_thresh
    brain_ss = gm_mask + wm_mask
    
    img1 = nb.Nifti1Image(m_mask, header = nb.load(gm).header, affine = nb.load(gm).affine)
    mmask_fn = os.path.join(os.getcwd(),'mmask.nii')
    img1.to_filename(mmask_fn)
    img2 = nb.Nifti1Image(brain_ss, header = nb.load(gm).header, affine = nb.load(gm).affine)
    brain_ss_fn = os.path.join(os.getcwd(),'brain_ss.nii')
    img2.to_filename(brain_ss_fn)
    return (mmask_fn, brain_ss_fn)

mmaskcalc = Node(Function(function = calc_mmask, input_names = ['gm', 'wm', 'csf', 'anat'], output_names = ['mmask_fn', 'brain_ss_fn']), name = 'm_mask')
#Outputs: binarised "matter" mask

def make_motion_plot(motion_parameters):
    """
    Plots realignment parameters
    """
    import matplotlib.pylab as plt
    import numpy as np
    import os

    motion_plot_fn = os.path.join(os.getcwd(), 'motion_plot.png')
    motion = np.genfromtxt(motion_parameters)
    deg_labels = ['x','y','z']
    rot_labels = ['pitch', 'yaw', 'roll']
    colour_list = ['r','b','g']

    for n, label in enumerate(deg_labels):
        plt.subplot(211), plt.plot(motion[:, n], colour_list[n], label = label)
    plt.legend()
    plt.xlabel('Volumes')
    plt.ylabel('Motion (degrees)')

    for n, label in enumerate(rot_labels):
        plt.subplot(212), plt.plot(motion[:, n + 3], colour_list[n], label = label)
    plt.legend()
    plt.xlabel('Volumes')
    plt.ylabel('Motion (radians)')
    plt.show()

    return(motion_plot_fn)

motion_plot = Node(Function(function = make_motion_plot, input_names = ['motion_parameters'], output_names = ['motion_plot_fn']), name = 'motion_plot')



def make_motion_regressor(motion_params_fn):
    '''
    Make the motion 24 regressor for use in compcor
    Note: motion_params is the output from realignment.
    '''
    
    import numpy as np
    import os
    
    motion = np.genfromtxt(motion_params_fn)
    
    #CALCULATE FRISTON 24 MODEL (6 motion params + preceeding vol + each values squared.)
    motion_squared = motion ** 2
    new_motion = np.concatenate((motion, motion_squared), axis = 1)
    motion_roll = np.roll(motion, 1, axis = 0)
    motion_roll[0] = 0
    new_motion = np.concatenate((new_motion, motion_roll), axis = 1)
    motion_roll_squared = motion_roll ** 2
    motion24 = np.concatenate((new_motion, motion_roll_squared), axis = 1)
    
    motion24_fn = os.path.join(os.getcwd(), 'motion24_regs.txt')
    np.savetxt(motion24_fn, motion24, delimiter = ',')
    return(motion24_fn)

motion_regressor = Node(Function(function = make_motion_regressor, input_names = ['motion_params_fn'], output_names = ['motion24_fn']), name = 'motion_regressor')
    

def make_wm_regressor(epi_fn, wm_mask_fn):
    '''
    Makes wm noise regressor for use in compcor.
    '''
    import nibabel as nb
    import numpy as np
    import scipy.signal
    import os
    
    epi_data = nb.load(epi_fn).get_data() #Load epi data
    wm_mask = nb.load(wm_mask_fn).get_data().astype('bool') #Load noise mask
    
    wm_data = epi_data[wm_mask].T #Return 2d matrix of wm vox x time
    
    #Remove constant and linear trends from wm
    wm_con = scipy.signal.detrend(wm_data, axis = 0, type = 'constant')
    wm_lin = scipy.signal.detrend(wm_con, axis = 0, type = 'linear')
    
    #Normalise variance
    wm_z = (wm_lin - np.mean(wm_lin, axis = 0)) / np.std(wm_lin, axis = 0)
    
    #Converts nan values to 0
    wm_z[np.isnan(wm_z) == 1] = 0    
    
    #Remove 0 variance time series
    wm_orig = wm_z.shape[1]
    wm_z = wm_z[:,np.std(wm_z,axis = 0) != 0]
    wm_drop = wm_orig - wm_z.shape[1]
    print('Dropped {} wm time series'.format(wm_drop))
    
    wm_drop_fn = os.path.join(os.getcwd(), 'wm_drop.txt')
    np.savetxt(wm_drop_fn, np.atleast_2d(wm_drop), fmt = str('%.5f'), delimiter = ',')
    
    #Compute SVD
    print('Calculating SVD decomposition.')
    [wm_u, wm_s, wm_v] = np.linalg.svd(wm_z)
    wm_var = (wm_s ** 2 / np.sum(wm_s ** 2)) * 100 #Calculate variance explained by individual eigenvectors from s
    wm_cumvar = np.cumsum(wm_s ** 2) / np.sum(wm_s ** 2) *100 #Calculate cumulative variance explained by eigenvectors from s
    
    wm_var_fn = os.path.join(os.getcwd(), 'wm_var_explain.txt')
    wm_cumvar_fn = os.path.join(os.getcwd(), 'wm_cumvar_explain.txt')
    np.savetxt(wm_var_fn, wm_var, fmt = str('%.5f'), delimiter = ',')
    np.savetxt(wm_cumvar_fn, wm_cumvar, fmt = str('%.5f'), delimiter = ',')
    print('File written to {}'.format(wm_var_fn))
    print('File written to {}'.format(wm_cumvar_fn))
    
    #Get components of interest
    nComp = 5 #Number of components
    wm_comps = wm_u[:,:nComp]
    
    #Save wm regressor
    wm_comps_fn = os.path.join(os.getcwd(), 'wm_regs.txt')
    np.savetxt(wm_comps_fn, wm_comps, delimiter = ',')
    
    return(wm_drop_fn, wm_var_fn, wm_cumvar_fn, wm_comps_fn)

wm_regressor = Node(Function(function = make_wm_regressor, input_names = ['epi_fn', 'wm_mask_fn'], output_names = ['wm_drop_fn', 'wm_var_fn', 'wm_cumvar_fn', 'wm_comps_fn']), name = 'wm_regressor')
    
    
    
def make_csf_regressor(epi_fn, csf_mask_fn):
    '''
    Makes csf noise regressor for use in compcor.
    '''
    import nibabel as nb
    import numpy as np
    import scipy.signal
    import os
    
    epi_data = nb.load(epi_fn).get_data() #Load epi data
    csf_mask = nb.load(csf_mask_fn).get_data().astype('bool') #Load noise mask
    
    csf_data = epi_data[csf_mask].T #Return 2d matrix of csf vox x time
    
    #Remove constant and linear trends from csf
    csf_con = scipy.signal.detrend(csf_data, axis = 0, type = 'constant')
    csf_lin = scipy.signal.detrend(csf_con, axis = 0, type = 'linear')
    
    #Normalise variance
    csf_z = (csf_lin - np.mean(csf_lin, axis = 0)) / np.std(csf_lin, axis = 0)
    
    #Converts nan values to 0
    csf_z[np.isnan(csf_z) == 1] = 0    
    
    #Remove 0 variance time series
    csf_orig = csf_z.shape[1]
    csf_z = csf_z[:,np.std(csf_z,axis = 0) != 0]
    csf_drop = csf_orig - csf_z.shape[1]
    print('Dropped {} csf time series'.format(csf_drop))
    
    csf_drop_fn = os.path.join(os.getcwd(), 'csf_drop.txt')
    np.savetxt(csf_drop_fn, np.atleast_2d(csf_drop), fmt = str('%.5f'), delimiter = ',')
    
    #Compute SVD
    print('Calculating SVD decomposition.')
    [csf_u, csf_s, csf_v] = np.linalg.svd(csf_z)
    csf_var = (csf_s ** 2 / np.sum(csf_s ** 2)) * 100 #Calculate variance explained by individual eigenvectors from s
    csf_cumvar = np.cumsum(csf_s ** 2) / np.sum(csf_s ** 2) *100 #Calculate cumulative variance explained by eigenvectors from s
    
    csf_var_fn = os.path.join(os.getcwd(), 'csf_var_explain.txt')
    csf_cumvar_fn = os.path.join(os.getcwd(), 'csf_cumvar_explain.txt')
    np.savetxt(csf_var_fn, csf_var, delimiter = ',')
    np.savetxt(csf_cumvar_fn, csf_cumvar, delimiter = ',')
    print('File written to {}'.format(csf_var_fn))
    print('File written to {}'.format(csf_cumvar_fn))
    
    #Get components of interest
    nComp = 5 #Number of components
    csf_comps = csf_u[:,:nComp]
    
    #Save csf regressor
    csf_comps_fn = os.path.join(os.getcwd(), 'csf_regs.txt')
    np.savetxt(csf_comps_fn, csf_comps, delimiter = ',')
    
    return(csf_drop_fn, csf_var_fn, csf_cumvar_fn, csf_comps_fn)

csf_regressor = Node(Function(function = make_csf_regressor, input_names = ['epi_fn', 'csf_mask_fn'], output_names = ['csf_drop_fn', 'csf_var_fn', 'csf_cumvar_fn', 'csf_comps_fn']), name = 'csf_regressor')
 
    

def run_compcor(epi_fn, global_mask_fn, wm_noise_fn, csf_noise_fn, motion24_fn):
    """
    Regresses out noise time series using the aCompCor method (Behzadi et al. (2007)
    Seperate WM and CSF components (5 a piece, similar to CONN) are used as the noise signal,
    24 motion parameters to "correct" for motion. Scrubbing is not done (see Muschelli et al, 2014).
    Global signal is the mean signal of a whole brain (GM, WM, CSF) mask.
    Output is residuals with global signal NOT removed and global removed.  
    """
    
    import nibabel as nb
    import numpy as np
    import scipy.signal    
    import os


    #Load epi
    epi_info = nb.load(epi_fn)
    epi_data = epi_info.get_data()
    
    #Load global mask
    global_mask = nb.load(global_mask_fn).get_data().astype(bool)
    
    #Load confounds
    motion24 = np.genfromtxt(motion24_fn, delimiter = ',')
    wm_noise = np.genfromtxt(wm_noise_fn, delimiter = ',')
    csf_noise = np.genfromtxt(csf_noise_fn, delimiter = ',')

         
    X1 = []
    B1 = []
    Y1 = []
    Y_resid1 = []      
    
    X1 = np.column_stack((wm_noise, csf_noise, motion24)) #Build regressors file
    
    X1n = (X1 - np.mean(X1, axis = 0)) / np.std(X1, axis = 0) #Normalise regressors
    
    Y1 = epi_data[global_mask].T
    Y1 = scipy.signal.detrend(Y1, axis = 0, type = 'linear') #Remove linear trend of TS
    
    B1 = np.linalg.lstsq(X1n, Y1)[0]
    Y_resid1 = Y1 - X1n.dot(B1) #Regresses nuisance from data
    
    global_regs_fn = os.path.join(os.getcwd(), 'global_regressors.txt')
    np.savetxt(global_regs_fn, X1, fmt = str('%.5f'), delimiter=',')
    print('File written to {}'.format(global_regs_fn))
    
    epi_data[global_mask] = Y_resid1.T
    epi_data = epi_data * (np.repeat(global_mask[:,:,:,np.newaxis],repeats = np.shape(epi_data)[3], axis = 3)) #Creates mask with dims = epi (including time), zeros everything outside brain mask.
    global_img = nb.Nifti1Image(epi_data, header = epi_info.header, affine = epi_info.affine)
    global_img_fn = os.path.join(os.getcwd(),'residuals_global.nii')
    global_img.to_filename(global_img_fn) 
    
    #REMOVING GLOBAL
    
    epi_data = epi_info.get_data() #Reload data
    global_ts = epi_data[global_mask].T
    
    
    X2 = []
    B2 = []
    Y2 = []
    Y_resid2 = []      
    
    X2 = np.column_stack((wm_noise, csf_noise, motion24, np.mean(global_ts,axis=1))) #Build regressors file
    
    X2n = (X2 - np.mean(X2, axis = 0)) / np.std(X2, axis = 0) #Voxel wise variance normalise
    
    Y2 = epi_data[global_mask].T
    Y2 = scipy.signal.detrend(Y1, axis = 0, type = 'linear') #Remove linear trend
    
    B2 = np.linalg.lstsq(X2n,Y2)[0]
    Y_resid2 = Y2-X2n.dot(B2) #Regresses nuisance from data
    
    noglobal_regs_fn = os.path.join(os.getcwd(), 'noglobal_regressors.txt')
    np.savetxt(noglobal_regs_fn, X2, fmt = str('%.5f'),delimiter=',')
    print('File written to {}'.format(noglobal_regs_fn))
    
    epi_data[global_mask] = Y_resid2.T
    epi_data = epi_data * (np.repeat(global_mask[:,:,:,np.newaxis], repeats = np.shape(epi_data)[3],axis=3)) #Creates mask with dims = epi (including time), zeros everything outside brain mask.
    noglobal_img = nb.Nifti1Image(epi_data, header = epi_info.header, affine = epi_info.affine)
    noglobal_img_fn = os.path.join(os.getcwd(), 'residuals_noglobal.nii')
    noglobal_img.to_filename(noglobal_img_fn)

    return(global_regs_fn, global_img_fn, noglobal_regs_fn, noglobal_img_fn)
    
compcor_clean = Node(Function(function = run_compcor, input_names = ['epi_fn', 'global_mask_fn', 'wm_noise_fn', 'csf_noise_fn', 'motion24_fn'], output_names = ['global_regs_fn', 'global_img_fn', 'noglobal_regs_fn', 'noglobal_img_fn']), name = 'compcor_clean')



def get_clean_files(global_img_fn, noglobal_img_fn):

    """
    Makes a list of the output files from compcor to pass to mapnodes..
    """
    clean_global = global_img_fn
    clean_noglobal= noglobal_img_fn
    cleaned = [clean_global, clean_noglobal]
    return(cleaned)

clean_list = Node(Function(function = get_clean_files, input_names = ['global_img_fn', 'noglobal_img_fn'], output_names = ['cleaned']), name = 'clean_list')

def bandpass_filter(epi_fn, global_mask_fn):
    """Bandpass filter the input files
    
    From code in CPAC (https://fcp-indi.github.io/)
    

    Parameters
    ----------
    files: list of 4d nifti files
    lowpass_freq: cutoff frequency for the low pass filter (in Hz)
    highpass_freq: cutoff frequency for the high pass filter (in Hz)
    fs: sampling rate (in Hz)
    """
    import nibabel as nb
    import numpy as np
    import os
    
    lowpass_freq = 0.08
    highpass_freq = 0.01
    fs = 1 / 3.0

    epi_info = nb.load(epi_fn)
    epi_data = epi_info.get_data()
    global_mask = nb.load(global_mask_fn).get_data().astype(bool)
   
    
    timepoints = epi_info.shape[-1]
    F = np.zeros((timepoints))
    lowidx = timepoints / 2 + 1
    if lowpass_freq > 0:
        lowidx = np.round(float(lowpass_freq) / fs * timepoints)
    highidx = 0
    if highpass_freq > 0:
        highidx = np.round(float(highpass_freq) / fs * timepoints)
    F[highidx:lowidx] = 1
    F = ((F + F[::-1]) > 0).astype(int)
    filter_data = epi_data[global_mask].T
    
    if np.all(F == 1):
        epi_data[global_mask] = filter_data.T
    else:
        filter_data = np.real(np.fft.ifftn(np.fft.fftn(filter_data) * F[:,np.newaxis]))
        epi_data[global_mask] = filter_data.T
    
    
    filter_img = nb.Nifti1Image(epi_data, epi_info.affine, epi_info.header)                         
    filter_img_fn = os.path.join(os.getcwd(), 'bp_'+ epi_fn.split('/')[-1])
    filter_img.to_filename(filter_img_fn)
  
    return (filter_img_fn)
    
bpfilter = MapNode(Function(function = bandpass_filter, input_names = ['epi_fn','global_mask_fn'], output_names = ['filter_img_fn']), iterfield = 'epi_fn',name = 'bpfilter')


def get_ants_files(ants_output):

    """
    Gets output from ANTs to pass to normalising all the things. 
    """
    trans = [ants_output[0], ants_output[1]]
    return(trans)

ants_list = Node(Function(function = get_ants_files, input_names = ['ants_output'], output_names = ['trans']), name = 'ants_list')
#Outputs: transformation matrix, inverse image


def smoothing_files(list1,list2,list3):

    """
    Makes a list of the filtered, non-filtered, global, no-global and non-cleaned files
    for smoothing
    """
    smoothing_files=list1+list2+[list3]
    return (smoothing_files)

smooth_list=Node(Function(function=smoothing_files,input_names = ['list1','list2','list3'], output_names = ['smoothing_files']),name='smooth_list')

def mmask_files(mmask_fn, brain_ss_fn):

    """
    Makes a list of the filtered, non-filtered, global, no-global and non-cleaned files
    for smoothing
    """
    mmask_files = [mmask_fn, brain_ss_fn]
    return(mmask_files)

mmask_list=Node(Function(function=mmask_files,input_names=['mmask_fn','brain_ss_fn'], output_names=['mmask_files']), name='mmask_list')


def make_aal_corrmat(smoothed_files):
    """
    Reads in a merged version of the AAL atlas and
    calculates the correlation matrix of all regions.
    Outputs both transformed and non-transformed versions. 
    """
    import nibabel as nb
    import numpy as np
    import sklearn.covariance
    import os
    aalatlas = nb.load('/home/peter/Desktop/test/templates/aal_pa_3mm.nii').get_data()
    
    glob_data = nb.load([s for s in smoothed_files if "sbp_residuals_global_trans.nii" in s][0]).get_data()
    noglob_data = nb.load([s for s in smoothed_files if "sbp_residuals_noglobal_trans.nii" in s][0]).get_data()
    noclean_data = nb.load([s for s in smoothed_files if "sraepi_despike_trans.nii" in s][0]).get_data()
    
    #Pre-allocate regional ts matrix
    aalatlas_ts_glob=np.zeros([glob_data.shape[3],len(np.unique(aalatlas))-1])
    aalatlas_ts_noglob=np.zeros([noglob_data.shape[3],len(np.unique(aalatlas))-1])
    aalatlas_ts_noclean=np.zeros([noclean_data.shape[3],len(np.unique(aalatlas))-1])
    
    #Loop through unique values (skipping background, 0), populate with mean regional ts.
    for x in range(1,len(np.unique(aalatlas))):
        roi=np.squeeze(aalatlas==x)
        aalatlas_ts_glob[:,x-1]=np.mean(glob_data[roi].T,axis=1)
        aalatlas_ts_noglob[:,x-1]=np.mean(noglob_data[roi].T,axis=1)
        aalatlas_ts_noclean[:,x-1]=np.mean(noclean_data[roi].T,axis=1)
        
    #Produce connectivity matrices     
    glob_corrmat = np.corrcoef(aalatlas_ts_glob.T)
    glob_lasso = sklearn.covariance.GraphLassoCV(max_iter = 1000).fit(aalatlas_ts_glob)

    noglob_corrmat = np.corrcoef(aalatlas_ts_noglob.T)
    noglob_lasso = sklearn.covariance.GraphLassoCV(max_iter = 1000).fit(aalatlas_ts_noglob)    

    noclean_corrmat = np.corrcoef(aalatlas_ts_noclean.T)
    noclean_lasso = sklearn.covariance.GraphLassoCV(max_iter = 1000).fit(aalatlas_ts_noclean)    

    #Save data as csv files.
    #Save global
    global_corr_fn = os.path.join(os.getcwd(), 'global_correlation_aal.csv')
    np.savetxt(global_corr_fn, glob_corrmat, fmt = str('%.5f'), delimiter = ',')

    global_lasso_fn = os.path.join(os.getcwd(),'global_lasso_aal.csv')
    np.savetxt(global_lasso_fn, glob_lasso.precision_, fmt = str('%.5f'), delimiter = ',')

    global_corr_trans_fn = os.path.join(os.getcwd(), 'global_correlation_aal_trans.csv')
    np.savetxt(global_corr_trans_fn, np.arctanh(glob_corrmat), fmt = str('%.5f'), delimiter=',')
    
    #Save no global
    noglobal_corr_fn = os.path.join(os.getcwd(), 'noglobal_correlation_aal.csv')
    np.savetxt(noglobal_corr_fn, noglob_corrmat, fmt = str('%.5f'), delimiter = ',')

    noglobal_lasso_fn = os.path.join(os.getcwd(),'noglobal_lasso_aal.csv')
    np.savetxt(noglobal_lasso_fn, noglob_lasso.precision_, fmt = str('%.5f'), delimiter = ',')

    noglobal_corr_trans_fn = os.path.join(os.getcwd(), 'noglobal_correlation_aal_trans.csv')
    np.savetxt(noglobal_corr_trans_fn, np.arctanh(noglob_corrmat), fmt = str('%.5f'), delimiter=',')
    
    #Save no clean - Baseline
    noclean_corr_fn = os.path.join(os.getcwd(),'noclean_correlation_aal.csv')
    np.savetxt(noclean_corr_fn, noclean_corrmat, fmt = str('%.5f'), delimiter=',')

    noclean_lasso_fn = os.path.join(os.getcwd(),'noclean_lasso_aal.csv')
    np.savetxt(noclean_lasso_fn, noclean_lasso.precision_, fmt = str('%.5f'), delimiter = ',')

    noclean_corr_trans_fn = os.path.join(os.getcwd(), 'noclean_correlation_aal_trans.csv')
    np.savetxt(noclean_corr_trans_fn, np.arctanh(noclean_corrmat), fmt = str('%.5f'), delimiter = ',')   

    return(global_corr_fn, global_lasso_fn, global_corr_trans_fn, noglobal_corr_fn, noglobal_lasso_fn, noglobal_corr_trans_fn, noclean_corr_fn, noclean_lasso_fn, noclean_corr_trans_fn)

aal_corrmat = Node(Function(function = make_aal_corrmat,input_names = ['smoothed_files'], output_names = ['global_corr_fn', 'global_lasso_fn', 'global_corr_trans_fn', 'noglobal_corr_fn', 'noglobal_lasso_fn', 'noglobal_corr_trans_fn', 'noclean_corr_fn', 'noclean_lasso_fn', 'noclean_corr_trans_fn']), name = 'aal_corrmat')


####Nipype script begins below####


#Set up iteration over subjects
infosource = Node(IdentityInterface(fields=['subject_id']),name='infosource')
infosource.iterables = ('subject_id',subject_list)

#Select files

selectfiles = Node(SelectFiles(template),name = 'selectfiles')
selectfiles.inputs.base_directory = raw_dir
selectfiles.inputs.sort_files = True
#Outputs: anat, epi, flair, mask,mni_template

####EPI preprocessing####

#Convert EPI dicoms to nii (with embeded metadata)
epi_stack=Node(dcmstack.DcmStack(),name='epistack')
epi_stack.inputs.embed_meta=True
epi_stack.inputs.out_format='epi'
epi_stack.inputs.out_ext='.nii'
#Outputs: out_file

#Despiking using afni (position based on Jo et al. (2013)).
despike=Node(afni.Despike(),name='despike')
despike.inputs.outputtype='NIFTI'
#Outputs: out_file

#Slice timing corrected (gets timing from header)
st_corr=Node(spm.SliceTiming(),name='slicetiming_correction')
#Outputs: timecorrected_files

#Realignment using SPM <--- Maybe just estimate and apply all transforms at the end?
realign=Node(spm.Realign(),name='realign')
realign.inputs.register_to_mean=False
realign.inputs.quality=1.0
#Outputs: realignment_parameters, reliced epi images (motion corrected)

tsnr=Node(confounds.TSNR(),name = 'tsnr')
tsnr.inputs.regress_poly = 2 #Note: This removes linear + constant drifts from the epi time series. Compcor removes from the wm / csf
tsnr.inputs.mean_file = 'mean.nii'
tsnr.inputs.stddev_file = 'stdev.nii'
tsnr.inputs.tsnr_file = 'tsnr.nii'

#Outputs: detrended_file, mean_file, stddev_file, tsnr_file

smooth=Node(spm.Smooth(),name='smooth')
smooth.inputs.fwhm=fwhm


####Anatomical preprocessing####

#dcmstack - Convert dicoms to nii (with embeded metadata)
anat_stack=Node(dcmstack.DcmStack(),name='anatstack')
anat_stack.inputs.embed_meta=True
anat_stack.inputs.out_format='anat'
anat_stack.inputs.out_ext='.nii'
#Outputs: out_file


#Coregisters T1, FLAIR + mask to EPI (NOTE: settings taken from Clinical Toolbox)
coreg2epi = Node(spm.Coregister(), name = 'coreg2epi')
coreg2epi.inputs.cost_function='nmi'
coreg2epi.inputs.separation=[4,2]
coreg2epi.inputs.tolerance=[0.02,0.02,0.02,0.001,0.001,0.001,0.01,0.01,0.01,0.001,0.001,0.001]
coreg2epi.inputs.fwhm = [7,7]
coreg2epi.inputs.write_interp=1
coreg2epi.inputs.write_wrap=[0,0,0]
coreg2epi.inputs.write_mask=False
#Output: coregistered_files

#Segment anatomical
seg = Node(spm.NewSegment(), name = 'segment')
#Outputs: 


#Warps to MNI space using a 3mm template image
#Note - The template is warped to subj space then the inverse transform (subj space > MNI) is used
#to warp the data.
antsnorm=Node(ants.Registration(),name = 'antsnorm')
antsnorm.inputs.output_transform_prefix = 'new'
antsnorm.inputs.collapse_output_transforms = True
antsnorm.inputs.initial_moving_transform_com = True
antsnorm.inputs.num_threads = 1
antsnorm.inputs.output_inverse_warped_image = True
antsnorm.inputs.output_warped_image = True
antsnorm.inputs.sigma_units = ['vox'] * 3
antsnorm.inputs.transforms = ['Rigid', 'Affine', 'SyN']
antsnorm.inputs.terminal_output = 'file'
antsnorm.inputs.winsorize_lower_quantile = 0.005
antsnorm.inputs.winsorize_upper_quantile = 0.995
antsnorm.inputs.convergence_threshold = [1e-06]
antsnorm.inputs.convergence_window_size = [10]
antsnorm.inputs.metric = ['MI', 'MI', 'CC']
antsnorm.inputs.metric_weight = [1.0] * 3
antsnorm.inputs.number_of_iterations = [[1000, 500, 250, 100], [1000, 500, 250, 100], [100, 70, 50, 20]]
antsnorm.inputs.radius_or_number_of_bins = [32, 32, 4]
antsnorm.inputs.sampling_percentage = [0.25, 0.25, 1]
antsnorm.inputs.sampling_strategy = ['Regular','Regular','None']
antsnorm.inputs.shrink_factors = [[8, 4, 2, 1]] * 3
antsnorm.inputs.smoothing_sigmas = [[3, 2, 1, 0]] * 3
antsnorm.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3.0, 0.0)]
antsnorm.inputs.use_histogram_matching = True
antsnorm.inputs.write_composite_transform = False

#Normalise anatomical
apply2anat=Node(ants.ApplyTransforms(),name='apply2anat')
apply2anat.inputs.default_value=0
apply2anat.inputs.input_image_type=0
apply2anat.inputs.interpolation='Linear'
apply2anat.inputs.invert_transform_flags=[True,False]
apply2anat.inputs.num_threads=1
apply2anat.inputs.terminal_output='file'

#Normalise EPI
apply2epi=MapNode(ants.ApplyTransforms(),iterfield='input_image',name='apply2epi')
apply2epi.inputs.default_value=0
apply2epi.inputs.input_image_type=3
apply2epi.inputs.interpolation='Linear'
apply2epi.inputs.invert_transform_flags=[True,False]
apply2epi.inputs.num_threads=1
apply2epi.inputs.terminal_output='file'

#Normalise matter mask
apply2mmask=MapNode(ants.ApplyTransforms(),iterfield='input_image',name='apply2mmask')
apply2mmask.inputs.default_value=0
apply2mmask.inputs.input_image_type=0
apply2mmask.inputs.interpolation='Linear'
apply2mmask.inputs.invert_transform_flags=[True,False]
apply2mmask.inputs.num_threads=1
apply2mmask.inputs.terminal_output='file'

#Normalise sanity check (Realigned EPI)
apply2epiNC=Node(ants.ApplyTransforms(),name='apply2epiNC')
apply2epiNC.inputs.default_value=0
apply2epiNC.inputs.input_image_type=3
apply2epiNC.inputs.interpolation='Linear'
apply2epiNC.inputs.invert_transform_flags=[True,False]
apply2epiNC.inputs.num_threads=1
apply2epiNC.inputs.terminal_output='file'

#Apply transform to non-filtered EPIs (for FALFF ETC)
apply2epiNF=MapNode(ants.ApplyTransforms(),iterfield='input_image',name='apply2epiNF')
apply2epiNF.inputs.default_value=0
apply2epiNF.inputs.input_image_type=3
apply2epiNF.inputs.interpolation='Linear'
apply2epiNF.inputs.invert_transform_flags=[True,False]
apply2epiNF.inputs.num_threads=1
apply2epiNF.inputs.terminal_output='file'

#Datasink
substitutions = ('_subject_id_', '')
sink = Node(DataSink(), name = 'sink')
sink.inputs.base_directory = out_dir
sink.inputs.substitutions = substitutions


preproc = Workflow(name = 'healthy_preproc')
preproc.base_dir = work_dir

                 ####POPULATE INPUTS, GET DATA, DROP EPI VOLS, GENERAL HOUSEKEEPING###
preproc.connect([(infosource, selectfiles,[('subject_id','subject_id')]),
                 (selectfiles, dropvols, [('epi','epi_list')]),
                 (dropvols, epi_stack, [('epi_list', 'dicom_files')]),
                 (epi_stack, metadata, [('out_file', 'nifti')]),
                 (epi_stack, despike, [('out_file', 'in_file')]),
                 
                 ###HERE BE SLICE TIMING###
                 (metadata,st_corr, [('sliceno','num_slices'),
                                     ('slicetimes','slice_order'),
                                     ('tr','time_repetition'),
                                     ('ta','time_acquisition'),
                                     ('mid_slice', 'ref_slice')]),
                 (despike,st_corr, [('out_file', 'in_files')]),
                 
                 ###REALIGNMENT / TSNR / SEGMENTATION###
                 (st_corr, realign, [('timecorrected_files','in_files')]),                 
                 #(realign, motion_plot, [('realignment_parameters', 'motion_parameters')]),
                 (realign, tsnr, [('realigned_files','in_file')]),
                 (selectfiles, anat_stack, [('anat','dicom_files')]),
                 (anat_stack, seg, [('out_file','channel_files')]),
                 
                 ###CREATE MASKS FOR aCOMPCOR###
                 (seg,mask_list, [('native_class_images','seg')]),
                 (mask_list,noisemask, [('wm','wm_in'),
                                       ('csf','csf_in')]),
                 
                 ###COREG TO EPI STARTS###             
                 (tsnr, coreg2epi, [('mean_file','target')]),
                 (anat_stack, coreg2epi, [('out_file','source')]),
                 (seg, seg_list, [('native_class_images', 'seg')]),
                 (noisemask, seg_list, [('wm_mask_fn','wmnoise'),
                                      ('csf_mask_fn','csfnoise')]),
                 (seg_list, coreg2epi, [('coreg_list','apply_to_files')]),

                 ###POPULATE COREG LISTS, MAKE MATTER MASKS###
                 (coreg2epi,anat2epi_list,[('coregistered_files','coreg_files'),
                                       ('coregistered_source','source')]),
                 (anat2epi_list,mmaskcalc,[('gm','gm'),
                                           ('wm','wm'),
                                           ('csf','csf')]),
                 (coreg2epi, mmaskcalc, [('coregistered_source', 'anat')]),
                 (mmaskcalc, mmask_list,[('mmask_fn','mmask_fn'),
                                       ('brain_ss_fn','brain_ss_fn')]),

                 ###CLEANING / FILTERING###
                 (realign, motion_regressor, [('realignment_parameters', 'motion_params_fn')]),
                            
                 (anat2epi_list, wm_regressor, [('wmnoise', 'wm_mask_fn')]),
                 (realign, wm_regressor, [('realigned_files', 'epi_fn')]),
                 
                 (anat2epi_list, csf_regressor, [('csfnoise','csf_mask_fn')]),
                 (realign, csf_regressor, [('realigned_files', 'epi_fn')]),
                 
                 #Run compcor
                 (realign, compcor_clean,[('realigned_files','epi_fn')]),
                 (mmaskcalc, compcor_clean, [('mmask_fn','global_mask_fn')]),
                 (wm_regressor, compcor_clean, [('wm_comps_fn', 'wm_noise_fn')]),
                 (csf_regressor, compcor_clean, [('csf_comps_fn', 'csf_noise_fn')]),
                 (motion_regressor, compcor_clean,[('motion24_fn','motion24_fn')]),                 
                 
                 (compcor_clean, clean_list, [('global_img_fn','global_img_fn'),
                                            ('noglobal_img_fn','noglobal_img_fn')]),
                 
                 #Filter
                 (clean_list, bpfilter,[('cleaned','epi_fn')]),
                 (mmaskcalc, bpfilter,[('mmask_fn','global_mask_fn')]),
                 
                 ###COMPUTE TRANSFORM TO MNI###
                 (mmaskcalc, antsnorm,[('brain_ss_fn', 'fixed_image')]),
                 (selectfiles, antsnorm,[('mni_template', 'moving_image')]),
                
                 ###POPULATE ANTS OUTPUT LIST###
                 (antsnorm, ants_list,[('reverse_transforms', 'ants_output')]),
                 
                 ###APPLY TRANSFORM TO T1 (test warp quality)###
                 (coreg2epi, apply2anat,[('coregistered_source', 'input_image')]),
                 (selectfiles, apply2anat,[('mni_template', 'reference_image')]),
                 (ants_list, apply2anat,[('trans','transforms')]),                 
                 
                 ###APPLY TRANSFORM TO EPI###
                 (bpfilter, apply2epi, [('filter_img_fn','input_image')]),
                 (selectfiles, apply2epi, [('mni_template','reference_image')]),
                 (ants_list, apply2epi, [('trans','transforms')]),

                 ###APPLY TRANSFORM TO EPI NO FILTER###
                 (clean_list,apply2epiNF,[('cleaned','input_image')]),
                 (selectfiles,apply2epiNF,[('mni_template','reference_image')]),
                 (ants_list,apply2epiNF,[('trans','transforms')]),

                 ###APPLY TRANSFORM TO EPI NO CLEAN###
                 (realign, apply2epiNC, [('realigned_files','input_image')]),
                 (selectfiles,apply2epiNC,[('mni_template','reference_image')]),
                 (ants_list,apply2epiNC, [('trans','transforms')]),

                 ###APPLY TRANSFORM TO MATTER MASK###
                 (mmask_list, apply2mmask,[('mmask_files','input_image')]),
                 (selectfiles, apply2mmask,[('mni_template','reference_image')]),
                 (ants_list, apply2mmask,[('trans','transforms')]),

                 ###LIST FOR SMOOTHING###
                 (apply2epi,smooth_list,[('output_image','list1')]),
                 (apply2epiNF,smooth_list,[('output_image','list2')]),
                 (apply2epiNC,smooth_list,[('output_image','list3')]),

                 ###SMOOTH EPI###
                 (smooth_list,smooth,[('smoothing_files', 'in_files')]),

                 ###COMPUTE AAL CORRELATION MATRIX###
                 (smooth,aal_corrmat,[('smoothed_files','smoothed_files')]),

                 ###GRAB OUTPUTS###
                 (infosource,sink,[('subject_id','container')]),
                 (infosource,sink,[('subject_id','strip_dir')]),

                 (dropvols,sink,[('volsdropped_fn','QC.@vols')]),

                 (smooth,sink,[('smoothed_files','preproc_epis')]),

                 (realign, sink,[('realignment_parameters','QC@motion_params')]),

                 #(motion_plot, sink,[('motion_plot_fn','QC@motion_plot')]),
                 (tsnr,sink,[('stddev_file','QC.@std'),

                             ('tsnr_file','QC.@tsnr')]),

                 (apply2anat,sink,[('output_image','mni_warped.@anat')]),
                 (apply2mmask,sink,[('output_image','mni_warped.@mask')]),
                 (wm_regressor, sink, [('wm_drop_fn','QC.@wm_drop'),
                                      ('wm_var_fn','QC.@wm_var'),
                                      ('wm_cumvar_fn','QC.@wm_cumvar')]),
                                      
                  (csf_regressor, sink, [('csf_drop_fn','QC.@csf_drop'),
                                      ('csf_var_fn','QC.@csf_var'),
                                      ('csf_cumvar_fn','QC.@csf_cumvar')]),

                 (compcor_clean,sink,[('global_regs_fn','QC.@global_regs'),
                                      ('noglobal_regs_fn','QC.@noglobal_regs')]),

                 (antsnorm,sink,[('forward_transforms','transforms.@forwardtrans'),
                                 ('reverse_transforms','transforms.@reversetrans'),
                                 ('warped_image','transforms.@warped'),
                                 ('inverse_warped_image','transforms.@invwarped')]),

                (aal_corrmat,sink,[('global_corr_fn','corrmat.@global'),
                                    ('global_lasso_fn','lasso.@global'),
                                    ('global_corr_trans_fn','corrmat.@global_trans'),
                                    ('noglobal_corr_fn','corrmat.@noglobal'),
                                    ('noglobal_lasso_fn','lasso.@noglobal'),
                                    ('noglobal_corr_trans_fn','corrmat.@noglobal_trans'),
                                    ('noclean_corr_fn','corrmat.@noclean'),
                                    ('noclean_lasso_fn','lasso.@noclean'),
                                    ('noclean_corr_trans_fn','corrmat.@noclean_trans')]),
                 ])


preproc.write_graph(graph2use = 'colored', format = 'svg', simple_form = False)
#preproc.run()
preproc.run(plugin = 'MultiProc', plugin_args = {'n_procs': 3})

stop = time.time() - start
print "Time taken to complele analysis was " + str(stop) + " seconds"
