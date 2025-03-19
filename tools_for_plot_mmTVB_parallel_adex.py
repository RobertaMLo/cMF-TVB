
"""
Multi-model TVB - tools for running the simulations
===================================================================================================================================
Multi-model TVB is developed to enable the usage of different models for different nodes (Lorenzi et al., Plos Cmput Biol., 2023)
was integrated into TVB formalism to model the cerebellar nodes activity, and then it was connected with Wong Wang model 
(Wong Wang, Journal of Neurosci., 2006;  Deco et al., Journal of Neurosci., 2014)
===================================================================================================================================

Library of functions for visualizing the output of a TVB multi model simulations

-------------------------------------------------------------------------------------------------------------------------------------
code prepared by robertalorenzi
March 2024, v1 - rev.00
------------------------------------------------------------------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(dateofres, sub_tvb_outfolder):
    
    data = {}

    files = os.listdir(sub_tvb_outfolder)


    matching_files = [f for f in files if f.startswith(dateofres)]
    
    for file_name in matching_files:

        # If on fixed part of the filenames defined in run_sim.
        if 'timeseries_data_cortex' in file_name:
            data['tseries_ctx'] = np.load(os.path.join(sub_tvb_outfolder, file_name), allow_pickle=True)
        
        elif 'timeseries_data_crbl' in file_name:
            data['tseries_crbl'] = np.load(os.path.join(sub_tvb_outfolder, file_name), allow_pickle=True)
        
        elif 'timeseries_time' in file_name:
            data['tseries_t'] = np.load(os.path.join(sub_tvb_outfolder, file_name), allow_pickle=True)
        
        elif 'bold_data' in file_name:
            data['bolddata'] = np.load(os.path.join(sub_tvb_outfolder, file_name), allow_pickle=True)
        
        elif 'bold_time' in file_name:
            data['bold_t'] = np.load(os.path.join(sub_tvb_outfolder, file_name), allow_pickle=True)

    t_mf, mf_act_crbl, mf_act_cerebral, boldt, bolddata = \
        data['tseries_t'], data['tseries_crbl'], data['tseries_ctx'],data['bold_t'],data['bolddata']


    return t_mf, mf_act_crbl, mf_act_cerebral, boldt, bolddata


def load_data_only_crbl(dateofres, sub_tvb_outfolder):
    
    data = {}

    files = os.listdir(sub_tvb_outfolder)


    matching_files = [f for f in files if f.startswith(dateofres)]
    
    for file_name in matching_files:

        # If on fixed part of the filenames defined in run_sim.
        if 'timeseries_data_crbl' in file_name:
            data['tseries_crbl'] = np.load(os.path.join(sub_tvb_outfolder, file_name), allow_pickle=True)
        
        elif 'timeseries_time' in file_name:
            data['tseries_t'] = np.load(os.path.join(sub_tvb_outfolder, file_name), allow_pickle=True)
        
        elif 'bold_data' in file_name:
            data['bolddata'] = np.load(os.path.join(sub_tvb_outfolder, file_name), allow_pickle=True)
        
        elif 'bold_time' in file_name:
            data['bold_t'] = np.load(os.path.join(sub_tvb_outfolder, file_name), allow_pickle=True)

    t_mf, mf_act_crbl,  boldt, bolddata = \
        data['tseries_t'], data['tseries_crbl'],data['bold_t'],data['bolddata']


    return t_mf, mf_act_crbl, boldt, bolddata


def plot_TVB_crblMF_activity(sub_tvb_outfolder, dateofres, sub_id, t_mf, t_tr, mf_act, show_plot = True, save_bool=True):
    fig, axs = plt.subplots(4, 1, figsize=(5.8,4.1))


    mask = np.ones(len(mf_act[0,0,:]), dtype=bool)
    mask[103-93:106-93] = False
    mask[113-93:116-93] = False

    mf_act_crbl_ctx = mf_act[:,:,mask]

    mf_act_crbl_ctx = mf_act_crbl_ctx*1e3

    
    axs[0].plot(t_mf[t_tr:]*1e-3, mf_act_crbl_ctx[t_tr:,0,:])
    axs[0].set_title('Granule Cells')

    axs[1].plot(t_mf[t_tr:]*1e-3, mf_act_crbl_ctx[t_tr:,1,:])
    axs[1].set_title('Golgi Cells')

    axs[2].plot(t_mf[t_tr:]*1e-3, mf_act_crbl_ctx[t_tr:,2,:])
    axs[2].set_title('Molecular Layer Interneuron')

    axs[3].plot(t_mf[t_tr:]*1e-3, mf_act_crbl_ctx[t_tr:,3,:])
    axs[3].set_title('Purkinje Cells')

    
    axs[3].set_xlabel('time [s] ', fontsize = 11)
    axs[3].set_ylabel('Activity [Hz]', fontsize = 11)
    
    fig.subplots_adjust(hspace=1.05)
    
    if show_plot:
        plt.plot()
    if save_bool:
        plt.savefig(os.path.join(sub_tvb_outfolder,sub_id +'_' + dateofres + '_activityCRBLMF.png'), dpi = 300, bbox_inches='tight')


def plot_TVB_crblBOLD(sub_tvb_outfolder, dateofres, sub_id, bold_t, vol_tr, bolddata, show_plot = True, save_bool=True):
    fig, axs = plt.subplots(2, 1, figsize=(5.8,4.1))

    axs[0].plot(bold_t[vol_tr:], bolddata[vol_tr:, 93:102], bold_t[vol_tr:], bolddata[vol_tr:,116:], bold_t[vol_tr:], bolddata[vol_tr:,106:112])
    axs[0].set_title('Cerebellar BOLD - Cortex')

    axs[1].plot(bold_t[vol_tr:], bolddata[vol_tr:,103:105], bold_t[vol_tr:], bolddata[vol_tr:,113:115])
    axs[1].set_title('Cerebellar BOLD - DCNs')

    
    axs[1].set_xlabel('time [s] ', fontsize = 11)
    axs[1].set_ylabel('BOLD signal', fontsize = 11)
    
    fig.subplots_adjust(hspace=0.5)

    if show_plot:
        plt.show()
    if save_bool:
        plt.savefig(os.path.join(sub_tvb_outfolder, sub_id +'_' + dateofres + '_BOLD_CRBLMF.png'),dpi = 300, bbox_inches='tight')


def plot_TVB_crblBOLD_only_cortex(sub_tvb_outfolder, dateofres, sub_id, bold_t, vol_tr, bolddata, show_plot = True, save_bool=True):
   
    fig, axs = plt.subplots(1, 1, figsize=(5.8,4.1))
    
    axs.plot(bold_t[vol_tr:], bolddata[vol_tr:,:])
    axs.set_title('Cerebellar BOLD - Cortex')

    axs.set_xlabel('time [s] ', fontsize = 11)
    axs.set_ylabel('BOLD signal', fontsize = 11)

    fig.subplots_adjust(hspace=0.5)

    if show_plot:
        plt.show()
    if save_bool:
        plt.savefig(os.path.join(sub_tvb_outfolder, sub_id +'_' + dateofres + '_BOLD_CRBLMF_CORTEX.png'),dpi = 300, bbox_inches='tight')


def plot_TVB_crblBOLD_2pop(sub_tvb_outfolder, dateofres, sub_id, bold_t, vol_tr, bolddata, show_plot = True, save_bool=True):
   
    fig, axs = plt.subplots(1, 1, figsize=(5.8,4.1))
    
    axs.plot(bold_t[vol_tr:], bolddata[vol_tr:,0,:])
    #axs.plot(bold_t[vol_tr:], bolddata[vol_tr:,:])
    axs.set_title('Cerebellar BOLD - Cortex')

    fig.subplots_adjust(hspace=0.5)

    if show_plot:
        plt.show()
    if save_bool:
        plt.savefig(os.path.join(sub_tvb_outfolder + '_' +sub_id +'_' + dateofres + '_BOLD_CRBLMF_2POP.png'),dpi = 300, bbox_inches='tight')


def plot_TVB_cerebral_activity(sub_tvb_outfolder, dateofres, sub_id, t_mf, t_tr, mf_act, show_plot = True, save_bool=True):
    
    fig, axs = plt.subplots(2, 1, figsize=(5.8,4.1))

    axs[0].plot(t_mf[t_tr:]*1e-3, mf_act[t_tr:,0,:]*1e3)
    axs[0].set_title('Excitatory Cells')

    axs[1].plot(t_mf[t_tr:]*1e-3, mf_act[t_tr:,1,:]*1e3)
    axs[1].set_title('Inhibitory Cells')

    
    axs[1].set_xlabel('time [s] ', fontsize = 11)
    axs[1].set_ylabel('Activity [Hz]', fontsize = 11)
    
    fig.subplots_adjust(hspace=1.05)
    
    if show_plot:
        plt.plot()
    if save_bool:
        plt.savefig(os.path.join(sub_tvb_outfolder,sub_id +'_' + dateofres +'_activityCEREBRAL.png'),dpi = 300, bbox_inches='tight')


def plot_TVB_cerebralBOLD(sub_tvb_outfolder, dateofres, sub_id, bold_t, vol_tr, bolddata, show_plot=True, save_bool = True):
    
    fig, axs = plt.subplots(1, 1, figsize=(5.8,4.1))
    axs.plot(bold_t[vol_tr:], bolddata[vol_tr:,:93])
    axs.set_title('Cerebral BOLD')

    axs.set_xlabel('time [s] ', fontsize = 11)
    axs.set_ylabel('BOLD signal', fontsize = 11)
    
    fig.subplots_adjust(hspace=0.4)
    
    if show_plot:
        plt.show()
    if save_bool:
        plt.savefig(os.path.join(sub_tvb_outfolder,sub_id +'_' + dateofres + '_BOLD_Cerebral.png'),dpi = 300, bbox_inches='tight')
