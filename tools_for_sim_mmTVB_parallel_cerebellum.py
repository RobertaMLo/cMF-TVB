"""
Cerebellum TVB
===================================================================================================================================
Cerebellum  TVB is developed to integrate the cerebellar MF model into TVB platform (Lorenzi et al., Plos Cmput Biol., 2023)
===================================================================================================================================
code prepared by robertalorenzi

Library of functions for running a TVB CEREBELLUM simulation
-------------------------------------------------------------------------------------------------------------------------------------
code prepared by robertalorenzi
March 2024, v1 - rev.00
------------------------------------------------------------------------------------------------------------------------------------
@author robilorenzi
"""




from tvb_model_reference.simulation_file.parameter.parallel_crbl_params import Parameter
import tvb_model_reference.src.parallel_tool_crbl as tools

from tvb.simulator.lab import *
import numpy as np

import os
import time




def init_TVBparameters(parameters, n_regions, SC_path_name, output_folder, simname, norm = False):

    parameters.parameter_connection_between_region['from_file'] = True
    parameters.parameter_connection_between_region['from_h5'] = False

    parameters.parameter_connection_between_region['path']= SC_path_name
    parameters.parameter_connection_between_region['number_of_regions']= n_regions
    parameters.parameter_connection_between_region['normalised'] = norm #normalised during creation

    try:
        os.listdir(output_folder)
    except:
        os.mkdir(output_folder)

    parameters.parameter_simulation['path_result'] = output_folder + simname

    print(parameters.parameter_simulation['path_result'])
    print(parameters.parameter_connection_between_region['path'])

    return parameters


def set_external_afferent_input(parameters, Iext):
    
    parameters.parameter_model['external_input_ex_ex']=Iext #afferent input
    parameters.parameter_model['external_input_in_ex']=Iext
    
    print('Afferent External Input: ',parameters.parameter_model['external_input_ex_ex'])

    return parameters


def set_TVBsimulator(parameters):
    simulator = tools.init(parameters.parameter_simulation,
                          parameters.parameter_model,
                          parameters.parameter_connection_between_region,
                          parameters.parameter_coupling,
                          parameters.parameter_integrator,
                          parameters.parameter_monitor)
    
    print('Default coupling set in parameters: ',simulator.coupling.a)
    
    return simulator


def set_external_stimulus(parameters, simulator, sim_len, cut_trans, stim_strength = 0, stimtime_mean= 2500., interstim_interval = 1e9, 
                          stim_dur =50, stim_region = 1, stim_state_var = [0,1]):
    
    """
    
    == 
    Function to include a stimulus in TVB -- need to define stim parameters, the target region, the kiccked state variables and to re-init the simulator in order to set the stimulus
    ==
    
    Parameters of stimulus are in ms
    stim_strength should be in kH --> TO BE CHECKED
    stimtime_mean = time after SIMULATION start --> TO BE CHECKED WHAT IT IS, ANYWAY NOT USED
    interstim_interval = interstimulus interval [ms]
    stim_dur = duration of the stimulus [ms]
    stim_node = target node (e.g., stimulated region)
    stim_state_var = state variable affected by stimulus
    
    """
  
    weight = list(np.zeros(simulator.number_of_nodes)) # need to set simulator before to get the number of nodes.
    weight[stim_region] = stim_strength # region and stimulation strength of the region 0 

    parameters.parameter_stimulus["tau"]= stim_dur
    parameters.parameter_stimulus["T"]= interstim_interval
    parameters.parameter_stimulus["weights"]= weight
    parameters.parameter_stimulus["variables"]=stim_state_var
    parameters.parameter_stimulus['onset'] = cut_trans + 0.5*(sim_len-cut_trans)


    stim_time = parameters.parameter_stimulus['onset']
    stim_steps = stim_time*10 #number of steps until stimulus --> NOT ISED

    #print('\nStrength of the Stimulus:', stimval, 'Duration: ', stimdur)

    simulator = tools.init(parameters.parameter_simulation,
                          parameters.parameter_model,
                          parameters.parameter_connection_between_region,
                          parameters.parameter_coupling,
                          parameters.parameter_integrator,
                          parameters.parameter_monitor,
                          parameter_stimulation=parameters.parameter_stimulus)

    return parameters, simulator


def set_coupling_scaling_crbl(simulator, n_regions,crbl_start_idx, a_crbl):

    """
    
    == 
    Function to set the coupling -- setting directly in the simulator, without passing throught parameters class
    ==
    
    :: Input ::
    simulator = TVB object defined using tool-init (see set_simulator function)
    n_regions = Int. N nodes in SC
    DCN idx = Int array. Index of DCNs
    crbl_start_idx = Int. Starting index of cerebellar nodes (i.e., index of the first cerebellar node in the matrix)
    a_crbl = Float. scaling of crbl MF
    a_cereb = Float. scaling of cerebral MF
    a_DCN = Float. Scaling of DCN MF

    ::Return::
    simulator (with the coupling updated as above)    
    """
    
    # Last modifications to simulator. Some components cannot be easly modified from parameters.parameter_type_of_params, 
    # but they can be directly modified insiede the object Simulator

    ## ERICE TUNED ----------------------------------------------------------------------------
    #bb = np.array(np.ones(shape=(126,1)))*0.5
    #simulator.coupling.a = bb
    #simulator.coupling.a[93:] = 0.0025
    #simulator.coupling.a[103:105] = 0.5
    #simulator.coupling.a[113:115] = 0.5

    ## DEFAULT --------------------------------------------------------------------------------
    #simulator.coupling.a = np.array(np.ones(shape=(126,1)))*0.0039

    ## AdEX TVB --------------------------------------------------------------------------------
    #0.20 last usage of AdEx TVB

    ##

    simulator.coupling.a = np.array(np.ones(shape=(n_regions,1)))

    simulator.coupling.a[crbl_start_idx:] = a_crbl

    
    return simulator


def set_Integrator(simulator, D):
    """
    D = Standard decviation of the noise
    """
    # Old configuration, no more used
    heunstoc=integrators.HeunStochastic(dt=0.1, noise=noise.Additive(nsig=np.array([(D**2)/2])))
    simulator.integrator = heunstoc
    
    return simulator


def saving_TVB_output(tavgtime, tavgdata, boldtime, bolddata, output_dir, simname, bool_to_save=True):
    
    np.save(output_dir+simname+'_timeseries_time.npy', tavgtime)
    np.save(output_dir+simname+'_timeseries_data_crbl.npy', tavgdata[:,:,:])
    np.save(output_dir+simname+'_bold_time.npy', boldtime)
    np.save(output_dir+simname+'_bold_data.npy', bolddata)

    print('Output Saved, Simulation name: ', simname)


def compute_simulated_fc(bolddata):
    
    """
    == 
    Function to compute the SIMULATED FUNCTIONAL CONNECTIVITY
    ==
    
    ::Input::
    bolddata =  TVB timeseries simulated using monitor BOLD
    
    ::Return::
    simulated_FC = PCC matrix, with 0 on the diagonal
    
    """
    
    simulated_FC = np.corrcoef(bolddata.T)
    np.fill_diagonal(simulated_FC,0)
    
    return simulated_FC




