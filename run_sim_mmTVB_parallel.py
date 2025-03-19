"""
Multi-model TVB
===================================================================================================================================
Multi-model TVB is developed to enable the usage of different models for different nodes (Lorenzi et al., Plos Cmput Biol., 2023)
was integrated into TVB formalism to model the cerebellar nodes activity, and then it was connected with Wong Wang model 
(Wong Wang, Journal of Neurosci., 2006;  Deco et al., Journal of Neurosci., 2014)
===================================================================================================================================
code prepared by robertalorenzi

"""

from tools_for_sim_mmTVB_parallel import *
import argparse
import numpy as np
import os


def main():


    parser = argparse.ArgumentParser(description="\
                                    TVB multimodel.\
                                    Script for running a simulation.\
                                    Usage: sun_sim_mmTVB_parallel SUB_ID root_path prot_folder [options]\
                                     ")
    
    parser.add_argument("SUB_ID", help="Subject ID")
    parser.add_argument("--prot_folder", help="Full path of Protocol folder", default='/home/bcc/HCP_TVBmm_30M')
    parser.add_argument("--conn_zip_name", help="SC zip folder name", default='/T1w/SC_dirCB.zip')
    parser.add_argument("--nregions", help="Number of regions in SC", type = int, default = 126)
    parser.add_argument("--crbl_start_idx", help="First index of the cerebellum", type = int, default = 93)
    parser.add_argument("--DCN_idx", help="DCN index", default=np.array([103, 104, 105, 113, 114, 115]))
    parser.add_argument("--sim_len", help="simulation length in ms", type=float, default=60000.)
    parser.add_argument("--id_sim", help="Id for the simulation protocol", default='parallel_crbl_ww')
    parser.add_argument("--a_crbl", help="Coupling (scaling) for cerebellar MF", type=float, default=1.)
    parser.add_argument("--a_cereb", help="Coupling (scaling) for cerebral MF", type = float, default=1.)
    parser.add_argument("--a_DCN", help="Coupling (scaling) for DCN MF", type =float, default=1.)

    args = parser.parse_args()

    SUB_ID = args.SUB_ID
    prot_folder = args.prot_folder
    sim_len = args.sim_len
    id_sim = args.id_sim
    a_crbl = args.a_crbl
    a_cereb = args.a_cereb
    a_DCN = args.a_DCN
    SC_zipfolder = args.conn_zip_name



    SUB_DIR = os.path.join(prot_folder, SUB_ID)
    SC_path_name = SUB_DIR + SC_zipfolder
    n_regions = args.nregions

    #Saving output ---------------------------------------------------------------------------------------------------------------------

    output_folder = os.path.join(SUB_DIR,'TVB_output/')

    id_cortex = args.crbl_start_idx #10 #set the final index for saving timeseries (avoid to have super long file)
    id_crbl = n_regions #126 #same as above


    ## TVB INITIALIZATION  ==============================================================================================================
    # Parameters initialisation ---------------------------------------------------------------------------------------------------------
    parameters = Parameter()

    date_time = time.strftime("%Y%m%d_%H%M%S")
    simname = date_time + id_sim +'_ctx'+str(id_cortex)+'_crbl'+str(id_crbl)+'_sim'+str(int(sim_len*1e-3))

    parameters_init = init_TVBparameters(parameters, n_regions, SC_path_name, output_folder, simname, norm = False)

    #Afferent input for the moment is set at 0
    parameters_stim = set_external_afferent_input(parameters_init, Iext=0.00035*0)

    #Simulator initialisation ---------------------------------------------------------------------------------------------------------
    simulator_init = set_TVBsimulator(parameters_stim)


    ## TVB SETTING ====================================================================================================================

    # Stimulation setting -------------------------------------------------------------------------------------------------------------
    # For the moment is set at 0 --> NO stimulus
    parameters_post_stim, simulator_post_stim  = set_external_stimulus(parameters_stim, simulator_init, sim_len, cut_trans = 2500., stim_strength = 0, stimtime_mean= 2500., interstim_interval = 1e9, 
                          stim_dur =50, stim_region = 1, stim_state_var = [0,1])

    # Coupling setting ----------------------------------------------------------------------------------------------------------------
    # Coupling = 1
    simulator_post_coupl = set_coupling_scaling_cereb_crbl(simulator_post_stim, n_regions, DCN_idx=args.DCN_idx, 
                                            crbl_start_idx=args.crbl_start_idx, a_crbl=args.a_crbl, a_cereb=args.a_cereb, a_DCN=args.a_DCN)



    # Negative setting of DCN connectivity --------------------------------------------------------------------------------------------
    simulator_last = set_connectivity_DCN(simulator_post_coupl, DCN_idx=args.DCN_idx, negative_scaling = -1)

    print('Coupling ', simulator_last.coupling.a[:])
    #print('Connectivity ', simulator_last.connectivity.weights[103])
    print('Models ', simulator_last.model.I_ext)
    print('Integrator ', simulator_last.integrator)



    ## TVB RUNNING ====================================================================================================================
    # Running simulation -------------------------------------------------------------------------------------------------------------
    
    t = time.localtime()
    current_time_start = time.strftime("%H:%M:%S", t)
    print('============== SIMULATION STARTS AT: ',current_time_start)

    tavgtime,tavgdata, boldtime, bolddata = tools.run_sim_for_bold(simulator_last, sim_len)

    t2 = time.localtime()
    current_time_end = time.strftime("%H:%M:%S", t2)
    print('============== SIMULATION ENDS AT: ',current_time_end)

    print('Shapes\n:bold: ',np.shape(bolddata),'\ntimeseries: ',np.shape(tavgdata))
    # Saving output -------------------------------------------------------------------------------------------------------------------
    saving_TVB_output(tavgtime, tavgdata, boldtime, bolddata, output_folder, simname, id_cortex, id_crbl, bool_to_save=True)
    
if __name__ == "__main__":
    main()
