"""
Cerebellum TVB
===================================================================================================================================
Cerebellum  TVB is developed to integrate the cerebellar MF model into TVB platform (Lorenzi et al., Plos Cmput Biol., 2023)
===================================================================================================================================
code prepared by robertalorenzi

@author robilorenzi
"""

from tools_for_sim_mmTVB_parallel_cerebellum import *
import argparse


def main():


    parser = argparse.ArgumentParser(description="\
                                    TVB multimodel.\
                                    Script for running a simulation.\
                                    Usage: sun_sim_mmTVB_parallel SUB_ID root_path prot_folder [options]\
                                     ")
    
    parser.add_argument("SUB_ID", help="Subject ID")
    parser.add_argument("--prot_folder", help="Full path of Protocol folder", default='/home/bcc/HCP_TVBmm_30M')
    parser.add_argument("--conn_zip_name", help="SC zip folder name", default='/T1w/SC_dirCB_ONLYCRBL.zip')
    parser.add_argument("--nregions", help="Number of regions in SC", type = int, default = 27)
    parser.add_argument("--crbl_start_idx", help="First index of the cerebellum", type = int, default = 0)
    parser.add_argument("--sim_len", help="simulation length in ms", type=float, default=120000.)
    parser.add_argument("--id_sim", help="Id for the simulation protocol", default='parallel_ONLYcrbl')
    parser.add_argument("--a_crbl", help="Coupling (scaling) for cerebellar MF", type=float, default=1.)

    args = parser.parse_args()

    SUB_ID = args.SUB_ID
    prot_folder = args.prot_folder
    sim_len = args.sim_len
    id_sim = args.id_sim
    a_crbl = args.a_crbl
    SC_zipfolder = args.conn_zip_name



    SUB_DIR = prot_folder + '/' + SUB_ID
    SC_path_name = SUB_DIR + SC_zipfolder
    n_regions = args.nregions

    #Saving output ---------------------------------------------------------------------------------------------------------------------

    output_folder = SUB_DIR + '/TVB_output/'

    id_crbl = 0

    ## TVB INITIALIZATION  ==============================================================================================================
    # Parameters initialisation ---------------------------------------------------------------------------------------------------------
    parameters = Parameter()

    date_time = time.strftime("%Y%m%d_%H%M%S")
    simname = date_time + id_sim +'_crbl_sim'+str(int(sim_len*1e-3))

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
    simulator_last = set_coupling_scaling_crbl(simulator_post_stim, n_regions, 
                                            crbl_start_idx=args.crbl_start_idx, a_crbl=args.a_crbl)


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
    saving_TVB_output(tavgtime, tavgdata, boldtime, bolddata, output_folder, simname, bool_to_save=True)


    sim_FC = compute_simulated_fc(bolddata)
    np.savetxt(output_folder+'simFC/'+simname+'_FC_ONLYCRBL', sim_FC)


    
if __name__ == "__main__":
    main()
