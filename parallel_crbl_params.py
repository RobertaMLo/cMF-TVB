
"""
parameters of the Cerebellar MF model [(Lorenzi et al., Plos Cmput Biol., 2023)] 

@author: robilorenzi
"""

import os

class Parameter :
    def __init__(self):
        path = os.path.dirname(os.path.abspath(__file__))
        self.parameter_simulation={
            'path_result':'./result/synch/',
            'seed':10, # the seed for the random generator
            'save_time': 1000.0, # the time of simulation in each file
        }

        self.parameter_model ={
            #parameters to compute the statistical moments and to solve MF equations
            'matteo':False,
            'robi': True,
            #order of the model
            'order':1, #choose between 1 and 2
            #Single cell parameters
            'g_L_grc': 0.29,
            'g_L_goc': 3.30,
            'g_L_mli': 1.60,
            'g_L_pc': 7.10,
            'E_L_grc': -62.0,
            'E_L_goc': -62.0,
            'E_L_mli': -68.0,
            'E_L_pc': -59.0,
            'C_m_grc': 7.0,
            'C_m_goc': 145.0,
            'C_m_mli': 14.60,
            'C_m_pc': 334.0,
            'E_e': 0.0,
            'E_i': -80.0,
            #Synaptic parameters
            'Q_mf_grc': 0.230,
            'Q_mf_goc': 0.24,
            'Q_grc_goc': 0.437,
            'Q_grc_mli': 0.154,
            'Q_grc_pc': 1.126,
            'Q_goc_goc': 1.12,
            'Q_goc_grc': 0.336,
            'Q_mli_mli': 0.532,
            'Q_mli_pc': 1.244,
            'tau_mf_grc': 1.9,
            'tau_mf_goc': 5.0,
            'tau_grc_goc': 1.25,
            'tau_grc_mli': 0.64,
            'tau_grc_pc': 1.1,
            'tau_goc_goc': 5.0,
            'tau_goc_grc': 4.5,
            'tau_mli_mli': 2.0,
            'tau_mli_pc': 2.8,
            'N_tot_grc': 28615,
            'N_tot_goc': 70,
            'N_tot_mli': 446,
            'N_tot_pc': 99,
            'N_tot_mossy': 2336,
            'K_mf_grc': 4.0,
            'K_mf_goc': 35.0,
            'K_grc_goc': 501.98,
            'K_grc_mli': 243.96,
            'K_grc_pc': 374.50,
            'K_goc_goc': 16.2,
            'K_goc_grc': 2.50,
            'K_mli_mli': 14.20,
            'K_mli_pc': 10.28,
            'T': 3.5,
            'P_GrC_e': [-0.426,  0.007,  0.023, 0.482, 0.216],
            'P_GoC_i': [-0.144,  0.003, 0.011, 0.031, 0.011],
            'P_MLI_i': [-0.128, -0.001, 0.012, -0.093, -0.063],
            'P_PC_i': [-0.080, 0.009, 0.004, 0.006, 0.014],
            'external_input_ex_ex':0.315*1e-3,
            'external_input_ex_in':0.000,
            'external_input_in_ex':0.315*1e-3,
            'external_input_in_in':0.000,
            'tau_OU': 3.5,
            'weight_noise': 4e-3,
            #'K_ext_e': 400, #Kmf_grc
            #'K_ext_i':0,
            #Initial condition :
            'initial_condition':{
                
                "d1": [0.5*1e3, 0.5*1e3], "d2": [5*1e3, 5*1e3], "d3": [15*1e3, 15*1e3], "d4": [38*1e3, 38*1e3], "noise":[0.0,0.0]} 
        }

        self.parameter_connection_between_region={
            ## CONNECTIVITY
            # connectivity by default
            'default':False,
            #from file (repertory with following files : tract_lengths.npy and weights.npy)
            'from_file':False,
            'from_h5':False,
            'from_folder':True,
            'path':path+'/../../data/Mouse_512/Connectivity_nuria_v1.h5', #the files
            # File description
            'number_of_regions':512, # number of regions
            # lenghts of tract between region : dimension => (number_of_regions, number_of_regions)
            'tract_lengths':[],
            # weight along the tract : dimension => (number_of_regions, number_of_regions)
            'weights':[],
            # speed of along long range connection
            'speed':3.0,
            'normalised':True
        }

        self.parameter_coupling={
            ##COUPLING
            'type':'Linear', # choice : Linear, Scaling, HyperbolicTangent, Sigmoidal, SigmoidalJansenRit, PreSigmoidal, Difference, Kuramoto
            'parameter':{'a':0.25, # changed from 0.45 to 0.25 (14/09/2021) then to 0.20 (15/09/2021)
                         'b':0.0}
        }

        self.parameter_integrator={
            ## INTEGRATOR
            'type':'Heun', # choice : Heun, Euler #per il momento lascio Heun stocastico dai
            'stochastic':True,
            'noise_type': 'Additive', #'Multiplicative', #'Additive', # choice : Additive
            'noise_parameter':{
                #'nsig':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                #'nsig': [(0.001**2)/2],
                'nsig':[(0.001**2)/2],
                'ntau':0.0,
                'dt': 0.1
                                },
            'dt': 0.1 # in ms
        }

        self.parameter_monitor= {
            'Raw':True,
            'TemporalAverage':False,
            'parameter_TemporalAverage':{
                'variables_of_interest':[0,1,2,3,4], #Interested in activities and their variances and noise
                'period':self.parameter_integrator['dt']#*10.0
            },
            'Bold':True,
            'parameter_Bold':{
                'variables_of_interest':[0],
                'period':720.0 #TR di HCP
            },
            'Ca':False,
            'parameter_Ca':{
                'variables_of_interest':[0,1,2],
                'tau_rise':0.01,
                'tau_decay':0.1
            }
        }


        self.parameter_stimulus = {
            'onset': 99.0,
            "tau": 9.0,
            "T": 99.0,
            "weights": None,
            "variables":[0, 1] #Zerlaut ha solo 0, metto 0 e 1 perchè arriva anche a GoC
        }

        self.parameter_surface = {
            'run_surface' : True,
            'load_surface': False,
            'h5_filename' : 'Connectivity_nuria_v1.h5',
            'zip_filename': 'Cortex.zip',
            'vertices_to_region_filename' : 'vertices_to_region.npy',
            'region_map_filename' : 'region_map.txt'
        }
