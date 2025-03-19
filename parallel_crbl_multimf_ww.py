#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:32:01 2021

@author: robertalorenzi
"""

"""
Mean field model based on Master equation about adaptative exponential leacky integrate and fire neurons population
"""

from tvb.simulator.models.base import Model,  numpy
from tvb.basic.neotraits.api import NArray, Range, Final, List
import scipy.special as sp_spec
from numba import jit

import numpy as np


class crbl_cortical_first_ord(Model):
    r"""

    WONG WANG Equations taken from Deco 2013,
    CEREBELLAR Equations taken from Lorenzi et al., 2023

    """

    _ui_name = "crbl_cortical_first_ord"
    ui_configurable_parameters = ['g_L_grc', 'g_L_goc', 'g_L_mli', 'g_L_pc', 'E_L_grc', 'E_L_goc','E_L_mli','E_L_pc',
                                    'C_m_grc', 'C_m_goc', 'C_m_mli', 'C_m_pc', 'E_e', 'E_i',
                                    'Q_mf_grc','Q_mf_goc','Q_grc_goc','Q_grc_mli', 'Q_grc_pc','Q_goc_goc','Q_goc_grc','Q_mli_mli','Q_mli_pc',
                                    'tau_mf_grc','tau_mf_goc','tau_grc_goc','tau_grc_mli', 'tau_grc_pc','tau_goc_goc','tau_goc_grc','tau_mli_mli','tau_mli_pc',
                                    'K_mf_grc','K_mf_goc','K_mf_goc','K_grc_mli','K_grc_pc','K_goc_goc','K_goc_grc','K_mli_mli','K_mli_pc',
                                    'N_grc','N_goc','N_mli','N_pc', 'N_mossy', 'T',
                                    'a_e', 'b_e', 'd_e', 'gamma_e', 'tau_e', 'w_p', 'J_N', 'W_e',
                                    'a_i', 'b_i', 'd_i', 'gamma_i', 'tau_i', 'J_i', 'W_i', 'I_o', 'I_ext', 'G', 'lambda']

    # Define traited attributes for this model, these represent possible kwargs.

    ## =============================================================================================================================
    ## ================================ CRBL MF PARAMETERS =========================================================================
    ## =============================================================================================================================

    ### AGGIUGNO DA GRIFFITH LE LABLES COME CONFIG PARAMETERS
    is_cortical = NArray(
        label=":math:`is_cortical`",
        dtype=bool,
        default=np.array([False]),
        required=False,
        doc="""Boolean flag vector for cortical regions""")

    is_crbl = NArray(
        label=":math:`is_crbl`",
        dtype=bool,
        default=np.array([True]),
        doc="""Boolean flag vector for specific cerebellar regions""")


    ## =============================================================================================================================
    ## ================================ CRBL MF PARAMETERS =========================================================================
    ## =============================================================================================================================
    
    g_L_grc =  NArray(
        label=":math:`gGrC_{L}`",
        default=numpy.array([0.29]), 
        domain=Range(lo=0.25, hi=0.35, step=0.1),  
        doc="""Granule cells leak conductance [nS]""")

    g_L_goc =  NArray(
        label=":math:`gGoC_{L}`",
        default=numpy.array([3.30]),  
        domain=Range(lo=3.25, hi=3-35, step=0.1),  
        doc="""Golgi cells leak conductance [nS]""")

    g_L_mli =  NArray(
        label=":math:`gMLI_{L}`",
        default=numpy.array([1.60]), 
        domain=Range(lo=1.55, hi=1.65, step=0.1),  
        doc="""leak conductance [nS]""")

    g_L_pc =  NArray(
        label=":math:`gPC_{L}`",
        default=numpy.array([7.10]),  
        domain=Range(lo=7.5, hi=7.15, step=0.1), 
        doc="""leak conductance [nS]""")

    #Standard deviation (Domanin) from Geminiani et al. 2019
    E_L_grc = NArray(
        label=":math:`EGrC_{L}`",
        default=numpy.array([-62.0]),
        domain=Range(lo=-62.1, hi=-61.9, step=0.1),
        doc="""leak reversal potential for excitatory [mV]""")

    E_L_goc = NArray(
        label=":math:`EGoC_{L}`",
        default=numpy.array([-62.0]),
        domain=Range(lo=-73.0, hi=-51.0, step=0.1), 
        doc="""leak reversal potential for inhibitory [mV]""")

    E_L_mli = NArray(
        label=":math:`EMLI_{L}`",
        default=numpy.array([-68.0]),
        domain=Range(lo=-68.01, hi=-67.9, step=0.1), 
        doc="""leak reversal potential for excitatory [mV]""")

    E_L_pc = NArray(
        label=":math:`E_{L}`",
        default=numpy.array([-59.0]),
        domain=Range(lo=-65.0, hi=-53.0, step=0.1),
        doc="""leak reversal potential for inhibitory [mV]""")

    # N.B. Not independent of g_L, C_m should scale linearly with g_L
    C_m_grc = NArray(
        label=":math:`CGrC_{m}`",
        default=numpy.array([7.0]),
        domain=Range(lo=5.0, hi=7.5, step=1.0), 
        doc="""membrane capacitance [pF]""")

    C_m_goc = NArray(
        label=":math:`CGoC_{m}`",
        default=numpy.array([145.0]),
        domain=Range(lo=72., hi=218.0, step=10.0), 
        doc="""membrane capacitance [pF]""")

    C_m_mli = NArray(
        label=":math:`CMLI_{m}`",
        default=numpy.array([14.6]),
        domain=Range(lo=14.5, hi=14.7, step=0.1),  
        doc="""membrane capacitance [pF]""")

    C_m_pc = NArray(
        label=":math:`CPC_{m}`",
        default=numpy.array([334.0]),
        domain=Range(lo=228.0, hi=440.0, step=10.0),
        doc="""membrane capacitance [pF]""")

    E_e = NArray(
        label=r":math:`E_e`",
        default=numpy.array([0.0]),
        domain=Range(lo=-20., hi=20., step=0.01),
        doc="""excitatory reversal potential [mV]""")

    E_i = NArray(
        label=":math:`E_i`",
        default=numpy.array([-80.0]),
        domain=Range(lo=-100.0, hi=-60.0, step=1.0),
        doc="""inhibitory reversal potential [mV]""")

    Q_mf_grc = NArray(
        label=r":math:`Q_mf_grc_e`",
        default=numpy.array([0.230]),
        domain=Range(lo=0.225, hi=0.235, step=0.001),
        doc="""excitatory quantal conductance [nS]""")

    Q_mf_goc = NArray(
        label=r":math:`Q_mf_goc_e`",
        default=numpy.array([0.240]),
        domain=Range(lo=0.235, hi=0.245, step=0.001),
        doc="""inhibitory quantal conductance [nS]""")

    Q_grc_goc = NArray(
        label=r":math:`Q_grc_goc_e`",
        default=numpy.array([0.437]),
        domain=Range(lo=0.432, hi=0.542, step=0.001),
        doc="""excitatory quantal conductance [nS]""")

    Q_grc_mli = NArray(
        label=r":math:`Q_grc_mli_e`",
        default=numpy.array([0.154]),
        domain=Range(lo=0.149, hi=0.159, step=0.001),
        doc="""excitatory quantal conductance [nS]""")

    Q_grc_pc = NArray(
        label=r":math:`Q_e`",
        default=numpy.array([1.126]),
        domain=Range(lo=1.120, hi=1.131, step=0.001),
        doc="""excitatory quantal conductance [nS]""")

    Q_goc_grc = NArray(
        label=r":math:`Q_goc_grc_i`",
        default=numpy.array([0.336]),
        domain=Range(lo=0.330, hi=0.341, step=0.001),
        doc="""excitatory quantal conductance [nS]""")

    Q_goc_goc = NArray(
        label=r":math:`Q_goc_goc_i`",
        default=numpy.array([1.120]),
        domain=Range(lo=1.115, hi=1.130, step=0.001),
        doc="""excitatory quantal conductance [nS]""")

    Q_mli_mli = NArray(
        label=r":math:`Q_mli_mli_i`",
        default=numpy.array([0.532]),
        domain=Range(lo=0.527, hi=0.537, step=0.001),
        doc="""excitatory quantal conductance [nS]""")

    Q_mli_pc = NArray(
        label=r":math:`Q_mli_pc_i`",
        default=numpy.array([1.244]),
        domain=Range(lo=1.240, hi=1.250, step=0.001),
        doc="""excitatory quantal conductance [nS]""")


    tau_mf_grc = NArray(
        label=":math:`\tau_mf_grc_e`",
        default=numpy.array([1.9]),
        domain=Range(lo=1.5, hi=1.15, step=0.1),
        doc="""excitatory decay [ms]""")

    tau_mf_goc = NArray(
        label=":math:`\tau_mf_goc_e`",
        default=numpy.array([5.0]),
        domain=Range(lo=4.5, hi=5.5, step=0.1),
        doc="""excitatory decay [ms]""")

    tau_grc_goc = NArray(
        label=":math:`\tau_grc_goc_e`",
        default=numpy.array([1.25]),
        domain=Range(lo=1.05, hi=1.45, step=0.1),
        doc="""excitatory decay [ms]""")

    tau_grc_mli = NArray(
        label=":math:`\tau_grc_mli_e`",
        default=numpy.array([0.64]),
        domain=Range(lo=0.44, hi=0.84, step=0.1),
        doc="""excitatory decay [ms]""")

    tau_grc_pc = NArray(
        label=":math:`\tau_grc_pc_e`",
        default=numpy.array([1.1]),
        domain=Range(lo=1., hi=1.2, step=0.1),
        doc="""excitatory decay [ms]""")

    tau_goc_grc = NArray(
        label=":math:`\tau_goc_grc_i`",
        default=numpy.array([4.5]),
        domain=Range(lo=4., hi=5., step=0.1),
        doc="""excitatory decay [ms]""")

    tau_goc_goc = NArray(
        label=":math:`\tau_goc_goc_i`",
        default=numpy.array([5.0]),
        domain=Range(lo=4.5, hi=5.5, step=0.1),
        doc="""excitatory decay [ms]""")

    tau_mli_mli = NArray(
        label=":math:`\tau_mli_mli_i`",
        default=numpy.array([2.0]),
        domain=Range(lo=1.5, hi=2.5, step=0.1),
        doc="""excitatory decay [ms]""")

    tau_mli_pc = NArray(
        label=":math:`\tau_mli_pc_i`",
        default=numpy.array([2.8]),
        domain=Range(lo=2.3, hi=3.2, step=0.1),
        doc="""excitatory decay [ms]""")

    K_mossy_grc = NArray(
        label=":math:`K_mossy_grc_e`",
        default=numpy.array([4.0]),
        domain=Range(lo=0.0, hi=10.0, step=1.0),
        doc="""synaptic convergence [-]""")

    K_mossy_goc = NArray(
        label=":math:`K_mossy_goc_e`",
        default=numpy.array([35.0]),
        domain=Range(lo=15.0, hi=55.0, step=10.0),
        doc="""synaptic convergence [-]""")

    K_grc_goc = NArray(
        label=":math:`K_grc_grc_e`",
        default=numpy.array([501.98]),
        domain=Range(lo=451.98, hi=551.0, step=10.0),
        doc="""synaptic convergence [-]""")

    K_grc_mli = NArray(
        label=":math:`K_grc_mli_e`",
        default=numpy.array([243.96]),
        domain=Range(lo=193.96, hi=293.96, step=10.0),
        doc="""synaptic convergence [-]""")

    K_grc_pc = NArray(
        label=":math:`K_grc_pc_e`",
        default=numpy.array([374.50]),
        domain=Range(lo=334.50, hi=404.50, step=10.0),
        doc="""synaptic convergence [-]""")

    K_goc_goc = NArray(
        label=":math:`K_goc_goc_e`",
        default=numpy.array([16.2]),
        domain=Range(lo=10.2, hi=20.2, step=1.0),
        doc="""synaptic convergence [-]""")

    K_mli_mli = NArray(
        label=":math:`K_mli_mli_i`",
        default=numpy.array([14.20]),
        domain=Range(lo=10.20, hi=20.20, step=1.0),
        doc="""synaptic convergence [-]""")

    K_mli_pc = NArray(
        label=":math:`K_mli_pc_i`",
        default=numpy.array([10.28]),
        domain=Range(lo=5.28, hi=15.28, step=1.0),
        doc="""synaptic convergence [-]""")

    N_grc = NArray(
        dtype=int,
        label=":math:`NGrC_{tot}`",
        default=numpy.array([28615]),
        domain=Range(lo=25615, hi=31615, step=1000),
        doc="""cell number""")

    N_goc = NArray(
        dtype=int,
        label=":math:`NGoC_{tot}`",
        default=numpy.array([70]),
        domain=Range(lo=10, hi=100, step=10),
        doc="""cell number""")

    N_mli = NArray(
        dtype=int,
        label=":math:`NMLI_{tot}`",
        default=numpy.array([446]),
        domain=Range(lo=146, hi=946, step=100),
        doc="""cell number""")

    N_pc = NArray(
        dtype=int,
        label=":math:`NPC_{tot}`",
        default=numpy.array([99]),
        domain=Range(lo=29, hi=149, step=10),
        doc="""cell number""")

    N_mossy = NArray(
        dtype=int,
        label=":math:`Nmossy_{tot}`",
        default=numpy.array([2336]),
        domain=Range(lo=336, hi=5336, step=1000),
        doc="""cell number""")

    alpha_grc = NArray(
        dtype=float,
        label=":math:`alphaGrC`",
        default=numpy.array([2]),
        domain=Range(lo=2, hi=2, step=1),
        doc="""cell number""")

    alpha_goc = NArray(
        dtype=float,
        label=":math: alphaGoC",
        default=numpy.array([1.3]),
        domain=Range(lo=1.3, hi=1.3, step=1),
        doc="""cell number""")

    alpha_mli = NArray(
        dtype=float,
        label=":math:`alphaMLI`",
        default=numpy.array([1.8]),
        domain=Range(lo=5, hi=5, step=1),
        doc="""Number of excitatory connexions from external population""")

    alpha_pc = NArray(
        dtype=float,
        label=":math:`alphaPC`",
        default=numpy.array([5]),
        domain=Range(lo=5, hi=5, step=1),
        doc="""Number of inhibitory connexions from external population""")

    T = NArray(
        label=":math:`T`",
        default=numpy.array([3.5]),
        domain=Range(lo=3.45, hi=3.55, step=0.01),
        doc="""time scale of describing network activity""")

    P_grc = NArray(
        label=":math:`PGrC_e`",  
        default=numpy.array([-0.426,  0.007,  0.023, 0.482, 0.216]),
        doc="""Polynome of excitatory GrC phenomenological threshold (order 5)""")

    P_goc = NArray(
        label=":math:`PGoC_i`",
        default=numpy.array([-0.144,  0.003, 0.011, 0.031, 0.011]),
        doc="""Polynome of inhibitory GoC phenomenological threshold (order 5)""")

    P_mli = NArray(
        label=":math:`PMLI_i`",  
        default=numpy.array([-0.128, -0.001, 0.012, -0.093, -0.063]),
        doc="""Polynome of inhibitory phenomenological threshold (order 5)""")

    P_pc = NArray(
        label=":math:`PPC_i`",  
        default=numpy.array([-0.080, 0.009, 0.004, 0.006, 0.014]),
        doc="""Polynome of inhibitory phenomenological threshold (order 5)""")

    tau_OU = NArray(
        label=":math:`\ntau noise`",
        default=numpy.array([5.0]),
        domain=Range(lo=0.10, hi=10.0, step=0.01),
        doc="""time constant noise""")

    ## =============================================================================================================================
    ## ================================ REDUCED WONG WANG PARAMETERS ===============================================================
    ## =============================================================================================================================
    a_e = NArray(
        label=":math:`a_e`",
        default=numpy.array([310., ]),
        domain=Range(lo=0., hi=500., step=1.),
        doc="[n/C]. Excitatory population input gain parameter, chosen to fit numerical solutions.")

    b_e = NArray(
        label=":math:`b_e`",
        default=numpy.array([125., ]),
        domain=Range(lo=0., hi=200., step=1.),
        doc="[Hz]. Excitatory population input shift parameter chosen to fit numerical solutions.")

    d_e = NArray(
        label=":math:`d_e`",
        default=numpy.array([0.160, ]),
        domain=Range(lo=0.0, hi=0.2, step=0.001),
        doc="""[s]. Excitatory population input scaling parameter chosen to fit numerical solutions.""")

    gamma_e = NArray(
        label=r":math:`\gamma_e`",
        default=numpy.array([0.641/1000, ]),
        domain=Range(lo=0.0, hi=1.0/1000, step=0.01/1000),
        doc="""Excitatory population kinetic parameter""")

    tau_e = NArray(
        label=r":math:`\tau_e`",
        default=numpy.array([100., ]),
        domain=Range(lo=50., hi=150., step=1.),
        doc="""[ms]. Excitatory population NMDA decay time constant.""")

    w_p = NArray(
        label=r":math:`w_p`",
        default=numpy.array([1.4, ]),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Excitatory population recurrence weight""")

    J_N = NArray(
        label=r":math:`J_N`",
        default=numpy.array([0.15, ]),
        domain=Range(lo=0.001, hi=0.5, step=0.001),
        doc="""[nA] NMDA current""")

    W_e = NArray(
        label=r":math:`W_e`",
        default=numpy.array([1.0, ]),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Excitatory population external input scaling weight""")

    a_i = NArray(
        label=":math:`a_i`",
        default=numpy.array([615., ]),
        domain=Range(lo=0., hi=1000., step=1.),
        doc="[n/C]. Inhibitory population input gain parameter, chosen to fit numerical solutions.")

    b_i = NArray(
        label=":math:`b_i`",
        default=numpy.array([177.0, ]),
        domain=Range(lo=0.0, hi=200.0, step=1.0),
        doc="[Hz]. Inhibitory population input shift parameter chosen to fit numerical solutions.")

    d_i = NArray(
        label=":math:`d_i`",
        default=numpy.array([0.087, ]),
        domain=Range(lo=0.0, hi=0.2, step=0.001),
        doc="""[s]. Inhibitory population input scaling parameter chosen to fit numerical solutions.""")

    gamma_i = NArray(
        label=r":math:`\gamma_i`",
        default=numpy.array([1.0/1000, ]),
        domain=Range(lo=0.0, hi=2.0/1000, step=0.01/1000),
        doc="""Inhibitory population kinetic parameter""")

    tau_i = NArray(
        label=r":math:`\tau_i`",
        default=numpy.array([10., ]),
        domain=Range(lo=5., hi=100., step=1.0),
        doc="""[ms]. Inhibitory population NMDA decay time constant.""")

    J_i = NArray(
        label=r":math:`J_{i}`",
        default=numpy.array([1.0, ]),
        domain=Range(lo=0.001, hi=2.0, step=0.001),
        doc="""[nA] Local inhibitory current""")

    W_i = NArray(
        label=r":math:`W_i`",
        default=numpy.array([0.7, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Inhibitory population external input scaling weight""")

    I_o = NArray(
        label=":math:`I_{o}`",
        default=numpy.array([0.382, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.001),
        doc="""[nA]. Effective external input""")

    I_ext = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.001),
        doc="""[nA]. Effective external stimulus input""")

    G = NArray(
        label=":math:`G`",
        default=numpy.array([2.0, ]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Global coupling scaling""")

    lamda = NArray(
        label=r":math:`\lambda`",
        default=numpy.array([0.0, ]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Inhibitory global coupling scaling""")

    I_ext_ampl = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([1e3]),
        domain=Range(lo=0.0, hi=1.0, step=0.001),
        doc="""[nA]. Effective external stimulus input""")


    ## =============================================================================================================================
    ## ================================ NOISE and EXT INPT (used for crblMF) ===================================================
    ## =========================================================================================================================

    weight_noise =  NArray(
        label=":math:`\nweight noise`",
        default=numpy.array([10.5]),
        domain=Range(lo=0., hi=50.0, step=1.0),
        doc="""weight noise""")


    external_input_ex_ex = NArray(
        label=":math:`\nu_e^{drive}`",
        default=numpy.array([0.000]),
        domain=Range(lo=0.00, hi=0.1, step=0.001),
        doc="""external drive""")

    external_input_ex_in = NArray(
        label=":math:`\nu_e^{drive}`",
        default=numpy.array([0.000]),
        domain=Range(lo=0.00, hi=0.1, step=0.001),
        doc="""external drive""")

    external_input_in_ex = NArray(
        label=":math:`\nu_e^{drive}`",
        default=numpy.array([0.000]),
        domain=Range(lo=0.00, hi=0.1, step=0.001),
        doc="""external drive""")

    external_input_in_in = NArray(
        label=":math:`\nu_e^{drive}`",
        default=numpy.array([0.000]),
        domain=Range(lo=0.00, hi=0.1, step=0.001),
        doc="""external drive""")



    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"d1": numpy.array([0.0, 0.0]), # GrC, E
                 "d2": numpy.array([0.0, 0.0]), # GoC, I
                 "d3": numpy.array([0.0, 0.0]), # MLI, We
                 "d4": numpy.array([0.0, 0.0]),  # PC, Wi
                 "noise": numpy.array([0.0, 0.0]),
                },

        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random initial
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plot
        """)

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("d1", "d2","d3","d4","noise"), #"GrC", "GoC","MLI","PC","noise"; # E, I, We, Wi, noise
        default=("d1",),
        doc="""This represents the default state-variables of this Model to be
               monitored. It can be overridden for each Monitor if desired""")

    state_variable_boundaries = Final(
        label="Firing rate of population is always positive",
        default={"d1": numpy.array([0.0, None]),
                 "d2": numpy.array([0.0, None]),
                 "d3": numpy.array([0.0, None]),
                 "d4": numpy.array([0.0, None]),
                 },
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. Set None for one-sided boundaries""")

    state_variables = 'd1 d2 d3 d4 noise'.split()

    _nvar = 5
    cvar = numpy.array([0, 3], dtype=int)

    _inds = {}

    ### TO BE UPDATE!! Here Index mask hard-coded, based on AAL+SUIT parcellation (Palesi et al., 2020)
    ### TO DO: Index as a method of this class...or defined as input parameters

    _inds["crbl"]  = np.arange(93,126,1)
    _inds["cortical"] = np.arange(0,93,1)
    _inds["dcn"] = np.array([103, 104, 105, 113, 114, 115])
    _region_label_mock = np.arange(1,127,1)


    _inds["crbl"] = np.arange(len(_region_label_mock)).astype('i') 
    #print("inds after astype\n",inds)
    _inds["crbl"] = np.delete(_inds["crbl"], _inds["cortical"]) #delete inds cortical(second input) from inds crbl(first input)
    #print("inds afeter delete\n", inds)
    _is_cortical = np.array([False] * _region_label_mock.shape[0]).astype("bool")
    #print('shape 0 di region_labels:',region_label_mock.shape[0])
    #print('Iscortical:\n', is_cortical)
    _is_cortical[_inds["cortical"]] = True
    _is_cortical[_inds["dcn"]] = True
    #print('Final Iscortical:\n', _is_cortical)
    _is_crbl = np.logical_not(_is_cortical)
    #print('Final crbl:\n', _is_crbl)

    is_cortical = _is_cortical
    is_crbl = _is_crbl


    def dfun(self, state_variables, coupling, local_coupling=0.00):
        r"""
        .. math::
            T \dot{\nu_\mu} &= -F_\mu(\nu_e,\nu_i) + \nu_\mu ,\all\mu\in\{e,i\}\\
            dot{W}_k &= W_k/tau_w-b*E_k  \\

        """
        d1 = state_variables[0, :]
        d2 = state_variables[1, :]
        d3 = state_variables[2, :]
        d4 = state_variables[3, :]
        noise = state_variables[4, :]
        derivative = numpy.empty_like(state_variables)

        #print('dimension of derivative ********', np.shape(derivative))
        #print('dimension of d1', np.shape(d1))
        #print('COUPLING SHAPE', np.shape(coupling))
        c_0 = coupling[0, :]
        #print('SHAPE C0', np.shape(c_0))
        
        # # now coupling shape is (2,126,1) --> cvar, nodes, mode
        c_3 = coupling[1, :]

        # # CREATING A 126X1 ARRAY WITH 0 FOR CORTICAL INDEX --> NO D4(PC) ACTIVITY FROM CEREBRUM!!!
        c_3_ww = np.ones_like(c_3)*c_3
        c_3_ww[self.is_cortical,:] = 0
        # # CREATING A 126X1 ARRAY WITH 0 FOR CRBL INDEX --> NO D1(GrC) ACTIVITY FROM CEREBELLUM!!!
        c_0_ww = np.ones_like(c_0) * c_0
        c_0_ww[self.is_crbl,:] = 0

        #print('C3 D4 PC PER CEREBRUM: ', c_3[:93])

        """
        # # CHECKING ON INDEX --> HERE TO VISUALIZE BECAUSE IS THE FUNCTION CALL
        print(np.shape(self.is_cortical))
        print(np.shape(self.is_crbl))
        print("check dcn in cortical: ", self.is_cortical[103])
        print("check dcn in crbl: ", self.is_crbl[103])
        """

        # local coupling --> Not yet in Hz because I multiplied it for the activity
        lc_d1 = local_coupling * d1
        lc_d2 = local_coupling * d2
        lc_d3 = local_coupling * d3
        lc_d4 = local_coupling * d4


        c_0_mossy = np.ones_like(c_0)*c_0

        #c_0_mossy[self.is_crbl,:] = 0

        #print('C MOSSY', c_0_mossy)

        c_0_parallel = np.ones_like(c_0)*c_0
        c_0_parallel[self.is_cortical,:] = 0
        
        #print('C PARALLEL', c_0_parallel)
        #print('c_0_parallel[self.is_crbl]',c_0_parallel[self.is_crbl])
        #print('c_0_mossy[self.is_cortical]', c_0_mossy[self.is_cortical])


        Fe_ext_tod1 = c_0_mossy + self.weight_noise * noise

        Fe_ext_tod2 = c_0_mossy + c_0_parallel + self.weight_noise * noise

        # # --------- TO MLI : From PARALLEL of ADJACENT MODULE
        Fe_ext_tod3 = c_0_parallel #+ self.weight_noise * noise

        # # --------- TO PC : From PARALLEL of ADJACENT MODULE
        Fe_ext_tod4 = c_0_parallel #+ self.weight_noise * noise

        # # ---------  TO GRC : only from mossy fibers!!!!!! From DCN or from cerebrum!!!!

        #Fe_ext_tod1 = (c_0 * 0.57) * 0.97 + self.weight_noise * noise #background noise from cerebrum

        #check on F_ext value. Must be positive
        index_bad_input = numpy.where(Fe_ext_tod1*self.K_mossy_grc  < 0)
        Fe_ext_tod1[index_bad_input] = 0.0

        index_bad_input = numpy.where(Fe_ext_tod2*self.K_mossy_goc  < 0)
        Fe_ext_tod2[index_bad_input] = 0.0

        index_bad_input = numpy.where(Fe_ext_tod3*self.K_grc_mli  < 0)
        Fe_ext_tod3[index_bad_input] = 0.0

        index_bad_input = numpy.where(Fe_ext_tod4*self.K_grc_pc  < 0)
        Fe_ext_tod3[index_bad_input] = 0.0

        Fi_ext = 0.0


        ############################## DERIVATIVE 1 ####################################################################
        #d1 = SE

        self.I_ext = (self.weight_noise * noise)*self.I_ext_ampl*0

        #coupling = self.G * self.J_N * (c_0 + lc_d1) #coupling è diverso in wong wang!!!! c'è un parametro G e JN che non ho in altri modelli (infatti in crbl ho solo c0)

        # # Modified wong wang to include coupling with cerebellum
        coupling_ww = self.G * self.J_N * ( (c_0_ww + lc_d1) + (c_3_ww +lc_d3) )

        J_N_S_e = self.J_N * d1

        x_e = self.w_p * J_N_S_e - self.J_i * d2 + self.W_e * self.I_o + coupling_ww + self.I_ext #+ self.weight_noise * noise #added to have same input of MF

        x_e = self.a_e * x_e - self.b_e
        H_e = x_e / (1 - numpy.exp(-self.d_e * x_e))

        derivative[0] = (d1 / self.tau_e) + (1 - d1) * H_e * self.gamma_e


        derivative[0, self.is_crbl] = (self.TF_excitatory_grc(
                        Fe_ext_tod1[self.is_crbl] + self.external_input_ex_ex,
                        d2[self.is_crbl],
                        0,
                        Fi_ext + self.external_input_ex_in,
                        0) - d1[self.is_crbl]) / self.T


        ############################## DERIVATIVE 2 ####################################################################

        # d2 = SI
        x_i = J_N_S_e - d2 + self.W_i * self.I_o + self.lamda * coupling_ww

        x_i = self.a_i * x_i - self.b_i
        H_i = x_i / (1 - numpy.exp(-self.d_i * x_i))

        derivative[1] = - (d2 / self.tau_i) + H_i * self.gamma_i

        # d2 = GoC
        derivative[1, self.is_crbl] = (self.TF_inhibitory_goc(
                        d1[self.is_crbl],
                        d2[self.is_crbl],
                        Fe_ext_tod2[self.is_crbl] + self.external_input_in_ex,
                        Fi_ext,
                        0)- d2[self.is_crbl])/self.T


        ############################## DERIVATIVE 3 ####################################################################
        #d3 =  WW has 2 states variable, here 0
        derivative[2] = 0.0

        # d3 = MLI
        derivative[2, self.is_crbl] = (self.TF_inhibitory_mli(
                        d1[self.is_crbl],
                        d3[self.is_crbl],
                        Fe_ext_tod3[self.is_crbl],
                        Fi_ext,
                        0) - d3[self.is_crbl]) / self.T

        ############################## DERIVATIVE 4 ####################################################################

        #d4 =  WW has 2 states variable, here 0
        derivative[3] = 0.0

        # d4 = PC
        derivative[3, self.is_crbl] = (self.TF_inhibitory_pc(
                        d1[self.is_crbl],
                        d3[self.is_crbl],
                        Fe_ext_tod4[self.is_crbl],
                        Fi_ext,
                        0)- d4[self.is_crbl])/self.T

        derivative[3, self.is_cortical]=  d4[self.is_cortical]*0


        ############################## DERIVATIVE 5 ####################################################################
        ######### noise derivative la lascio pure io che non si sa mai
        derivative[4] = -noise/self.tau_OU

        return derivative


    ### TF erano solo _exc and _inhi ma sono diverse per il return di P-- quindi ne faccio 4
    def TF_excitatory_grc(self, fe_ext, fi, fe, fi_ext=0, W=0):
        """
        transfer function for excitatory population: Granule cells
        return: result of transfer function
        """
        #input TF : Fe, Fi, Fe_ext, Fi_ext, W, P, Q_e, Q_i, tau_e, tau_i, E_e, E_i, g_L, C_m, E_L, Ke, Ki
        return self.TF(fe_ext, fi, fe, fi_ext, W, self.P_grc, self.Q_mf_grc, self.Q_goc_grc, self.tau_mf_grc, self.tau_goc_grc,
                       self.E_e, self.E_i, self.g_L_grc, self.C_m_grc, self.E_L_grc, self.K_mossy_grc, self.K_mossy_goc, self.alpha_grc)

    def TF_inhibitory_goc(self, fe, fi, fe_ext, fi_ext=0, W=0):
        """
        transfer function for inhibitory population: Golgi cells
        :return: result of transfer function
        """
        return self.TF_goc(fe, fi, fe_ext, fi_ext, W, self.P_goc, self.Q_grc_goc, self.Q_goc_goc, self.tau_grc_goc, self.tau_goc_goc,
                           self.E_i, self.E_i, self.g_L_goc, self.C_m_goc, self.E_L_goc, self.K_grc_goc, self.K_goc_goc,
                           self.Q_mf_goc, self.tau_mf_goc, self.K_mossy_goc, self.alpha_goc)
    #(self, Fe, Fi, Fe_ext, Fi_ext, W, P, Qe_gr, Qi, Te_gr, #Ti, Ee, Ei, Gl, Cm, El, Ke_grc, Ki, Qe_ext, Te_ext, Ke_ext, Ki_ext=0):

    def TF_inhibitory_mli(self, fe, fi, fe_ext, fi_ext=0, W=0):
        """
        transfer function for inhibitory population: Molecular layer interneurons
        :return: result of transfer function
        """
        return self.TF(fe, fi, fe_ext, fi_ext, W, self.P_mli, self.Q_grc_mli, self.Q_mli_mli, self.tau_grc_mli, self.tau_mli_mli,
                       self.E_e, self.E_i, self.g_L_mli, self.C_m_mli, self.E_L_mli, self.K_grc_mli, self.K_mli_mli, self.alpha_mli)

    def TF_inhibitory_pc(self, fe, fi, fe_ext, fi_ext=0, W=0):
        """
        transfer function for inhibitory population: Purkinje cells
        :return: result of transfer function
        """
        return self.TF(fe, fi, fe_ext, fi_ext, W, self.P_pc, self.Q_grc_pc, self.Q_mli_pc, self.tau_grc_pc, self.tau_mli_pc,
                       self.E_e, self.E_i, self.g_L_pc, self.C_m_pc, self.E_L_pc, self.K_grc_pc, self.K_mli_pc, self.alpha_pc)

    def TF(self, Fe, Fi, Fe_ext, Fi_ext, W, P, Q_e, Q_i, tau_e, tau_i, E_e, E_i, g_L, C_m, E_L, Ke, Ki, alpha):
        """
        2D transfer functions
        https://github.com/RobertaMLo/CRBL_MF

        :return: result of transfer function
        """
        #mu_V, sigma_V, T_V = self.get_fluct_regime_vars(fe, fi, fe_ext, fi_ext, W, self.Q_e, self.tau_e, self.E_e,
        #                                                   self.Q_i, self.tau_i, self.E_i,
        #                                                   self.g_L, self.C_m, E_L, self.N_tot,
        #                                                   self.p_connect_e,self.p_connect_i, self.g,self.K_ext_e,self.K_ext_i)



        mu_V, sigma_V, T_V, muGn = self.get_fluct_regime_vars(Fe, Fi, Fe_ext, Fi_ext, W, Q_e, tau_e, E_e, Q_i,
                                                                        tau_i, E_i, g_L, C_m, E_L, Ke, Ki,
                                                                        K_ext_e=0, K_ext_i = 0)

        V_thre = self.threshold_func(mu_V, sigma_V, T_V, muGn,
                                     P[0], P[1], P[2], P[3], P[4])
        V_thre *= 1e3  # the threshold need to be in mv and not in Volt #OK LASCIATA IN VOLT
        f_out = self.estimate_firing_rate(mu_V, sigma_V, T_V, V_thre, g_L, C_m, alpha)
        return f_out


    def TF_goc(self, Fe, Fi, Fe_ext, Fi_ext, W, P, Qe_gr, Qi, Te_gr,
               Ti, Ee, Ei, Gl, Cm, El, Ke_grc, Ki, Qe_ext, Te_ext, Ke_ext, alpha, Ki_ext=0):
        """
        3D transfer functions
        https://github.com/RobertaMLo/CRBL_MF
        :return: result of transfer function
        """
        #mu_V, sigma_V, T_V = self.get_fluct_regime_vars(fe, fi, fe_ext, fi_ext, W, self.Q_e, self.tau_e, self.E_e,
        #                                                   self.Q_i, self.tau_i, self.E_i,
        #                                                   self.g_L, self.C_m, E_L, self.N_tot,
        #                                                   self.p_connect_e,self.p_connect_i, self.g,self.K_ext_e,self.K_ext_i)

        ### TO CHECK CHE ABBIA SENSO CORRISPONDENZA CON INPUT FUNCTION!
        #(fe, fi, fe_ext, fi_ext, W, self.P_goc, self.Q_grc_goc, self.Q_goc_goc, self.tau_grc_goc, self.tau_goc_goc,
        #self.E_i, self.E_i, self.g_L_goc, self.C_m_goc, self.E_L_goc, self.K_grc_goc, self.K_goc_goc,
        #self.Q_mf_goc, self.tau_mf_goc, self.K_mossy_goc)

        #Fe, Fi, Fe_ext, Fi_ext, XX, Qe_g, Te_g, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke_g, Ki, Ke_ext, Qe_ext, Te_ext, Ki_ext = 0

        mu_V, sigma_V, T_V, muGn  = self.get_fluct_regime_vars_goc(Fe, Fi, Fe_ext, Fi_ext, W, Qe_gr, Te_gr, Ee, Qi, Ti, Ei,
                                                         Gl, Cm, El, Ke_grc, Ki, Ke_ext, Qe_ext, Te_ext, Ki_ext = 0)

        V_thre = self.threshold_func(mu_V, sigma_V, T_V, muGn,
                                     P[0], P[1], P[2], P[3], P[4])
        V_thre *= 1e3  # the threshold need to be in mv and not in Volt
        f_out = self.estimate_firing_rate(mu_V, sigma_V, T_V, V_thre, Gl, Cm, alpha)
        return f_out


    @staticmethod
    @jit(nopython=True,cache=True)
    def get_fluct_regime_vars(Fe, Fi, Fe_ext, Fi_ext, XX, Q_e, tau_e, Ee, Q_i, tau_i, Ei, Gl, Cm, El, Ke, Ki, K_ext_e=0.,
                              tau_ext_e=0., Q_ext_e=0., tau_ext_i=0., Q_ext_i=0., K_ext_i = 0.):
    
        #Ke e Ki synaptic convergence. K_ext_e = exc. external input to GrC and GoC. Ke_ext_i = 0.
        """
        Compute the mean characteristic of neurons.
        Repository :
        https://github.com/RobertaMLo/CRBL_MF

        :return: mean and variance of membrane voltage of neurons and autocorrelation time constant
        """
        # firing rate
        # 1e-6 represent spontaneous release of synaptic neurotransmitter or some intrinsic currents of neurons
        fe = (Fe+1.0e-6) + Fe_ext
        fi = (Fi+1.0e-6) + Fi_ext

        # conductance fluctuation and effective membrane time constant
        # # stantard
        # mu_Ge, mu_Gi = Q_e*tau_e*fe*Ke + Q_ext_e*tau_ext_e*Fe_ext*K_ext_e, Q_i*tau_i*fi*Ki + Q_ext_i*tau_ext_i*Fi_ext*K_ext_i

        # # to include parallel:
        mu_Ge, mu_Gi = Q_e*tau_e*fe*Ke + Q_e*tau_e*Fe_ext*Ke, Q_i*tau_i*fi*Ki + Q_ext_i*tau_ext_i*Fi_ext*K_ext_i

        mu_G = Gl+mu_Ge+mu_Gi

        # membrane potential
        mu_V = (np.e * (mu_Ge * Ee + mu_Gi * Ei + Gl * El) - XX) / mu_G
        muGn, Tm = mu_G / Gl, Cm / mu_G  # normalization

        # post-synaptic membrane potential event s around muV
        Ue, Ui = Q_e / mu_G * (Ee - mu_V), Q_i / mu_G * (Ei - mu_V)  # EQUAL TO EXP

        # Standard deviation of the fluctuations
        # Eqns 8 from [MV_2018]
        sVe = (2 * Tm + tau_e) * ((np.e * Ue * tau_e) / (2 * (tau_e + Tm))) ** 2 * Ke * fe
        sVi = (2 * Tm + tau_i) * ((np.e * Ui * tau_i) / (2 * (tau_i + Tm))) ** 2 * Ki * fi
        sigma_V = np.sqrt(sVe + sVi) #THIS IS MY SIGMA_V

        fe, fi = fe + 1e-9, fi + 1e-9  # just to insure a non zero division


        # Autocorrelation-time of the fluctuations Eqns 9 from [MV_2018]
        Tv_num = Ke * fe * Ue ** 2 * tau_e ** 2 * np.e ** 2 + Ki * fi * Ui ** 2 * tau_i ** 2 * np.e ** 2
        Tv = 0.5 * Tv_num / ((sigma_V + 1e-20) ** 2)

        T_V = Tv * Gl / Cm  # normalization. THIS IS MY TVN

        return mu_V, sigma_V, T_V, muGn

    @staticmethod
    @jit(nopython=True,cache=True)

    def get_fluct_regime_vars_goc(Fe, Fi, Fe_ext, Fi_ext, XX, Qe_g, Te_g, Ee, Qi, Ti, Ei, Gl, Cm, El, Ke_g, Ki, Ke_ext, Qe_ext, Te_ext, Ki_ext = 0):

        """
        Compute the mean characteristic of neurons.
        Repository :
        https://github.com/RobertaMLo/CRBL_MF

        :return: mean and variance of membrane voltage of neurons and autocorrelation time constant
        """
        # firing rate
        # 1e-6 represent spontaneous release of synaptic neurotransmitter or some intrinsic currents of neurons
        fe_g = Fe + 1e-6
        fe_m = Fe_ext
        fi = Fi+1e-6 + Fi_ext

        # conductance fluctuation and effective membrane time constant
        # ---------------------------- Pop cond:  mu GrC and MLI ---------------------------------------
        muGe_g, muGe_m, muGi = Qe_g * Ke_g * Te_g * fe_g, Qe_ext * Ke_ext * Te_ext * fe_m, Qi * Ki * Ti * fi #EQUAL TO EXP
        # ---------------------------- Input cond:  mu PC -----------------------------------------------
        muG = Gl + muGe_g + muGe_m + muGi #EQUAL TO EXP
        # ---------------------------- Membrane Fluctuation Properties ----------------------------------
        mu_V = (np.e * (muGe_g * Ee + muGe_m * Ee + muGi * Ei + Gl * El) - XX) / muG  # XX = adaptation

        muGn, Tm = muG / Gl, Cm / muG  # normalization

        Ue_g, Ue_m, Ui = Qe_g / muG * (Ee - mu_V), Qe_ext / muG * (Ee - mu_V), Qi / muG * (Ei - mu_V) #EQUAL TO EXP

        sVe_g = (2 * Tm + Te_g) * ((np.e * Ue_g * Te_g)/ (2 * (Te_g + Tm))) ** 2 * Ke_g * fe_g
        sVe_m = (2 * Tm + Te_ext) * ((np.e * Ue_m * Te_ext) / (2 * (Te_ext + Tm))) ** 2 * Ke_ext * fe_m
        sVi = (2 * Tm + Ti) * ((np.e * Ui * Ti) / (2* (Ti + Tm))) ** 2 * Ki * fi

        sigma_V = np.sqrt(sVe_g + sVe_m + sVi) #SV IN MIEI CODICI

        fe_m, fe_g, fi = fe_m + 1e-15, fe_g + 1e-15, fi + 1e-15  # just to insure a non zero division

        Tv_num= Ke_g * fe_g * Ue_g ** 2 * Te_g ** 2 * np.e ** 2 + \
            Ke_ext * fe_m * Ue_m ** 2 * Te_ext ** 2 * np.e ** 2 + \
            Ki * fi * Ui ** 2 * Ti ** 2 * np.e ** 2
        Tv = 0.5 * Tv_num / ((sigma_V+1e-20) ** 2)

        T_V = Tv * Gl / Cm  # normalization TVN IN MIEI CODICI

        return mu_V, sigma_V, T_V, muGn

    @staticmethod
    @jit(nopython=True,cache=True) #DA CAPIRE COME UAA NUMBA
    def threshold_func(muV, sigmaV, TvN, muGn, P0, P1, P2, P3, P4):
        """
        The threshold function of the neurons
        :param muV: mean of membrane voltage
        :param sigmaV: variance of membrane voltage
        :param TvN: autocorrelation time constant
        :param P: Fitted coefficients of the transfer functions
        :return: threshold of neurons
        """
        # Normalization factors page 48 after the equation 4 from [ZD_2018]
        muV0, DmuV0 = -60.0, 10.0
        sV0, DsV0 = 4.0, 6.0
        TvN0, DTvN0 = 0.5, 1.
        V = (muV-muV0)/DmuV0
        S = (sigmaV-sV0)/DsV0
        T = (TvN-TvN0)/DTvN0

        return P0 + P1*V + P2*S + P3*T + P4*np.log(muGn)

    @staticmethod
    def estimate_firing_rate(muV, sV, TvN, Vthre, Gl, Cm, alpha):
        """
        The threshold function of the neurons
        :param muV: mean of membrane voltage
        :param sigmaV: variance of membrane voltage
        :param Tv: autocorrelation time constant
        :param Vthre:threshold of neurons
        """
        # Eqns 10 from [MV_2018]
        return .5 / TvN * Gl / Cm * (sp_spec.erfc((Vthre - muV) / np.sqrt(2) / sV)) * alpha
