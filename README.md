# *cMF-TVB*

Integration of the cerebellar mean field model into "The Virtual Brain" (TVB) neuroinformatic platform as the first example of:
1) Region-specificity
2) Multi-model simulation

A detailed description of the framework and its validation can be found in:
Lorenzi et al., 2025, BioRxiv, https://doi.org/10.1101/2022.11.24.517708

---

## **Repository Content Summary**

- **multimf_ww**: Simulations using CRBL MF + WW, allowing cerebellar closed-loop or whole-brain simulations.
- **crbl**: Simulations using only the cerebellar cortex (open-loop configuration).
- **tool**: Collection of functions for performing simulations.
- **run_sim_mmTVB_parallel_cerebellum**: Pipeline to run the crbl codes - open-loop simulation (cerebellar activity only).
- **run_sim_mmTVB**: Pipeline to run the multimf_ww codes - closed-loop simulation (whole-brain activity).
- **./data**: Example input data. `data/data103818`: Structural connectivity matrices computed for subject ID 103818 from the Human Connectome Project.
- **./Adex and _adex**: Prototype of AdEx + CRBL MF TVB. This package is a **work in progress** and **not publicly available**. Please contact robertamaria.lorenzi01@universitadipavia.it if you are interested. First application: Lorenzi et al., ISMRM 2023, Congress Proceedings.

---

## **Getting Started**

### **Folder Preparation**
1) Create `<TVB_folder>`
2) Set up a Python environment and install TVB software:
   ```sh
   pip install tvb
   ```
3) Download the sample data (`./data`) and store it in `<TVB_folder>`

### **Integration of the Cerebellar Mean Field Model into TVB**
4) Save `parallel_crbl.py` in `<TVB_folder>/tvb_model_reference/src`
5) Save `parallel_tool_crbl.py` in `<TVB_folder>/tvb_model_reference/src`
6) Save `parallel_crbl_params.py` in `<TVB_folder>/tvb_model_reference/simulation_file/parameter`

### **Validation (Constructive and Predictive Analysis)**
7) Save `run_sim_mmTVB_parallel_cerebellum.py` and `tools_for_sim_mmTVB_parallel_cerebellum.py` in `<TVB_folder>` (not mandatory but recommended).
8) Save `plot_run_sim_mmTVB_parallel_cerebellum.py` and `tools_for_plot_mmTVB_parallell_cerebellum.py` in `<TVB_folder>` (not mandatory but recommended).

### **Running cMF-TVB**
Follow the same steps (4 to 8) using the cMF-TVB files:
- **Integration:**
  - (4) `parallel_crbl_multimf_ww.py`
  - (5) `parallel_tool_crbl_multimf_ww.py`
  - (6) `parallel_crbl_multimf_ww_params.py`
- **Validation:**
  - (7) `run_sim_mmTVB_parallel.py`, `tools_for_sim_mmTVB_parallel.py`
  - (8) `plot_run_sim_mmTVB_parallel.py`, `tools_for_plot_mmTVB_parallell.py`

### **Important Notes**
- Ensure all file paths in (7) and (8) are updated according to the sample data (`./data`) or your own data.
- Ensure your Python environment contains all the required packages to run files (7) and (8).
- To run TVB simulations, first activate the Python environment where TVB is installed, then execute:
   ```sh
   python run_sim_mmTVB_parallel_cerebellum.py
   python plot_run_sim_mmTVB_parallel_cerebellum.py
   ```
- For any additional details, please contact robertamaria.lorenzi01@universitadipavia.it.
  
## References
- **Full pipeline** and results in *Lorenzi et al., 2025, BioRxiv https://doi.org/10.1101/2022.11.24.517708*
- **CRBL mean-field model** equations are from *Lorenzi et al., 2023, PlosCompBio https://doi.org/10.1371/journal.pcbi.1011434*.
- **Construction pipeline** is inspired by *Goldman et al., 2023, FrontCompNeurosci https://doi.org/10.3389/fncom.2022.1058957*.
