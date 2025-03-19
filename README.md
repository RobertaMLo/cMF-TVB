# *cMF-TVB*
Integration of the cerebellar mean field model into "The Virtual Brain" (TVB) neuroinformatic platform as the first example of:
1) Region-specificity
2) Multi-model simulation
Detailed description and output in Lorenzi et al., 2025, BioRxiv, https://doi.org/10.1101/2022.11.24.517708


- **multimf_ww**: Simulations using CRBL MF + WW, allowing cerebellar closed-loop or whole-brain simulations.
- **crbl**: Simulations using only the cerebellar cortex (open-loop configuration).
- **run_sim_mmTVB**: pipeline to run the multimf_ww codes - closed loop simulation (whole-brain activity)
- **run_sim_mmTVB_parallel_cerebellum** : pipeline to run the crbl codes - open loop simulation (cerebellar activity only)
- **tool** : tools for the simulations pipeline

## File Organization

- **Parameter files** (`*_params`): Should be placed in:
  ```
  <TVB_folder>/tvb_model_reference/simulation_file/parameter
  ```
- **Model equations and TVB interface codes**: Should be saved in:
  ```
  <TVB_folder>/tvb_model_reference/src
  ```

## References
- **Full pipeline** and results in *Lorenzi et al., 2025, BioRxiv https://doi.org/10.1101/2022.11.24.517708*
- **CRBL mean-field model** equations are from *Lorenzi et al., 2023, PlosCompBio https://doi.org/10.1371/journal.pcbi.1011434*.
- **construction pipeline** is inspired by *Goldman et al., 2023, FrontCompNeurosci https://doi.org/10.3389/fncom.2022.1058957*.

## Running Simulations
Ensure that all files are placed in the appropriate directories within your TVB installation. Then, configure and execute simulations according to your experimental setup.
Example input data: data103818 - Strucutral connectivity matrices computed for subject ID 103818 of Human Connectome Project

For additional details, please contact robertamaria.lorenzi01@universitadipavia.it.
