# *cMF-TVB*
Integration of the cerebellar mean field model into "The Virtual Brain" (TVB) neuroinformatic platform as the first example of:
1) Region-specificity
2) Multi-model simulation


- **multimf_ww**: Simulations using CRBL MF + WW, allowing cerebellar closed-loop or whole-brain simulations.
- **crbl**: Simulations using only the cerebellar cortex (open-loop configuration).

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
- The **CRBL mean-field model** equations are from *Lorenzi et al., 2023*.
- The **construction pipeline** is inspired by *Goldman et al., 2023*.

## Running Simulations
Ensure that all files are placed in the appropriate directories within your TVB installation. Then, configure and execute simulations according to your experimental setup.

For additional details, please contact robertamaria.lorenzi01@universitadipavia.it.
