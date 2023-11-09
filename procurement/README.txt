[FILE LIST]

Main program for the experiments:

- procure_simulator.py, to run the experiments with a specific method on a set 
                        or a pair of sets of scenarios (one is for modelling, 
                        the other is for testing on). Use option --help for
                        more details of these, or check sbatch* files to see
                        examples how this program is used

Linear models used by the methods:

- procure_model_det.py, deterministic model
- procure_model_stoch.py, stochastic model
- procure_model_stochcvar.py, stochastic model with CVaR on the shortage
- procure_model_stochcvarw.py, stochastic model with CVaR on the waste
- procure_model_affine.py, robust adjustable model with affine rules

Utility function:

- procure_plotting.py, to produce the plots from the output files
- procure_tabling.py, to produce the tables from the output files
- utils.py, utility functions and structures

General modelling (not used by the simulator, but useful to know this is possible):

- procure_model_cvar.py, stochastic model with CVaR set on all objectives
- procure_model_affinecvar.py, robust adjustable model with CVaR objectives

SLURM batch files to submit the experiments to the HPC cluster:

- sbatch.sh, to run the experiments with simconfig.json
- sbatch_n13r0.sh, to run the experiments with simconfign13r0.json

