# Generating the training data by running OpeFOAM simulations
For succesfully running the OpenFOAM simulations, the following steps must be followed:
1. Install the discreate adjoint version of OpenFOAM (information available [here](https://gitlab.stce.rwth-aachen.de/towara/discreteadjointopenfoam_adwrapper)).
2. Compile the OpenFOAM solvers contained in the *solvers* folder:
   - **scalarTransportFoam**: Basic implementation of the convection-diffusion solver. Constant velocity and diffusivity.
   - **scalarTransportADFoam**: Automatic differentiable vesion of scalarTransportFoam (experiment 1).
   - **scalarTransportHetFoam**: Modified implementation of the convection-diffusion solver. Constant velocity and heterogeneous diffusivity.
   - **scalarTransportHetADFoam**: Automatic differentiable vesion of scalarTransportHetFoam (experiment 2).
   - **scalarTransportHetUparFoam**: Modified implementation of the convection-diffusion solver. Parametric velocity and heterogeneous diffusivity.
   - **scalarTransportHetUparADFoam**: Automatic differentiable vesion of scalarTransportHetUparFoam (experiment 3).
3. Run the file *createTrainingData.sh*, contained in the folder of the corresponding experiment.
4. For unknown reasons, the automatic differentiable execution numerically diverges for certain parameter realizations. The solution consists of perturbating the parameters. For that purpose, execute the following files iteratively (normally for a maximum of 3 iterations):
  ```
  correct_data.py
  correctTrainingData.sh
  ```
