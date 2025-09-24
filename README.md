# Derivative-enhanced and hybrid optimization for DeepONets
This repository contains the files to generate the data and reproduce the results presented in the paper: *"Efficient Operator Learning with Derivative-Enhanced Parameter Sensitivity Information
and Hybrid Optimization"*.

The files are organized as follows:
- **OpenFOAM**: Contains the OpenFOAM files to run the simulations needed for generating the training data.
- **SRC**: Contains the source files of the DeepONet implementation (Keras+Tensorflow).
- **parallel_training**: Contains the files employed for training the models of each experiment. The code is parallelized to run trainings with different initializations. It is optimized for a 16 Gb GPU.
- **single_training**: Contains the files to run a single training for each experiment.
- **postprocess**: Contains the files for the generation figures and for evaluating the performance of the models in each experiment.
- **generate_data.py**: Creates the realization of the random variables employed for generating the training data.

## Workflow
1. Generate the realization of the random variables by running *generate_data.py*. By default the files will generate realizations for the three experiments. You can select the desired experiment by commenting the corresponding lines.
2. Run the OpenFOAM simulation based on the realizations of the random variables. More information in the OpenFOAM folder.
3. Training of the models:
   - single_training: Run the training of a single model with certain initialization.
   - parallel_training: Run in parallel the training of different models and different initializations. The automatic GPU memory (buffer) assigment is deactivated, so the parallelization is carried out in a single GPU. The files must be adapted depending of the features of the GPU.
4. Once the models are trained, the performance comparison and the generation of figures can be performed by executing the files contained in the postprocess folder.
