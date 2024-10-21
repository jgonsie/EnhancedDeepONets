#!/bin/bash

# Define the parameter array
#params=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 )
#params=(0.{1..9} 1.{0..9} 2.{0..5})
params=(0.5 1.0)
DT=(0.1 0.01)
# Define the template case directory and destination directory
template_dir="./baseCaseSteadyState"
destination_dir="./training_data"
N_proc=5

# Copy the template case to the destination directory for each parameter value
for param_DT in "${DT[@]}"; do
	for param_x in "${params[@]}"; do
		for param_y in "${params[@]}"; do
			(	# Create a new directory for each parameter value
				param_dir="$destination_dir/DT_$param_DT-U_$param_x-$param_y"
				mkdir -p "$param_dir"
				
				# Copy the contents of the template case to the new directory
				cp -r "$template_dir"/* "$param_dir"
				
				# Modify the file with the parameter value
				sed -i "s/internalField   uniform (1 1 0)/internalField   uniform ($param_x $param_y 0)/g" "$param_dir/0/U"
				sed -i "s/volScalarFieldValue DT 0.1/volScalarFieldValue DT $param_DT/g" "$param_dir/system/setFieldsDict"
				
				# Navigate to the case directory
				cd "$param_dir" || exit
				
				# Run the simulation (assuming you have a script or command to run the simulation)
				echo "Running simulation for parameter value: DT = $param_DT; U =($param_x,$param_y)"
				# Run your OpenFOAM simulation command here
				setFields >> log.setFields
				scalarTransportHetADFoam >> log.Sim
				postProcess -func "grad(T)" >> log.Post
				
				# Once the simulation is complete, navigate back to the original directory
				cd - || exit
			) &
			
			# allow to execute up to $N_proc jobs in parallel
			if [[ $(jobs -r -p | wc -l) -ge $N_proc ]]; then
				# now there are $N jobs already running, so wait here for any job
				# to be finished so there is a place to start next one.
			wait -n
			fi
			
			done
	done
done

wait

echo "Generation of data completed!"