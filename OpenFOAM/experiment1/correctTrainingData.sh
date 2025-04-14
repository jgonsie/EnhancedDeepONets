#!/bin/bash

# Bash implementation of linspace
linspace () {
    local start=$1
    local end=$2
    local num=$3
    local step=$(awk -v s="$start" -v e="$end" -v n="$num" 'BEGIN{if (n>1) print (e - s) / (n - 1); else print 0}')
    local result=()
    for ((i = 0; i < num; i++)); do
        val=$(awk -v s="$start" -v st="$step" -v i="$i" 'BEGIN{printf "%.4f", s + i * st}')
        result+=("$val")
    done
    echo "${result[@]}"   # You can also return as a global array if needed
}

# Read the parameters from a file .dat
read_dat_to_array() {
    local file="$1"
    local -n output_array="$2"  # Pass by reference (Bash 4.3+)

    output_array=()  # Initialize array

    while read -r line; do
        for val in $line; do
            output_array+=("$val")
        done
    done < "$file"
}

# Define the parameter array
params=(1.0)
read_dat_to_array "exp1_corr_randomDT.dat" DT
read_dat_to_array "exp1_corr_dirs.dat" corr_dirs


# Define the template case directory and destination directory
template_dir="./baseCaseSteadyState"
destination_dir="./training_data"
N_proc=8

# Delete failed simulations
for dir in "${corr_dirs[@]}"; do
	(	# Remove directories of failed simulations
		param_dir="$destination_dir/$dir"
		rm -r "$param_dir" 
		echo "Deleting simulation $dir)"
	
	)
done

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
				sed -i "s/DT              0/DT              $param_DT/g" "$param_dir/constant/transportProperties"
				
				# Navigate to the case directory
				cd "$param_dir" || exit
				
				# Run the simulation (assuming you have a script or command to run the simulation)
				echo "Running simulation for parameter value: DT = $param_DT; U =($param_x,$param_y)"
				# Run your OpenFOAM simulation command here
				#setFields >> log.setFields
				scalarTransportADFoam >> log.Sim
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
