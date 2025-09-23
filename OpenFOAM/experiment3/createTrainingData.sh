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
read_dat_to_array "exp3_randomDT13.dat" DT1
read_dat_to_array "exp3_randomDT23.dat" DT2
read_dat_to_array "exp3_randomV3.dat" V
items=${#DT1[@]}


# Define the template case directory and destination directory
template_dir="./baseCaseSteadyState"
destination_dir="./training_data"
N_proc=8

# Copy the template case to the destination directory for each parameter value
for ((i=0; i<items; i++)); do
			(	# Access elements from each array using current index "i"
				param_DT1="${DT1[i]}"
				param_DT2="${DT2[i]}"
				param_V="${V[i]}"
			
				# Create a new directory for each parameter value
				param_dir="$destination_dir/DT1_$param_DT1-DT2_$param_DT2-V_$param_V"
				mkdir -p "$param_dir"
				
				# Copy the contents of the template case to the new directory
				cp -r "$template_dir"/* "$param_dir"
				
				# Modify the file with the parameter value
				sed -i "s/regionDT (0.01 1);/regionDT ($param_DT1 $param_DT2);/g" "$param_dir/constant/regionDTDict"
				sed -i "s/Uparam (1);/Uparam ($param_V);/g" "$param_dir/constant/regionDTDict"
				
				# Navigate to the case directory
				cd "$param_dir" || exit
				
				# Run the simulation (assuming you have a script or command to run the simulation)
				echo "Running simulation for parameter value: DT1 = $param_DT1, DT2 = $param_DT2, V = $param_V"
				# Run your OpenFOAM simulation command here
				scalarTransportHetUparADFoam >> log.Sim
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

wait

echo "Generation of data completed!"
