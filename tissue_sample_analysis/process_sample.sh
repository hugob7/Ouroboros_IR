#!/bin/bash

run_step() {
    local step_num="$1"
    local step_desc="$2"
    local script_name="$3"
    
    echo "Step ${step_num}: ${step_desc}"
    python ${script_name}.py ${sample_id}
    if [ $? -ne 0 ]; then
        echo "Error in Step ${step_num}"
        exit 1
    fi
    echo "Step ${step_num} complete"
    echo "--------------"
}

# Process single sample
sample_id=$1

if [[ ! $sample_id =~ ^[A-D]$ ]]; then
    echo "Error: Sample ID must be A, B, C, or D"
    exit 1
fi

echo "Processing sample ${sample_id}"
echo "=========================="

run_step 1 "Creating IR Matrix" "ir_processing"
run_step 2 "Performing Registration" "registration"

# Not using K-means clustering currently
# run_step 3 "Performing K-means clustering..." "kmeans_clustering"

run_step 3 "Extracting Patches" "patch_extraction"

echo "Sample ${sample_id} processing complete"
echo "=========================="
