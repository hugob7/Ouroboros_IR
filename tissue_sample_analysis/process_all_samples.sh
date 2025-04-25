#!/bin/bash

# Process all samples A - D
for sample_id in A B C D; do
    ./process_sample.sh ${sample_id}
    if [ $? -ne 0 ]; then
        echo "Error processing sample ${sample_id}"
        exit 1
    fi
done

echo "Processed all samples successfully"
