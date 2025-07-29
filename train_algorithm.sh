#!/bin/bash


# Files:
normal="train/Normal_nogenes.csv"
tumor="train/Tumoral_nogenes.csv"


## XGBoost parameters
d=9 # max_depth
c=20	 # min_child_weight
t="hist" # tree_method


# Vector building with the negative(no-oncogenic) and positives(oncogenic) fusions.
echo "## Vector"
python "main/main.py" $normal "tmp/normal_vector" -v > "log.log"
python "main/main.py" $tumor "tmp/tumor_vector" -v -t "AVG">> "log.log"


# Build the classifier with the vectors previously build
echo "## Build" 
python "main/binari_tensor.py" -d $d -c $c -t $t >> "log.log"

# Run the evaluation of the models
echo "## Eval" 
python "main/Evaluate_models.py" >> "log.log"

# Clean the tmp files
rm tmp/*_vector
rm best_model.h5












