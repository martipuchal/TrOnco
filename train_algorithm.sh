#!/bin/bash

if [ "$1" == "-h" ]; then
  echo "Usage: ` $0` [Normal file path] [Tumoral file path]"
  exit 0
fi


# Files:
normal= $1 #"train/Normal_nogenes.csv" 
tumor=$2 #"train/Tumoral_nogenes.csv"


## XGBoost parameters
d=10 # max_depth
c=1	 # min_child_weight



# Vector building with the negative(no-oncogenic) and positives(oncogenic) fusions.
echo "## Vector"
python "main/Vec_gen.py" $normal "tmp/Normal_vector" > "log.log"
python "main/Vec_gen.py" $tumor "tmp/Tumor_vector" >> "log.log"


# Build the classifier with the vectors previously build
echo "## Build" 
python "main/binari_tensor.py" -d $d -c $c  >> "log.log"

# Run the evaluation of the models
echo "## Eval" 
python "main/Multy_ROC.py" >> "log.log"

# Clean the tmp files
rm tmp/*_vector












