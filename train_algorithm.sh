usage="$(basename "Trai_algorithm") [-h] [-n path/of/Normal_file] [-t path/of/Turmoral_file]  -- Script to re-train the different models to perform the classification.\n
\n
where:\n
    -h  show this help text\n
    -s  File and path to the normal tissue fusions\n
    -t  File and path to the tumoral tissue fusions\n"

declare -i helpval=0
while getopts ":h:n:t:" opt; do
  case $opt in
    h) echo $usage
       exit
       ;;
    n) normal=${OPTARG}
       helpval+=1
      ;;
    t) tumor=${OPTARG}
       helpval+=1
      ;;
  esac
done

echo $helpval
if [ ${helpval} = 2 ]; then

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



else 
	echo "Wrong flags used"
	echo $usage

fi
