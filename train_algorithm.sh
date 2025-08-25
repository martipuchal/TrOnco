usage="$(basename "Trai_algorithm") [-h] [-n path/of/Normal_file] [-t path/of/Turmoral_file] [-s] -- Script to re-train the different models to perform the classification.\n
\n
where:\n
    -h  show this help text\n
    -n  File and path to the normal tissue fusions\n
    -t  File and path to the tumoral tissue fusions\n
    -s	Flag to not save the models\n"

declare -i helpval=0
save="False"
while getopts ":h:n:t:s" opt; do
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
    s) save="True"
      ;;
  esac
done


if [ ${helpval} = 2 ]; then

# Vector building with the negative(no-oncogenic) and positives(oncogenic) fusions.
echo "## Vector"
python "main/Vec_gen.py" $normal "tmp/normal_vector_train" > "log.log"
python "main/Vec_gen.py" $tumor "tmp/tumor_vector_train" >> "log.log"


# Build the classifier with the vectors previously build
# Run the evaluation of the models
echo "## Build and Eval" 
python "main/train_algorithms.py" -s $save >> "log.log"

# Generate the markdown
echo "## Report"
R -e "rmarkdown::render('main/Report.Rmd')" >>"log.log"

# Clean the tmp files
mv main/Report.html .
#rm tmp/*




else 
	echo "Wrong flags used"
	echo $usage

fi
