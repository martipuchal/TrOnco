
# TrOnco

TrOnco is a novel machine learning tool based on random forest. Designed to classify translocations as oncogenic or non-oncogenic by integrating multiple omics data, including genomics, transcriptomics, or proteomics.

The source code and the latest binary versionof the framework is present in this repository.



## Table of Contents
- [Installation](#installation)
- [Usage/Examples](#usage)
- [Features](#features)
- [Appendix](#appendix)





## Installation

- To obtain all the code clone the git repository

```bash
  git clone https://github.com/martipuchal/TrOnco.git
  cd TrOnco
```

- Obtain the Human reference genome (hg19.fa)
The Humann Reference Genome is a big fasta file wich can not be uploaded in to the git repository for that we are going to dowload it direcly from the UCSC. 
```bash
wget https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.fa.gz
gunzip hg19.fa.gz
mv hg19.fa resources/common/hg19.fa
```    

- Build the image with singularity to use TrOnco
The hole progrma was build under the Python version 3.8 due to packages requirements. In order to improve the user experience we build a images with all the python packages and some R tools to analyze the data everything under a Ubuntu OS using conda to merge everything to a single image.
```bash
sudo singularity build images/image_python3_8.sif images/defFile_python3_8.def
```

## Usage

All the different scrips are goint to be run under the prebuild image. The main script, TrOnco.py will run all the analyis of the different genen translocations.

```bash
singularity exec image/image_python3_8.sif \
  python main/TrOnco.py /Path/To/imput_file /Path/For/Result_file -t AVG
```

TrOnco use tissue specific data to analyses the different gene translocations. The two principal ways to mark the trasnlocations are: Inside the file or outseide the file. With a column name **tissue** each fusion can be mark with a tissue of origin. TrOnco have the optionality to mark all fusions from a file with teh same tissue of origin without the need of the additon of an extra column. Using the flag **-t** you can mark all the fusions with the same tissue. The different tissues are:

- EPI -> Epithelial 
- HEM -> Hematological
- MES -> Mesenchymal
- END -> Endometrial
- AVG -> Not defined(Averege)

As coordinates are used to retreave gene information some problems can ocur in the annotation of the genes. To avoid this problematic we add a flag to correct this issue(**-g**). With the addition of two suplementari columns on the input file with the gene name of the 5' gene(**geneName5**) and the 3' gene(**geneName3**) we are able to get the information from the refSeq insted of the coordinates with the gene names.

In order to select the Genome version we add the flag(**-a**). By default TrOnco uses **hg19** and we provide all files to perform all the analysis. New or old versions can be used. 
```
**IMPORTANT:** Additional genome version must be added manually by the user. The UCSC fasta and the refSeq file must be added on the common folder before its use.
```
The last flag is specific for the training of the different algorithms(**-v**). With this flag we change the output file. Insted of saving the analysis from Tronco we only save the vector used for the differnt algorithms to predict the problavility of being DRIVER.

For more information you can use the help parameter.
```bash
singularity exec images/image_python3_8.sif python main/TrOnco.py -h
```

### Examples

As examples of the different input formals allowed are stored in the **example folder** inside of the **resources folder**. 

Example with the tissue information insede the file:
```bash
singularity exec image/image_python3_8.sif python main/TrOnco.py \
  resources/example/example_tissue.csv example_results.csv
```
Example with the gene names on the file.
```bash
singularity exec image/image_python3_8.sif python main/TrOnco.py \
  resources/example/example_genes.csv example_results.csv -g
```
Example with only the positional coordinates on teh file and teh tissue information pased as a parameter.
```bash
singularity exec image/image_python3_8.sif python main/TrOnco.py \
  resources/example/example_pos.csv example_results.csv -t AVG
```
## Features

- Vector modification: 

A vector is a combination of elements, in this case, we have a numeric Vectorcomposed by only values. This values are obtain with the analysis of the fusions. The different parameters used in this analysis are descrived in the *Appendix*. 

A clue addition to this program is the easy implementation of new features to the developed vector. By modifying the **vector_list_str** variable(schema of the vector) and the **vector** variable both on the TrOnco.py direcly modify the vector. And after a retrain of teh model the TrOnco.py script can be use normaly.

- Train algorithm:
In this project we want to facilitate the re-training of the models developed. For this we generate a script(**train_algorithm.sh**) to retrain the model with the desired data.

Firts lines of this files are:
```bash
# Files:
normal="train/Normal_nogenes.csv"
tumor="train/Tumoral_nogenes.csv"
```
Changing this with new dataSets open the posibiliti to retrain the model with new data and generate new models. When this script is executed the old models are preplaced with the new ones. This afect the three different algorithms(**RandomForest**, **XGBoost** and **the TensorFlow CNN**)



## Appendix

### Schema of the Vecor used to perfom this analysis.
- promData_5 / promData_3 -> ChIPseq data form normal tissue of the gene involved in the fusion. This values is tissue specific.
- n_domain_retained_5 / n_domain_retained_3 -> Number of domains retained of the resulting protein of the 5' gene / 3' gene
- n_domain_broken_5 / n_domain_broken_3 -> Number of domains broken of the resulting protein of the 5' gene / 3' gene. A domain broken is a domain with the breakpoint inside it.
- n_domain_lost_5 / n_domain_lost_3 -> Number of domains lost of the resulting protein of the 5' gene / 3' gene.
- CTF_5 / CTF_3 -> Number of Gene ontology terms associated with the Transcrition co-facors of the the 5' gene / 3' gene.
- G_5 / G_3 -> Number of Gene ontology terms associated with the GTPase of the the 5' gene / 3' gene.
- H_5 / H_3 -> Number of Gene ontology terms associated with the Helicase/Histone modifiersof the the 5' gene / 3' gene.
- K_5 / K_3 -> Number of Gene ontology terms associated with the Kinase activity of the the 5' gene / 3' gene.
- P_5 / P_3 -> Number of Gene ontology terms associated with the Protein binding of the the 5' gene / 3' gene.
- TF_5 / TF_3 -> Number of Gene ontology terms associated with the Transcrition facors of the the 5' gene / 3' gene.
- exprData_5 / exprData_3 -> TPM form RNA-seq from normal tissue of the gene involved in the fusion. This values is tissue specific.
- utrData_5 / utrData_3 -> Expresion value for each gene. This values is not tissue specific.
- n_piis_retained_5 / n_piis_retained_3 -> Protein interactions retained of the resulting protein of the 5' gene / 3' gene
- n_piis_lost_5 / n_piis_lost_3 -> Protein interactions lost of the resulting protein of the 5' gene / 3' gene


