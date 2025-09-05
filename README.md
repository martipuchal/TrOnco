<img src="https://www.uvic.cat/sites/default/files/logo_3linies_uvic_color.jpg" alt="UVic Logo" width="200"/> <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSKMKJOIj0vlNde2xe47xQ-u_BLfb1xJtnCbg&s" alt="VHIO Logo" width="208" align="right"/>




<img src="logos/banner.png" alt="TrOnco Logo" />

<img alt="Python" src="https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white" /> <img alt="R" src="https://img.shields.io/badge/-R-276DC3?style=flat-square&logo=r&logoColor=white" /> 
# TrOnco

TrOnco is a novel machine learning tool based on random forest, XGBoost and CNN of TensorFlow. Designed to classify translocations as oncogenic or non-oncogenic by integrating multiple omics data, including genomics, transcriptomics, or proteomics. With an easy retraining capability, TrOnco offers the option to keep updating the models.

The source code and the latest binary version of the framework are present in this repository.



## Table of Contents
- [Installation](#installation)
- [Usage/Examples](#usage)
- [Features](#features)
- [Appendix](#appendix)





## Installation

- To obtain all the code, clone the Git repository.

```bash
  git clone https://github.com/martipuchal/TrOnco.git
  cd TrOnco
```

- Obtain the Human reference genome (hg19.fa)
The Human Reference Genome is a big FASTA file, which cannot be uploaded into the Git repository; for that, we are going to download it directly from the UCSC. 
```bash
wget https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.fa.gz
gunzip hg19.fa.gz
mv hg19.fa resources/common/hg19.fa
```    
- Build the image with singularity to use TrOnco.

The whole program was built under the Python version 3.8 due to package requirements. In order to improve the user experience, we build an image with all the Python packages and some R tools to analyze the data, everything under an Ubuntu OS, using conda to merge everything to a single image.

```bash
sudo singularity build images/image_python3_8.sif images/defFile_python3_8.def
```

## Usage

All the different scripts are going to be run under the pre-built image. The main script, TrOnco.py, will run all the analyses of the different gene translocations.

```bash
singularity exec image/image_python3_8.sif \
  python main/TrOnco.py /Path/To/imput_file /Path/For/Result_file -t AVG
```

TrOnco uses tissue-specific data to analyze the different gene translocations. The two principal ways to mark the translocations are inside the file or outside the file. With a column named **tissue**, each fusion can be marked with a tissue of origin. TrOnco has the optionality to mark all fusions from a file with the same tissue of origin without the need of the addition of an extra column. Using the flag **-t**, you can mark all the fusions with the same tissue. The different tissues are:

- EPI -> Epithelial 
- HEM -> Hematological
- MES -> Mesenchymal
- END -> Endometrial
- AVG -> Not defined(Averege)

As coordinates are used to retrieve gene information, some problems can occur in the annotation of the genes. To avoid this problem, we add a flag to correct this issue (**-g**). With the addition of two supplementary columns on the input file with the gene name of the 5' gene (**geneName5**) and the 3' gene (**geneName3**), we are able to get the information from the refSeq instead of the coordinates with the gene names.

In order to select the Genome version, we add the flag (**-a**). By default TrOnco uses **hg19**, and we provide all files to perform all the analysis. New or old versions can be used. 
```
**IMPORTANT:** Additional genome version must be added manually by the user. The UCSC fasta and the refSeq file must be added on the common folder before its use.
```
The last flag is specific for the training of the different algorithms (**-v**). With this flag we change the output file. Instead of saving the analysis from TrOnco, we only save the vector used for the different algorithms to predict the probability of being DRIVER.

For more information you can use the help parameter.
```bash
singularity exec images/image_python3_8.sif python main/TrOnco.py -h
```

### Examples

Examples of the different input formats allowed are stored in the **example folder** inside of the **resources folder**. 

Example with the tissue information inside the file:
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


#### Input table columns

| Column Name | Type     | Optionality  | Description |
| :---------- | :------- | :----------- | :-----------|
| `chrom5`    | `string` | **Required** | Chormosomome of the 5' gene |
| `pos5`      | `int`    | **Required** | Position of the breakpoint of the 5' Chormosomome |
| `chrom3`    | `string` | **Required** | Chormosomome of the 3' gene |
| `pos3`      | `int`    | **Required** | Position of the breakpoint of the 3' Chormosomome |
| `tissue`    | `string` | **Optional** | Tissue of origin of the fusion |
| `strand5`   | `string` | **Required** | Strand of the 5'gene |
| `strand3`   | `string` | **Required** | Strand of the 3'gene |
| `geneName5` | `string` | **Optional** | Gene name of the 5'gene |
| `geneName3` | `string` | **Optional** | Gene name of the 3'gene |


## Features

- Vector modification: 

A vector is a combination of elements, in this case, we have a numeric Vectorcomposed by only values. This values are obtain with the analysis of the fusions. The different parameters used in this analysis are descrived in the *Appendix*. 

A clue addition to this program is the easy implementation of new features to the developed vector. By modifying the **vector_list_str** variable(schema of the vector) and the **vector** variable both on the TrOnco.py direcly modify the vector. And after a retrain of teh model the TrOnco.py script can be use normaly.

- Train algorithm:
In this project we want to facilitate the re-training of the models developed. For this we generate a script(**train_algorithm.sh**) to retrain the model with the desired data. This scripts needs to be run on the TrOnco folder.

As you can see on the help message you have to parse the path to Normal fusions file and the tumor fusiosn file

```bash
singularity run images/TrOnco_image.sif ./train_algorithm.sh [-h] [-n path/of/Normal_file] [-t path/of/Turmoral_file] [-s]
    -h  show this help text
    -n  File and path to the normal tissue fusions
    -t  File and path to the tumoral tissue fusions
    -s	Flag to not save the models
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


### Table columns used in the TrOnco analysis 


| Column Name       | Type     |  Description |
| :---------------- | :------- | :------- |
| `Fusion_id`       | `int`    | Position of the fusion on the input file |
| `chrom5`          | `string` | Chormosomome of the 5' gene |
| `pos5`            | `int`    | Position of the breakpoint of the 5' Chormosomome |
| `tissue`          | `string` | Tissue of origin of the fusion |
| `strand5`         | `int`    | Strand of the 5' gene |
| `geneName5`       | `string` | Name2 in the refSeq of the 5' gene
| `exonStarts`      | `list`   | List of positions of the exons starts of the 5' gene |
| `exonEnd`         | `list`   | List of positions of the exons end of the 5' gene |
| `cdsStart`        | `int`    | Position of the cdsStart |
| `cdsEnd`          | `int`    | Position of the cdsEnd |
| `inFrame_5`       | `string` | YES/NO if the 5' gene is in frame |
| `aapos5`          | `int`    | Number of aminoacids from 5' gene |
| `Sequence5`       | `string` | Sequence of nucleotids of the 5' gene |
| `domain_retained` | `list`   | ID of the domains retained in the 5' protein |
| `domain_broken`   | `list`   | ID of the domains broken in the 5' protein |
| `domain_lost`     | `list`   | ID of the domains lost in the 5' protein |
| `pii_retained_5`  | `list`   | List of protein names wich interact with 5' protein |
| `pii_lost_5`      | `list`   | List of protein names wich lost interaction with 5' protein |
| `name2`           | `string` | Gene name of the 5' from the refSeq annotation |
| `promData`        | `float`  | Promotor actibity of the 5' gene |
| `exprData`        | `float`  | Expression data of the 5' gene |
| `utrData`         | `float`  | Expression data of the 5' gene from utr region |
| `n_self_pii_retained` | `int`| Number of protein interactions corrected of 5' protein |
| `n_pii_retained`  | `int`    | Number of protein interactions of 5' protein |
| `n_self_pii_lost` | `int`    | Number of protein interactions corrected lost of 5' protein |
| `n_pii_lost`      | `int`    | Number of protein interactions lost of 5' protein |
| `GO`              | `list`   | List of the GO terms of the 5' gene |
| `Theme`           | `list`   | List of GO groups present in 5' gene |
| `IdTheme`         | `list`   | List of GO groups present as ID in 5'gene |
| `CTF`             | `int`    | Number of CTF in the GO groups in 5' gene |
| `G`               | `int`    | Number of G in the GO groups in 5' gene |
| `H`               | `int`    | Number of H in the GO groups in 5' gene |
| `K`               | `int`    | Number of P in the GO groups in 5' gene |
| `P`               | `int`    | Number of P in the GO groups in 5' gene |
| `TF`              | `int`    | Number of TF in the GO groups in 5' gene |
| `chrom3`          | `string` | Chormosomome of the 3' gene |
| `pos3`            | `int`    | Position of the breakpoint of the 3' Chormosomome |
| `tissue3`         | `string` | Duplicate of the tissue Column |
| `strand3`         | `int`    | Strand of the 3' gene |
| `geneName3`       | `string` | Name2 in the refSeq of the 3' gene
| `exonStarts3`     | `list`   | List of positions of the exons starts of the 3' gene |
| `exonEnd3`        | `list`   | List of positions of the exons end of the 5' gene |
| `cdsStart3`       | `int`    | Position of the cdsStart of the 3' gene|
| `cdsEnd3`         | `int`    | Position of the cdsEnd of the 3' gene|
| `inFrame_3`       | `string` | YES/NO if the 3' gene is in frame |
| `aapos3`          | `int`    | Number of aminoacids from 3' gene |
| `Sequence3`       | `string` | Sequence of nucleotids of the 3' gene |
| `domain_retained3`| `list`   | ID of the domains retained in the 3' protein |
| `domain_broken3`  | `list`   | ID of the domains broken in the 3' protein |
| `domain_lost3`    | `list`   | ID of the domains lost in the 3' protein |
| `pii_retained_3`  | `list`   | List of protein names wich interact with 3' protein |
| `pii_lost_3`      | `list`   | List of protein names wich lost interaction with 3' protein |
| `name23`          | `string` | Gene name of the 3' from the refSeq annotation |
| `promData3`       | `float`  | Promotor activity of the 3' gene |
| `exprData3`       | `float`  | Expression data of the 3' gene |
| `utrData3`        | `float`  | Expression data of the 3' gene from utr region |
| `n_self_pii_retained3`| `int`| Number of protein interactions corrected of 3' protein |
| `n_pii_retained3` | `int`    | Number of protein interactions of 3' protein |
| `n_self_pii_lost3`| `int`    | Number of protein interactions corrected lost of 3' protein |
| `n_pii_lost3`     | `int`    | Number of protein interactions lost of 3' protein |
| `GO3`             | `list`   | List of the GO terms of the 3' gene |
| `Theme3`          | `list`   | List of GO groups present in 3' gene |
| `IdTheme3`        | `list`   | List of GO groups present as ID in 3'gene |
| `CTF3`            | `int`    | Number of CTF in the GO groups in 3' gene |
| `G3`              | `int`    | Number of G in the GO groups in 3' gene |
| `H3`              | `int`    | Number of H in the GO groups in 3' gene |
| `K3`              | `int`    | Number of P in the GO groups in 3' gene |
| `P3`              | `int`    | Number of P in the GO groups in 3' gene |
| `TF3`             | `int`    | Number of TF in the GO groups in 3' gene |
| `FUSION_PROTEIN`  | `string` | Sequence of the protein result of the fusion |
| `vector`          | `list`   | Vector used to classify the fusion |
| `RF_prob`         | `float`  | Probability obtain with the Random Forest model |
| `XGB_prob`        | `float`  | Probability obtain with the XGBoost model |
| `TF_prob`         | `float`  | Probability obtain with the TensorFlow CNN model |
| `RF_Classification`  | `string` | Class obtain with the probability from the Random Forest model |
| `XGB_Classification` | `string` |  Class obtain with the probability from the XGBoost model |
| `TF_Classification` | `string` |  Class obtain with the probability from the TensorFlow CNN model |


