from pybiomart import Dataset
from pathlib import Path
import polars as pl

Homedir = str(Path(__file__).resolve().parent.parent)
path_to_file = Homedir+"/resources/common/biomart_results"

# Connect to a dataset
dataset = Dataset(
    name='hsapiens_gene_ensembl',
    host='http://www.ensembl.org'
)

# Query data
df = dataset.query(
    attributes=[
        'external_gene_name',
        'go_id',
        'name_1006'
    ]
)



# Save to CSV
df.to_csv(path_to_file, index=False)

polars_df = pl.read_csv(path_to_file,has_header=True)

rm_nans = polars_df.drop_nulls()

merged_df = rm_nans.with_columns(
    pl.concat_str(['GO term accession','GO term name'],separator="~").alias("GO")
)

final_df = merged_df.drop(['GO term accession','GO term name'])
final_df = final_df.rename({"Gene name":"geneName"})

final_df.write_parquet(path_to_file)

