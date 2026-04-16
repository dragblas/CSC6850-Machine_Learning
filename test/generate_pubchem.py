"""
testing for generating pubchem dataset
"""

from test.generate_dataset import generate_dataset

generate_dataset(
    input_csv="data/pubchem/raw.csv",
    smiles_column="ConnectivitySMILES",
    output_dir="data/pubchem"
)
