import pandas as pd
import numpy as np

# Read feature matrix from CSV file
def get_Feature_from_csv(file_path):

    Feature_matrix = np.array(pd.read_csv(file_path)).transpose()
    
    return Feature_matrix

# Read Metadata from CSV file
def get_metadata_from_csv(file_path):

    Table_metadata = pd.read_csv(file_path)

    return Table_metadata

