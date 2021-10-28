# Getting Started with Edge AI Model Fusion API 
Shared ML learning without shared data.

## Overview
As massive amount of data collected at the edge of network, and is it even feasible or cost effective to transport this data to a central location? As data privacy becomes a key resource as country like Germany with GDPR, data cannot leave the country boundary or data cannot leave the given enterprise premise. 

Model Fusion API solve this problem by on sharing learned Machine Learning models without sharing the raw data distributed across multiple edge sites insuring data privacy, autonomous model learning, and model ensemble from multiple edges. Model Fusion API provides mechanisms to create global and local models when appropriate fine-tune them.

## Version 1.0 API
The following are API services offered for version 1.0:
​
Methods for data/label management:
- Model and Data Utility creation: Creates the following data utilities for the data management APIs 
    1. Fasttext model (fasttext file) - used to validate and impute data during assementment
    2. Cooccurrence matrix (pickle file) - used to impute data based on statistics of the test set
    3. data/column mapping (pickle file) - used to validate data is correctly remediated in the appropriate column
Creates a fastext model to learn the relationship between column values in the data set to validate or impute data.
- Data Validation: Determines if data input is out of vocabulary (i.e. not present in the data set)
- Data Imputation: Offers remediation to impute null column values
- Data Noise: Identifies data inputted in the incorrect column
​
​
## Interpreting the Output

