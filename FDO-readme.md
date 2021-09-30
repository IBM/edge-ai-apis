# Federated DataOps v1.0 [![Build Status](https://travis.ibm.com/data-readiness-for-ai/dart.svg?token=FteDhaP1ixcbGERzJ2he&branch=master)](https://travis.ibm.com/data-readiness-for-ai/dart)
`Repo for all version 1.0 APIs for Federate DataOps`
​
Federated DataOps is a library extending the data assessment and remediation functions found in the Data Assessment and Readiness Toolkit (DART) with data and label remediation for tabular data sets with text data. Each function provides a quality score and a an explanation towards remediation
These metrics quantify data issues as a score between 0 and 1, where 1 indicates no issues were detected. Currently, these metrics are for text tabular datasets and accepts the input in the form of a comma .    
​
Version 1.0 of the Federated Data Ops API is concerned with accessing data quality for a single edge site. Future iterations planned for version 2 of the library are to enable federated edge data quality assessment, such as schema reconciliation methods like column mapping and label standardization.
​
​
## Version 1.0 APIs
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
 
The output JSON can be got from the response field. The output in the response field is formatted as JSON with a defined schema. To interpret the output JSON, there are two main fields as metadata field and the results field. The metadata field stores metadata information about the job submitted as dataset name, name of the metric, run time information and the label column, which is provided by the user.
​
The results field contains the actual results. One can start interpreting the results field by looking at the score, which provides a value of 0-1. A score of 1 indicates a problem free dataset. The explanation and details filed help you understand why the quality is low, by pointing to regions of data which are problematic. The details field helps you understand how to access the problematic data points. Below is a schematic that gives a one liner for each field in the JSON.
​
#### Data Quality JSON
```
{
    "results": { // Consist results related details for data quality metric.
    
        "title": "", // Title of performed data quality metric.
        
        "score": 1, // Data quality score for given metric as a real number between 0 and 1.\
                    Score 1 indicate for good quality and 0 indicate for bad quality data.
                
        "explanation": "", // Simple explanation on why data quality score is high or low.  
        
        "details": {} // All the details on data quality metric to access the problematic data samples
​
    },
    "metadata": { // Consist metadata related information to data quality metric.
        
        "dataset_details": [ // Dataset details 
            {
                "name": "",
                "type": "",
                "path": ""
            }
        ],
        "method_details": { // Data quality metric config details 
            "name": "",
            "type": "",
            "definition": ""
        },
        "starting_timestamp": "",
        "runtime": ""
        
    }
}
```
In order to use the APIs effectively, we provide a step-by-step guide on using the Federated DataOps API to assess and remediate data quality issues with a text tabular data set. 
The tutorial is a complete Python notebook to help you get started using the API, which you can run in your preferred IDE.
