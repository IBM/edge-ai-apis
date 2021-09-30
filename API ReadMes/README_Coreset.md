
# Edge AI Coreset API 

In edge computing, there are many applications where data collected from edge locations needs to be transmitted to a central location for processing (e.g., model training, federated inferencing), so that knowledge obtained at the edges can be combined to produce a global view. As the amount of data collected at the edges can be huge, data compression is needed to efficiently deliver different types of data over networks due to transmission cost and latency without losing data fidelity.

Coreset API is designed for this purpose and implements a set of compression algorithms with the primary focus on the creation of AI models when collecting training data from edge locations. 

Coreset API has implemented the following algorithms.

1.	Lossless compression which removes redundant information from dataset to achieve compression. It has no change in model fidelity, but data reduction is not significant.

2.	Feature extraction which uses domain specific feature extractors to significantly reduce the amount of data. It has high compression but no change in model fidelity.

3.	Lossy compression which extracts features from frequency/temporal domain and then encode them. Typical examples are standard compression algorithms such as MP3, JPEG and MP4.

4.	Advanced algorithms

    a.	Principle component analysis which computes the principal components and use them to perform a change of basis on the data

    b.	Clustering-based approach which characterizes data with centroids and sample distributions

    c.	Autoencoder which is an artificial neural network that learns the distribution inherent in the data and recreates statistically similar data at the core location
    d.	MFCC extraction from sound data which can be used for audio analysis.



## Version 1.0 APIs
The following are API services offered for version 1.0:

Methods for data/label management:
- Model and Data Utility creation: Creates the following data utilities for the data management APIs 
    1. Fasttext model (fasttext file) - used to validate and impute data during assementment
    2. Cooccurrence matrix (pickle file) - used to impute data based on statistics of the test set
    3. data/column mapping (pickle file) - used to validate data is correctly remediated in the appropriate column
Creates a fastext model to learn the relationship between column values in the data set to validate or impute data.
- Data Validation: Determines if data input is out of vocabulary (i.e. not present in the data set)
- Data Imputation: Offers remediation to impute null column values
- Data Noise: Identifies data inputted in the incorrect column


## Interpreting the Output
 
The output JSON can be got from the response field. The output in the response field is formatted as JSON with a defined schema. To interpret the output JSON, there are two main fields as metadata field and the results field. The metadata field stores metadata information about the job submitted as dataset name, name of the metric, run time information and the label column, which is provided by the user.

The results field contains the actual results. One can start interpreting the results field by looking at the score, which provides a value of 0-1. A score of 1 indicates a problem free dataset. The explanation and details filed help you understand why the quality is low, by pointing to regions of data which are problematic. The details field helps you understand how to access the problematic data points. Below is a schematic that gives a one liner for each field in the JSON.

#### Data Quality JSON
```
{
    "results": { // Consist results related details for data quality metric.
    
        "title": "", // Title of performed data quality metric.
        
        "score": 1, // Data quality score for given metric as a real number between 0 and 1.\
                    Score 1 indicate for good quality and 0 indicate for bad quality data.
                
        "explanation": "", // Simple explanation on why data quality score is high or low.  
        
        "details": {} // All the details on data quality metric to access the problematic data samples

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
