## Getting Started with Distributed AI Federated DataOps API
​
Smart data remediation and curation with Federated DataOps API.

## Overview
Data generated at distributed network by people and machines create many challenges.  Data at various geographic locations may have different quality and provenance.  In order to gain insights from this massive amount of data, data scientists and data engineers spend enormous amount of time to cleanse, understand and curate, self-managing without requiring manual intervention and prepares the data ready for AI pipeline. 
​
Data Quality for AI (DQAI) API and Federated DataOps API solves this problem.
​
Data Quality for AI performs the following:
a)	Quantifies data quality
b)	Remediates data with explanation 
​
Federated DataOps extend DQAI with the following enhanced capabilities described in the next section focused on solving distributed data issues.

​
Federated DataOps API extends the data assessment and remediation functions found in the Data Quality for AI with data and label remediation for tabular data sets with text data. Each function provides a quality score and a an explanation towards remediation
These metrics quantify data issues as a score between 0 and 1, where 1 indicates no issues were detected. Currently, these metrics are for text tabular datasets and accepts the input in the form of a comma.    
​
Version 1.0 of the Federated Data Ops API is concerned with accessing data quality for a single distributed site. Future iterations planned for version 2 of the library are to enable federated distributed data quality assessment, such as schema reconciliation methods like column mapping and label standardization.
​
​
## Version 1.0 API
The following are API services are offered in version 1.0.
​

Methods for data/label management:
1. Generate NLP Model: models and creates data utilities
2. Data Imputation: remediates to impute null column values
3. Data Validation: determines if data input is out of vocabulary 
4. Data Noise: identifies data inputted in the incorrect column

### 1.	Generate NLP Model
#### Description:  
Model and Data Utility creation: Creates the following data utilities for the data management APIs 
    1. Fasttext model (fasttext file) - used to validate and impute data during assementment
    2. Cooccurrence matrix (pickle file) - used to impute data based on statistics of the test set
    3. data/column mapping (pickle file) - used to validate data is correctly remediated in the appropriate column
Creates a fastext model to learn the relationship between column values in the data set to validate or impute data.
#### Arguments: 
```
data_file: csv file to assess data quality/train NLP model
```	
#### Return: 
```
200: Returns success message
500: Returns error message logged by server
```


### 2.	Data Imputation
#### Description:  
Performs a column level remediation to impute null values
#### Arguments: 
```
data_file: csv file to assess data quality/train NLP model
model: keyed vectors of fasttext model
matrix: pickled file of co-occurance matrix derived training data set
dictionary: dictionary of data/column map derived from training data
```	
#### Return: 
```
200: Returns success message
500: Returns error message logged by server
```
### 3.	Data Validation
#### Description:  
Determines if data input is out of vocabulary (i.e. not present in the data set)
#### Arguments: 
```
data_file: csv file to assess data quality/train NLP model
model: keyed vectors of fasttext model
```	
#### Return: 
```
200: Returns success message
500: Returns error message logged by server
```
### 4.	Data noise
#### Description:  
Determines if data input is out of vocabulary (i.e. not present in the data set)
#### Arguments: 
```
data_file: csv file to assess data quality/train NLP model
model: keyed vectors of fasttext model
```	
#### Return: 
```
200: Returns success message
500: Returns error message logged by server
```

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
​
# Prerequisite
Before proceeding to the follow the steps of the notebook, ensure you are already subscribed for a trial of the APIHub to obtain the Client ID and Client Secret to authorize API request. 
You need to provide these keys in the header of your request - like so...
```python
import requests 
headers = {
    'x-ibm-client-id': 'REPLACE_THIS_KEY',
    'x-ibm-client-secret': 'REPLACE_THIS_KEY'
}
data_file='PATH_TO_FILE'
response = requests.post(' POST_Request_URL', headers=headers, files=data_file)
print(response.json())
```
​
​
​
# Invoke REST service endpoint to create utilities for data evaluation
Retrieve the REST endpoint (IP and Port) as reported when the REST server was started and invoke the data imputation service.
​
The data service accepts multipart/formdata requests with the following arguments.
​
- data_file: path to data file to use for corpus. Type=file
​
​
```python
url = 'URL_for_nlp_model_generation'
data_file = [('data_file', ('corpus.csv', open('corpus.csv', 'rb'), 'file/csv'))]
r = requests.post(url, headers=headers, files=data_file, verify=False)
```
​
# Check response code
​
​
```python
r
```
​
# Download the zip file returned from API call, containing utilities for the data management methods
1. Fasttext model (fasttext file) - used to validate and impute data during assementment
2. Cooccurrence matrix (pickle file) - used to impute data based on statistics of the test set
3. data/column mapping (pickle file) - used to validate data is correctly remediated in the appropriate column
​
​
```python
with open('dataset_utils.zip', 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
```
​
## Before proceeding, unzip the dataset_utils.zip file downloaded in the previous cell
​
​
```python
​
```
​
# Data Imputation API Example
The data imputation API evauluates a data set and provides data values for which to impute missing column values by using either a cooccurrence matrix to impute values based on the stastical likelihood given by the column neighbors, or by using n surrounding neighbors to predict the null value.
​
To test the data imputation api, we need to first modify the training set for example purposes, as the Drug Review Dataset does not contain missing values. Before proceeding, unzip the dataset_utils zip file returned from the previous cell.
* Load the training set of the Drug Review Dataset 
* Add null values to the condition column, where "xanax" is the drug in the urlDrugName column for example purposes
* Use these columns to build a subset csv for this example (API exects csv as the format for the data file)
​
​
​
```python
df_train = pd.read_csv('drugLibTrain_raw.csv', encoding='utf8')
df_train = df_train[["urlDrugName","condition"]]
​
```
​
​
```python
df_train.loc[df_train['urlDrugName'] == "xanax", 'condition'] = ''
df_train.to_csv('imputation_example.csv', encoding='utf-8', index=False)
```
​
# Invoke REST service endpoint
Retrieve the REST endpoint (IP and Port) as reported when the REST server was started and invoke the data imputation service.
​
The data service accepts multipart/formdata requests with the following arguments.
​
- data_file: path to data file to evaluate. Type=file
- model: path to saved kevyed vectors of fasttext model used to predict. Type=file
- matrix: path to pickled file of cooccurrence matrix used to determine likelihood of value to impute. Type=file
- dictionary: path to picked file of data/column map to ensure imputed value is appropriate. Type=file
​
​
```python
url = 'URL_for_data_imputation'
​
multiple_files = [
    ('data_file', ('imputation_example.csv', open('imputation_example.csv', 'rb'), 'file/csv')),
    ('model', ('dataset_utils/fasttext.kv', open('dataset_utils/fasttext.kv', 'rb'), 'file/kv')),
    ('matrix', ('dataset_utils/fasttext_matrix.pkl', open('dataset_utils/fasttext_matrix.pkl', 'rb'), 'file/pickle')),
    ('dictionary', ('dataset_utils/label_col_map.pkl', open('dataset_utils/label_col_map.pkl', 'rb'), 'file/pickle'))]
​
r = requests.post(url, headers=headers, files=data_file, verify=False)
```
​
# Check response code
​
​
```python
r
```
​
# Retrive results of data quality score and remediation recommendations  
In this example, score returned from server is the number of null values in data set compared to the number of complete values. Remediation shows the row number and column with the missing value, followed by the percentages of the data values that should be used for imputation
​
​
```python
r.text
```
​
# Data Validation Example 
​
The data validation API evauluates a data set and determines if there is data that is out of vocabulary (OOV), i.e. was not used to train the fasttext model. This could be because there was not enough samples to be represented, or is mispelled from it's original representation. This service identifies those OOV values and uses the fasttext model's keyed vectors to find the most similar in-vocabulary data value to replace the OOV value with. 
​
To test the data validation api, we need to use a subset of the training set of the Drug Review Dataset, and use the categorical text columns - as they are the values most likely to be invalid.
* Load the training set of the Drug Review Dataset 
* Use the 'effectiveness','sideEffects', and 'condition' categorical columns respectively, to build a subset csv for this example (API exects csv as the format for the data file)
​
​
```python
df_train = pd.read_csv('drugLibTrain_raw.csv', encoding='utf8')
df_train = df_train[["effectiveness","sideEffects","condition"]]
```
​
​
```python
df_train.to_csv('data_validation_example.csv', encoding='utf-8', index=False)
```
​
# Invoke REST service endpoint
Retrieve the REST endpoint (IP and Port) as reported when the REST server was started and invoke the data validation service.
​
The data service accepts multipart/formdata requests with the following arguments.
​
- data_file: path to data file to evaluate. Type=file
- model: path to saved kevyed vectors of fasttext model used to predict. Type=file
​
​
```python
url = 'URL_for_data_validation'
​
multiple_files = [
    ('data_file', ('data_validation_example.csv', open('data_validation_example.csv', 'rb'), 'file/csv')),
    ('model', ('dataset_utils/fasttext.kv', open('dataset_utils/fasttext.kv', 'rb'), 'file/kv'))]
​
r = requests.post(url, headers=headers, files=data_file, verify=False)
```
​
# Check response code
​
​
```python
r
```
​
# Retrive results of data quality score and remediation recommendations  
In this example, score returned from server is the number of OOV values in data set compared to the number of in-vocab data. Remediation shows the row number and column with the missing value, followed by the most similar in vocabulary substitutions. 
​
​
```python
r.text
```
​
# Data Noise Example
​
The data noise API evauluates a data set and determines if there is data that is in the wrong column - likely due to manual input error. 
​
To test the data imputation api, we need to first modify the training set for example purposes, as the Drug Review Dataset does not contain noisy column values.
* Load the training set of the Drug Review Dataset 
* Change the values in the condition column, where "adhd" is the condition, and substitute it as "Mild Side Effects" for example purposes
* Build a subset csv for this example (API exects csv as the format for the data file)
​
​
```python
df_train = pd.read_csv('drugLibTrain_raw.csv', encoding='utf8')
df_train = df_train[["urlDrugName","effectiveness","sideEffects","condition"]]
```
​
​
```python
df_train.loc[df_train['condition'] == "adhd", 'condition'] = 'Mild Side Effects'
df_train.to_csv...
