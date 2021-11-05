# Getting Started with Distributed AI Model Management API 

Maintain the most suitable models at distributed site(s) with (1) model fingerprinting and (2) model outlier detection.

## Overview

It is common today to find many pre-trained machine learning models from model zoo, such as Tensorflow Hub, Caffe model zoo, modelzoo.co, 
and ModelDepot.io are other examples of model zoos. These models differ in the training dataset, model hyper parameters, or the task being solved. 
For a given distributed environment, it is not straightforward to choose the model that may perform the best, with or without 
customization to the local environment. In addition, there may not be sufficient training data available on the target distributed site to train a 
model from scratch in a collaborative environments, each party may have a different environment.

In essence, how well a model discriminates the target dataset is a good indicator of the model's accuracy, regardless of the label set, dataset, 
or hyper parameters.


## Version 1.0 APIs

The following are API services offered in version 1.0.

Methods for Model Management API.
1. Generate Fingerprint: Model fingerprint solves the problem of model selection, i.e., choosing the most suitable model from a model zoo of pre-trained models.
2. Outlier Detection: Determines if the test data is an outlier to the model based on its fingerprint.
3. Get Result: Retrieves result from completed "Generate Fingerprint" operation.

## Interpreting the Output

### 1. Generate Fingerprint
#### Descripion:
Generates fingerprint for model or dataset.
#### Arguments:
```
dataset: Pickled file of numpy array dataset used during model training.
model: aved keras model that will be used to fingerprint.  Pytorch model will be supported in future releases.
partial_activations: percentage of partial activations to retain when fingerprinting

```
### Return:
```
202: Returns task id to check status of fingerprint generation process
500: Returns error message logged by server
```

### 2. Outlier Detection 
#### Descripion:
Determines if test data is an outlier to the model based on its fingerprint.
#### Arguments:
```
fingerprint:
num_layers: number of layers in the model fingerprint
dataset:
model: model used to generate fingerprint; none if only dataset if fingerprinted
model_type: type of saved model; 'keras or pytorch'
model_def: model definition of pytorch module if model_type is 'pytorch'
activation_mapping: acknowledges whether activation mapping was used for fingerprint
percentile: the percentile for outlier scoring


```
### Return:
```
200: Returns percentage of test data samples deemed outliers
500: Returns error message logged by server
```
### 3. Get Result 
#### Descripion:
Retrieves result from completed generate fingerprint task
#### Arguments:
```
none
```
### Return:
```
200: Returns success message
500: Returns error message logged by server
```



    
