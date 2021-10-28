# Getting Started with Edge AI Model Management API 

Maintain the most suitable models at the edge with (1) model fingerprinting and (2) model outlier detection.

## Overview

It is common today to find many pre-trained machine learning models from model zoo, such as Tensorflow Hub, Caffe model zoo, modelzoo.co, 
and ModelDepot.io are other examples of model zoos. These models differ in the training dataset, model hyper parameters, or the task being solved. 
For a given edge environment, it is not straightforward to choose the model that may perform the best, with or without 
customization to the local environment. In addition, there may not be sufficient training data available on the target edge site to train a 
model from scratch in a collaborative environments, each party may have a different environment.

In essence, how well a model discriminates the target dataset is a good indicator of the model's accuracy, regardless of the label set, dataset, 
or hyper parameters.


## Version 1.0 APIs

The following are API services offered in version 1.0.

Methods for Model Management API.
1. Generate Fingerprint: Model fingerprint solves the problem of model selection, i.e., choosing the most suitable model from a model zoo of pre-trained models.
2. Outlier Detection: Determines if the test data is an outlier to the model based on its fingerprint.
3. Get Result: Retrieves result from completed "Generate Fingerprint" operation.



    
