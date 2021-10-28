# Getting Started with Edge AI Model Fusion API 
Shared ML learning without shared data.

## Overview
As massive amount of data collected at the edge of network, and is it even feasible or cost effective to transport this data to a central location? As data privacy becomes a critical issue as country like Germany with GDPR, i.e.data cannot leave the country boundary or given enterprise premise or geopolitical boundary.

Model Fusion API extends IBM FL framework to solve this edge scenario by only sharing learned Machine Learning models without sharing the raw data distributed across multiple edge sites insuring data privacy.  In addition, the API autonomously learns model and ensemble them from multiple edges. Model Fusion API provides mechanisms to create global and local models when appropriate fine-tune them.

## Version 1.0 API
The following are API services offered for version 1.0:
​
Methods for model fusion:
- Start Aggregator: Aggregator is responsible for orchestration the federated learning environment  
- Start Party: Registering a party with the aggregator before joining the federation
- Train: Initiate the training
- Stop: Ends the federated learning framework to signify end of training
- Get Weights: For edge sites to retrieve the weights training is complete

​
## Interpreting the Output

Learn more about Model Fusion in this medium.com [blog](https://sw-ibm.medium.com/?p=df2cff3ac20d).

