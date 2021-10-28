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
The follow are a list of parameters to set the **Start Aggregator** config:

​
*fusion_algorithm*: The name of the fusion algorithm for the federation learning process. Options includes 'fedavg', 'iter_avg', and 'doc2vec'
*model_type*: The type of model used for fusion. Options includes 'keras', 'pytorch', and 'doc2vec'
*model_file*: The saved initial model to distribute to parties to train in isolation
*num_parties* : The number of nodes participating in the fusion
*rounds* : The number of fusion rounds to complete
*epochs* : The number of epochs to to train for each fusion round
*learning_rate* : The learnig rate for the parties to use for training
*optimizer* : The name of the optimizer used for training (not applicable for doc2vec). Should be the name used by the keras or pytorch libraries (ex: optim.Adam for pytorch)
​

Learn more about Model Fusion in this medium.com [blog](https://sw-ibm.medium.com/?p=df2cff3ac20d).

