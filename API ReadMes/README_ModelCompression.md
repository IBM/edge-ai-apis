# Edge AI Model Compression API Overview

In edge computing, there are many applications where edge devices are typically resource constrained e.g. power, memory, storage, compute or network.  
As deep learning and machine learning models become more complex with time, inferencing at the edge of network requires model fine tuning.  
There are two methods in fine tuning models. 

## 1. Model pruning: 
   Model size reduction – pruning least important weights while keeping the most important neurons

## 2. Model quantization – 
  •	Models are usually represented as multi-dimensional arrays of 32-bit and 64-bit floats 
  •	Reduce granularity to 16-bit floats, or 8-bit integers; Go as far as 4-bit and even 1-bit
  •	Candidates for quantization – Weights; Biases (though not recommended); Activations/Outputs
  •	Tradeoff between Accuracy, Storage, Space, Memory
