
# Getting Started with Distributed AI Coreset API 
Smart compression and data extraction with Coreset API.

## Overview
In `Distributed computing`, there are many applications where data collected from distributed locations needs to be transmitted to a central location for processing (e.g., model training, federated inferencing), so that knowledge obtained at the distributed environments can be combined to produce a global view. As the amount of data collected at the distributed environments can be huge, data compression is needed to efficiently deliver different types of data over networks due to transmission cost and latency without losing data fidelity.

`Coreset API` is designed for this purpose and implements a set of compression algorithms with the primary focus on the creation of AI models when collecting training data from distributed locations. 

Coreset API has implemented the following algorithms.

1.	`Lossless compression` which removes redundant information from dataset to achieve compression. It has no change in model fidelity, but data reduction is not significant.

2.	`Feature extraction` which uses domain specific feature extractors to significantly reduce the amount of data. It has high compression but no change in model fidelity.

3.	`Lossy compression` which extracts features from frequency/temporal domain and then encode them. Typical examples are standard compression algorithms such as MP3, JPEG and MP4.

4.	`Advanced algorithms`

    a.	`Principle component analysis` which computes the principal components and use them to perform a change of basis on the data

    b.	`Clustering-based` approach which characterizes data with centroids and sample distributions

    c.	`Autoencoder` which is an artificial neural network that learns the distribution inherent in the data and recreates statistically similar data at the core location
    
    d.	`MFCC (Mel Frequency Cepstral Coefficient) extraction` from sound data which can be used for audio analysis.


## Version 1.0 APIs
The following are API services are offered in version 1.0:

Methods for Coreset API.

REST API uses an HTTP post method to invoke the backend Coreset service. The payload of the post method has a multipart/form-data format in which form is used to pass arguments and file is used to pass data. 

In the following, a example is given for each method in python syntax. 

API calls have a format of 
```
res = requests.post(url='https://URL/method', data=meta_data, files=file_data) 
```
where meta_data defines the arguments passed to the algorithm and file_data specifies the dataset used for processing. The return of each call carries a status code and the processing result in a pickled format. 

The status code 200 indicates success, and the status code 400 denotes an error condition.

### Compress Methods

### 1.	jpeg_compress
#### Description:
Compress an image to JPEG image.
#### Arguments:
```
quality: An integer specifying the quality of the compressed JPEG image ranging from 0 to 100.
file: Specifies the image to be compressed.
```
#### Return:
```
200: The resulting JPEG image.
400: An error message if error occurred at runtime.
```
#### Example:
```
meta_data = {'quality': 10}
file_data = {'file': image_data}
res = requests.post(url='URL/jpeg_compress', data=meta_data, files=file_data)
```
### 2.	wav-to-mp3
#### Description:  
Converts a wav audio clip to an mp3 audio clip.
#### Arguments: 
```
rate: A string defining the compression data rate and ???128k???, ???192k??? and ???256k??? are allowed
file: Specifies the wav audio clip to be compressed.	
```	
#### Return: 
```
200: The resulting mp3 clip.
400: An error message if error occurred at runtime.
```
#### Example:
```
meta_data = {'rate': '192k'}
file_data = {'file': wav_data}
res = requests.post(url="URL/wav_to_mp3", data=meta_data, files=file_data) 
```
### 3.	compress_nparray
#### Description:
Losslessly compress the specified numpy array using the default algorithm.
#### Arguments:
```
file: The specified numpy array to be compressed. It is in a pickled format.
```
#### Return:
```
200: The compressed numpy array.
400: An error message if error occurred at runtime.
```
#### Example:
```
file_data = {'file': memfile}
res = requests.post(url="URL/compress_nparray", files=file_data)
```
### 4.	compress_nparray_zlib
#### Description:
Losslessly compress a numpy array using the zlib algorithm.
#### Arguments:
```
file: The specified numpy array to be compressed. It is in a pickled format.
```
#### Return:
```
200: the compressed numpy array.
400: an error message if error occurred at runtime.
```
#### Example:
```
file_data = {'file': memfile}
res = requests.post(url="URL/compress_nparray_zlib", files=file_data)
```
### 5.	compress_nparray_bz2
#### Description:
Losslessly compress a numpy array using the bz2 algorithm.
#### Arguments:
```
file: the specified numpy array to be compressed using the bz2 algorithm.
```
#### Return:
```
200: the compressed numpy array.
400: an error message if error occurred at runtime.
```
#### Example:
```
file_data = {'file': memfile}
res = requests.post(url="URL:port/compress_nparray_bz2", files=file_data)
```
### 6.	compress_obj
#### Description:
Losslessly compress a python object using the default compression algorithm.
#### Arguments:
```
file: the specified python object to be compressed. It is in a pickled format.
```
### Return:
```
200: the compressed python object.
400: an error message if error occurred at runtime.
```
#### Example:
```
file_data = {'file': memfile}
res = requests.post(url="/compress_nparray_obj", files=file_data)
```
### 7.	clustering_compress_dataset
#### Description:
Compress dataset using the clustering approach.
#### Arguments:
```
num_cluster: an integer specifying the number of clusters to be produced.
file: specifies the dataset to be compressed.
```
#### Return:
```
200: the compressed dataset in a pickled format.
400: an error message if error occurred at runtime.
```
#### Example:
```
meta_data = {'num_cluster': n_cluster}
file_data = {'file': memfile}
res = requests.post(url="URL:port/clustering_compress_dataset", data=meta_data, files=file_data)
```
### 8.	ae_extract_feature
#### Description:
Extract features from a dataset using autoencoder.
#### Arguments:
```
input_dim: an integer specifying the dimension of input dataset.
output_dim: an integer specifying the dimension of the extracted data.
num_layers: an integer specifying the number of layers of neural network.
epochs: an integer specifying the number of iterations.
file_train: specifies the training dataset.
file_test: specifies the dataset from which to extract features.
```
#### Return:
```
200: the extracted features which are in a pickled format.
400: an error message if error occurred at runtime.
```
#### Example:
```
meta_data = {'input_dim': 784, 'output_dim': 16, 'num_layers': 7, 'epochs': 10}
file_data = {'file_train': memfile_train, 'file_test': memfile_test}
res = requests.post(url='URL/ae_extract_feature',      data=meta_data, files=file_data)
```
#### 9.extract_mfcc
#### Description:
Extract MFCCs (Mel-Frequency Cepstral Coefficients) from sounds.
#### Arguments:
```
path: a string specifying the path to sound dataset.
file: specifies the sound dataset from which to extract MFCCs and it is in a pickled format.
```
#### Return:
```
200: the extracted labels and MFCCs in a pickled form.
400: an error message if error occurred at runtime.
```
#### Example:
```
meta_data = {'path': './examples/audio_data/dataset'}
file_data = {'file': memfile}
res = requests.post(url='URL:port/extract_mfcc', data=meta_data, files=file_data)
```


### 10.	pca_analysis
#### Description:
Extract principal components from a dataset using principal component analysis.
#### Arguments:
```
percentage: a float number specifying the number of principal components.
file_train: specifies the training dataset used to train the model.
file_test: specifies the dataset from which to extract principal components.
```
#### Return:
```
200: the extracted principal components in a pickled form.
400: an error message if error occurred at runtime.
```
#### Example:
```
meta_data = {'percentage': 0.95}
file_data = {'file_train': memfile_train, 'file_test': memfile_test}
res = requests.post(url='URL', data=meta_data, files=file_data)
```		



### Decompress Methods

### 11.	jpeg_decompress
#### Description:
Decompress the specified JPEG image to a bit map image.
#### Arguments:
```
file: Specifies the JPEG image to be decompressed.
```
#### Return:
```
200: The resulting bit map image.
400: An error message if error occurred at runtime.
```
#### Example:
```
file_data = {'file': jpg_image}
res = requests.post(url='URL:port/jpeg_decompress', files=file_data)
```

### 12.	mp3-to-wav
#### Description:
Converts an mp3 audio clip to a wav audio clip.
#### Arguments:
```
file: Specifies the mp3 audio clip to be decompressed.
```
#### Return:
```
200: The resulting wav clip.
400: An error message if error occurred at runtime.
```
#### Example:
```
file_data = {'file': mp3_data}
res = requests.post(url="URL:port/mp3_to_wav", files=file_data)
```

### 13.	decompress_nparray
#### Description:
Reconstruct a compressed numpy array.
#### Arguments:
```
file: Specifies the numpy array to be reconstructed.
```
#### Return:
```
200: the reconstructed numpy array. It is in a pickled format.
400: an error message if error occurred at runtime.
```
#### Example:
```
file_data = {'file': memfile}
res = requests.post(url="URL:port/decompress_nparray", files=file_data)
```



### 14.	decompress_nparray_zlib
#### Description:
Reconstruct the specified numpy array using the zlib algorithm.
#### Arguments:
```
file: specifies the numpy array to be reconstructed.
```
#### Return:
```
200: the reconstructed numpy array. It is in a pickled format.
400: an error message if error occurred at runtime.
```
#### Example:
```
file_data = {'file': memfile}
res = requests.post(url="URL:port/decompress_nparray_zlib", files=file_data)
```



### 15.	decompress_nparray_bz2
#### Description:
Reconstruct a bz2-compressed numpy array.
#### Arguments:
```
file: specifies the numpy array to be reconstructed.
```
#### Return:
```
200: the reconstructed numpy array.
400: an error message if error occurred at runtime.
```
#### Example:
```
file_data = {'file': memfile}
res = requests.post(url="URL:port/decompress_nparray_bz2", files=file_data)
```


### 16.	decompress_obj
#### Description:
Reconstruct a compressed python object.
#### Arguments:
```
file: specifies the python object to be reconstructed.
```
#### Return:
```
200: the reconstructed python object.
400: an error message if error occurred at runtime.
```
#### Example:
```
file_data = {'file': memfile}
res = requests.post(url="URL:port/decompress_obj", files=file_data)
```



### 17.	clustering_decompress_dataset
#### Description:
Reconstruct a compressed dataset.
#### Arguments:
```
num_points_in_cluster: an integer specifying the number of samples in each cluster.
file: specifies the dataset to be reconstructed.
```
#### Return:
```
200: the reconstructed dataset.
400: an error message if error occurred at runtime.
```
#### Example:
```
meta_data = {'num_points_in_cluster': n_points_in_cluster}
file_data = {'file': memfile}
res = requests.post(url='URL:port/clustering_decompress_dataset', data=meta_data, files=file_data)
```



## Interpreting the Output
 
Return a Pickeled format ready for AI pipeline. See Return code for additional explanations.

For questions and feedback please email us at EdgeAI.User@ibm.com

