
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

Methods for Coreset API.

REST APIs are using an HTTP post method to invoke the backend Coreset service. The payload of the post method has a multipart/form-data format in which form is used to pass arguments and file is used to pass data. 

In the following, a sample is given for each method in python syntax. 

All the calls have a format of res = requests.post(url='http://hostname:port/method', data=meta_data, files=file_data) where meta_data defines the arguments passed to the algorithm and file_data specifies the dataset used for processing. The return of each call carries a status code and the processing result in a pickled format. 

The status code 200 indicates success, and the status code 400 denotes an error condition.


1.	wav-to-mp3

Description:  
Converts a wav audio clip to an mp3 audio clip.
Arguments: 
rate: a string defining the compression data rate and ‘128k’, ‘192k’ and ‘256k’ are allowed.
	          	file: specifies the wav audio clip to be compressed.
Return: 
200: the resulting mp3 clip.
400: an error message if error occurred at runtime.
Example:
meta_data = {'rate': '192k'}
file_data = {'file': wav_data}
res = requests.post(url="URL:port/wav_to_mp3", data=meta_data, files=file_data) 


2.	mp3-to-wav

 	Description:
Converts an mp3 audio clip to a wav audio clip.
 Arguments:
file: specifies the mp3 audio clip to be decompressed.
Return:
	200: the resulting wav clip
400: an error message if error occurred at runtime.
Example:
file_data = {'file': mp3_data}
res = requests.post(url="URL:port/mp3_to_wav", files=file_data)


3.	jpeg_compress

Description:
Compress an image to JPEG image
Arguments:
		quality: an integer specifying the quality of the compressed JPEG image. 
It ranges from 0 to 100.
		file: specifies the image to be compressed
Return:
		200: the resulting JPEG image
400: an error message if error occurred at runtime.
	Example:
meta_data = {'quality': 10}
file_data = {'file': image_data}
res = requests.post(url='URL:port0/jpeg_compress', data=meta_data, files=file_data)


4.	jpeg_decompress

Description:
Decompress the specified JPEG image to a bit map image.
Arguments:
		file: specifies the JPEG image to be decompressed.
Return:
	200: the resulting bit map image
400: an error message if error occurred at runtime.
Example:
file_data = {'file': jpg_image}
res = requests.post(url='URL:port/jpeg_decompress', files=file_data)

5.	compress_nparray

Description:
Losslessly compress the specified numpy array using the default algorithm.
Arguments:
		file: the specified numpy array to be compressed. It is in a pickled format.
Return:
	200: the compressed numpy array.
400: an error message if error occurred at runtime.
Example:
file_data = {'file': memfile}
res = requests.post(url="URL:port/compress_nparray", files=file_data)


6.	decompress_nparray
Description:
Reconstruct a compressed numpy array.
Arguments:
file: specifies the numpy array to be reconstructed.
Return:
		200: the reconstructed numpy array. It is in a pickled format.
400: an error message if error occurred at runtime.
           Example:
	file_data = {'file': memfile}
  res = requests.post(url="URL:port/decompress_nparray", files=file_data)



7.	compress_nparray_zlib

Description:
		Losslessly compress a numpy array using the zlib algorithm.
Arguments:
		file: the specified numpy array to be compressed. It is in a pickled format.
Return:
		200: the compressed numpy array.
400: an error message if error occurred at runtime.
Example:
file_data = {'file': memfile}
    	res = requests.post(url="URL:port/compress_nparray_zlib", files=file_data)




8.	decompress_nparray_zlib
Description:
Reconstruct the specified numpy array using the zlib algorithm.
Arguments:
file: specifies the numpy array to be reconstructed.
Return:
		200: the reconstructed numpy array. It is in a pickled format.
400: an error message if error occurred at runtime.
	Example:
		file_data = {'file': memfile}
res = requests.post(url="URL:port/decompress_nparray_zlib", files=file_data)



9.	compress_nparray_bz2

Description:
		Losslessly compress a numpy array using the bz2 algorithm.
Arguments:
		file: the specified numpy array to be compressed using the bz2 algorithm.
Return:
		200: the compressed numpy array.
400: an error message if error occurred at runtime.
Example:
file_data = {'file': memfile}
res = requests.post(url="URL:port/compress_nparray_bz2", files=file_data)


10.	decompress_nparray_bz2

Description:
Reconstruct a bz2-compressed numpy array.
Arguments:
file: specifies the numpy array to be reconstructed.
Return:
		200: the reconstructed numpy array.
400: an error message if error occurred at runtime.
	Example:
		file_data = {'file': memfile}
res = requests.post(url="URL:port/decompress_nparray_bz2", files=file_data)



11.	compress_obj

Description:
		Losslessly compress a python object using the default compression algorithm.
Arguments:
		file: the specified python object to be compressed. It is in a pickled format.
Return:
	200: the compressed python object.
400: an error message if error occurred at runtime.
Example:
file_data = {'file': memfile}
res = requests.post(url="/compress_nparray_obj", files=file_data)

12.	decompress_obj

Description:
Reconstruct a compressed python object.
Arguments:
file: specifies the python object to be reconstructed.
Return:
		200: the reconstructed python object.
400: an error message if error occurred at runtime.
	Example:
		file_data = {'file': memfile}
res = requests.post(url="URL:port/decompress_obj", files=file_data)



13.	clustering_compress_dataset

Description:
		Compress dataset using the clustering approach.
Arguments:
		num_cluster: an integer specifying the number of clusters to be produced.
		file: specifies the dataset to be compressed.
Return:
		200: the compressed dataset in a pickled format.
400: an error message if error occurred at runtime.
Example:
		meta_data = {'num_cluster': n_cluster}
file_data = {'file': memfile}
res = requests.post(url="URL:port/clustering_compress_dataset", data=meta_data, files=file_data)



14.	clustering_decompress_dataset

Description:
Reconstruct a compressed dataset.
Arguments:
num_points_in_cluster: an integer specifying the number of samples in each cluster.
file: specifies the dataset to be reconstructed.
Return:
		200: the reconstructed dataset.
400: an error message if error occurred at runtime.
           Example:
meta_data = {'num_points_in_cluster': n_points_in_cluster}
file_data = {'file': memfile}
res = requests.post(url='URL:port/clustering_decompress_dataset', data=meta_data, files=file_data)


15.	extract_mfcc

Description:
		Extract MFCCs (Mel-Frequency Cepstral Coefficients) from sounds.
Arguments:
		path: a string specifying the path to sound dataset.
file: specifies the sound dataset from which to extract MFCCs and it is in a pickled      format.
Return:
		200: the extracted labels and MFCCs in a pickled form.
400: an error message if error occurred at runtime.
Example:
		meta_data = {'path': './examples/audio_data/dataset'}
file_data = {'file': memfile}
res = requests.post(url='URL:port/extract_mfcc', data=meta_data, files=file_data)











16.	ae_extract_feature

Description:
	Extract features from a dataset using autoencoder.
Arguments:
	input_dim: an integer specifying the dimension of input dataset.
	output_dim: an integer specifying the dimension of the extracted data.
	num_layers: an integer specifying the number of layers of neural network.
	epochs: an integer specifying the number of iterations.
	file_train: specifies the training dataset.
	file_test: specifies the dataset from which to extract features.
Return:
	200: the extracted features which are in a pickled format.
400: an error message if error occurred at runtime.
Example:
    	meta_data = {'input_dim': 784, 'output_dim': 16, 'num_layers': 7, 'epochs': 10}
file_data = {'file_train': memfile_train, 'file_test': memfile_test}
res = requests.post(url='URL/ae_extract_feature',      data=meta_data, files=file_data)



17.	pca_analysis

Description:
		Extract principal components from a dataset using principal component analysis.
Arguments:
		percentage: a float number specifying the number of principal components.
		file_train: specifies the training dataset used to train the model.
		file_test specifies the dataset from which to extract principal components.
Return:
	200: the extracted principal components in a pickled form.
 	400: an error message if error occurred at runtime.
Example:
    	meta_data = {'percentage': 0.95}
    	file_data = {'file_train': memfile_train, 'file_test': memfile_test}
res = requests.post(url='URL', data=meta_data, files=file_data)
		

## Interpreting the Output
 
blah blah here

