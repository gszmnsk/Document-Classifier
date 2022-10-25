# Scanned document classification based on its structure.

Methods used:
<ul>
<li>OpenCV, ORB for points detection and extraction</li>
<li>Kmedoids for clustering and codebook making</li>
<li>Vector Quantization for as a data compression method</li>
<li>CNN as a Deep Learning classifier</li>
<li>Flask for model testing</li>
</ul>

### The preprocessing part.

The aim of the project is to classify scanned documents by utilising classical methods of Machine Learning. The data collection is the Tabocco scanned documents dataset and it divides into 3 categories. <br>
The data is contained in ordered folders, each folder is a separate category of document. The first load_data function classifies each folder based on their indexes and assigns the category index to the class_number variable. Then it loads the data from directory and, based on in which folder the scanned image of a document is, assignes it to corresponding category. The data array containing the images and corresponding categories value is created. Then the X (image data) and y (labeles) are separated. <br>
The denoising_images function denoises the images for a better points detection obtained using ORB feature extraction. The ORB_extractor function takes an input of the image dataset, detects and extracts interesting points from each image and creates an array containing the array of images and corresponding array of descriptors. The function outputs two seperate arrays: descriptors_scaled - the list of scaled desctiptors, which elements are ordered as in the X dataset, and descriptors_list - the array of images and descriptors extracted from each image. Then the codebook using Kmedoids method is being made. The build_codebook function takes the descriptors array (descriptors_scaled) and the selected vocabulary size (the number of clusters we want to have) and then it clusters the descriptors that are the closest to each other. After that the function outputs the center points of these clusters. The data used in the codebook making is only a part of the X set for the sake of time of the model trainig. It should be adjusted for a better accuracy. <br>
The vector_encoder function takes the X, descriptors_list, codebook and previously fixed codebook (vocabulary) size. At first it creates image_features which is an array of zeros in a size of the length of X and the vocabulary size. For each image in descriptors_list it takes the descriptors, calculates the distance between the descriptors and the codebook centers and assigns each descriptor to the closest center (word) in a codebook. After that if the considered descriptor is closest to a given center it adds +1 to the proper place in image_features array that row is corresponding with the index of the image in X that the descriptor was taken from and the column - with the index of the codebook center. As a result each element of an array is the sum over the number of each center that the descriptors were assigned to.

### The modeling part.

Several classical machine learning models have been tested and turned out to be similarly accurate, therefore the One vs Rest algorithm is picked as an example. <br>
The deep learning classifier was trained in a Google Colab.
