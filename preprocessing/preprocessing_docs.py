import numpy as np
import cv2
import scipy.cluster.vq as vq
import json
import joblib
# from sklearn.preprocessing import StandardScaler
scaler = joblib.load('../data/std_scaler.pkl')


class Preprocess:
    def __init__(self, image_path):
        self.image_path = image_path

    def resizing_images(self):
        '''
        Resizing an image.
        '''
        IMG_HEIGHT = 1000
        IMG_WIDTH = 800

        image_array = cv2.imread(self.image_path)
        resized_image_array = cv2.resize(image_array, (IMG_HEIGHT, IMG_WIDTH))
        return resized_image_array

    def denoising_images(self):
        """
        Denoising for better generalization.
        """

        thresh = cv2.threshold(self.resizing_images(), 220, 255, cv2.THRESH_BINARY_INV)[1]

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)

        result = 255 - opening
        return result

    def ORB_extractor(self):

        '''
        Extracting interesting points from images.

        Output:
        descriptors_scaled - array containing scaled descriptors
        descriptors_list - array containing images and corresponding set of descriptors

        '''
        image = self.denoising_images()  # denoised image

        # checking if the image is a list
        #if type(image) is list:
            #pass
        #else:
            #image = [image]

        descriptors_list = []
        orb = cv2.ORB_create(nfeatures=1200)

        keypoint = orb.detect(image, None)
        keypoints, descriptor = orb.compute(image, keypoint)
        descriptors_list.append((image, descriptor))

        return descriptors_list

    def read_codebook_json(self):
        '''
        Loading a codebook from a json file
        '''
        with open(r'C:\Users\i\Documents\PROJECTS\SPM_docs\model_docs\flaskDocProject\codebook.json', 'r') as handle:
            codebook = json.load(handle)
        return codebook

    def vector_encoder(self):
        '''
        Vector Quantization

        Input:
        X - image dataset
        descriptors_list - array: images and corresponding list of its descriptors
        codebook,
        voc_size  - vocabulary size (the number of desired clusters)
        Output:
        image_features - list of compressed images
        '''
        image = self.denoising_images()
        descriptors_list = self.ORB_extractor()
        codebook = self.read_codebook_json()
        voc_size = len(codebook)

        image_features = np.zeros((1, voc_size), "float32")

        words, distance = vq.vq(descriptors_list[0][1], codebook)

        for w in words:
            image_features[0][w] += 1

        # standard_scaler = StandardScaler().fit(image_features)
        image_features = scaler.transform(image_features)

        return image_features

    def print_image_features(self):
        img_features = self.vector_encoder()
        print(img_features)

# decoding labels
def decoding_labels(prediction):

    predicted_list = []
    for i in range(len(prediction[0][0])):
      if prediction[0][0][i] == True:
        if i == 0:
            predicted_list.append("News")
        elif i == 1:
            predicted_list.append("Resume")
        elif i == 2:
            predicted_list.append("Scientific")
      else:
        pass

    return predicted_list