"""
PneumoniaClassifier
- Harsha Vardhan, Khurdula
hkhurdul@purdue.edu
"""
import joblib
import base64
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing import image
import numpy as np 

class VGG16():
    """
    This module acts as a middleware between the endpoint (Flask-api) and the Binary of the model.
    """
    __model = None
    __image_data = None
    __xray = None
    __THRESHOLD__ = None

    def __init__(self) -> None:
        """
        The constructor must initialize the model, prior to anything.
        """
        try:
            print("Attempting to load Model please wait...")
            __path__ = "../Models/VGG16.joblib"
            self.__model = joblib.load(__path__)
            print("[LOG] Model has been initialized successfully.")
            self.__THRESHOLD__ = 0.40
            print("[LOG] Model-Threshold has been set to 0.400 successfully.")
        except Exception as e:
            print("[ERR] The following exception occured while trying to load the model: "+str(e))

    def __decode_image(self, encoded_image: str) -> str:
        """
        The VGG16.decode_image() accepts a base64 encoded string which is passed from the client side.
        This image is used to retracted by decoding it and then creating a string of Bytes, which represents data of the image,
        Which in this scenario is Chest X-ray.
        """
        try:
            image_data = base64.b64decode(encoded_image) #decode the image.
            self.__image_data = Image.open(BytesIO(image_data)) #create bytes data from the image data.
            return self.__image_data
        except Exception as e:
            return {
                "Status": 500,
                "Error": "The following error occured while trying to decode your image: "+str(e)
            }

    def __preprocess_xray(self) -> np.array:
        """
        Preprocessing is an mandatory step for the VGG16,
        1. xray has to be reshaped to 224 x 224.
        2. covert the image data into an numpy array.
        3. expland the dimensions to 3 channels.
        4. normalize the array to 0~1.
        """
        try:
            self.__xray = self.__image_data.resize((224, 224)) # Resize the xray to 224 x 224.
        
            if self.__xray.mode != 'RGB':
                self.__xray = self.__xray.convert('RGB')  # Convert grayscale to RGB if necessary.
            self.__xray = image.img_to_array(self.__xray) # Convert the PIL image to a numpy array.
            self.__xray = np.expand_dims(self.__xray, axis=0) # Expand dimensions to 3 channels.
            self.__xray /= 255.0 # Normalize the values to 0~1.

            return self.__xray
        except Exception as e:
            return {
                "Status": 500,
                "Error": "The following error occured while trying to preprocess your image: "+str(e)
            }

    def __get_label(self, probability: float) -> str:
        """
        The get_label method compares the probablity/confidence against the Threshold:
        If the probablity is greater than the threshold we predict the label to be, "Postive"
        else, "Negative"
        """
        try:
            if probability> self.__THRESHOLD__:
                return "Positive"
            else:
                return "Negative"
            
        except Exception as e:
            return {
                "Status": 500,
                "Error": "The following error occured while trying to generate label your image: "+str(e)
            }

    def prediction(self, encoded_image: str) -> dict:
        """
        This method is the primary method which is used to make prediction on the image provided.
        """
        try:
            xray = self.__decode_image(encoded_image=encoded_image) #decode the xray.
            xray = self.__preprocess_xray() #preprocess the xray.
            if self.__model is None:
                raise Exception("Model did not load!")
            predicted =  self.__model.predict(xray)
            probability = predicted[0][0]

            result = dict()
            result["Status"] = 200
            result["Pneumonia"] = self.__get_label(probability= probability)
            result["Confidence"] = float(probability)
            result["Copyright"] = "Harsha Vardhan, Khurdula :-: hkhurdul@pfw.edu" 
            return result
        except Exception as e:
            return {
                "Status": 500,
                "Error": "The following error occured while trying to make prediction your image: "+str(e)
            }
        