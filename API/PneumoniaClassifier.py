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

    def __init__(self) -> None:
        """
        The constructor must initialize the model, prior to anything.
        """
        try:
            __path__ = "Models/VGG16.joblib"
            self.__model = joblib.load(__path__)
        except Exception as e:
            print("[ERR] The following exception occured while trying to load the model: "+str(e))

    def decode_image(self, encoded_image: str) -> str:
        """
        The VGG16.decode_image() accepts a base64 encoded string which is passed from the client side.
        This image is used to retracted by decoding it and then creating a string of Bytes, which represents data of the image,
        Which in this scenario is Chest X-ray.
        """
        try:
            image_data = base64.b64decode(encoded_image) #decode the image.
            self.__image_data = Image.open(BytesIO(image_data)) #create bytes data from the image data.
        except Exception as e:
            {
                "Status": 500,
                "Error": "The following error occured while trying to decode your image: "+str(e)
            }



def driver():
    obj = VGG16()


driver()