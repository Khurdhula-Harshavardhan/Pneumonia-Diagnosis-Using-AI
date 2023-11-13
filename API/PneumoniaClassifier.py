"""
PneumoniaClassifier
- Harsha Vardhan, Khurdula
hkhurdul@purdue.edu
"""
import joblib 

class VGG16():
    """
    This module acts as a middleware between the endpoint (Flask-api) and the Binary of the model.
    """
    __model = None

    def __init__(self) -> None:
        """
        The constructor must initialize the model, prior to anything.
        """
        try:
            __path__ = "Models/VGG16.joblib"
            self.__model = joblib.load(__path__)
        except Exception as e:
            print("[ERR] The following exception occured while trying to load the model: "+str(e))



def driver():
    obj = VGG16()


driver()