# Pneumonia-Diagnosis-Using-AI

Pneumonia is a lung inflammation primarily affecting the tiny air sacs, or alveoli. Various pathogens, such as bacteria, viruses, and fungi, can trigger it. Common symptoms encompass coughing, chest discomfort, fever, and breathing challenges.

One pivotal tool in detecting pneumonia is the Chest X-ray. When pneumonia takes hold, the alveoli might fill with pus or fluid, disrupting normal air flow. This obstruction is discernible on X-rays. Specifically, the X-ray can spotlight white opaque regions, hinting at inflammation and fluid buildup, characteristic of pneumonia.

Our new system offers a novel approach: leveraging the power of machine learning to classify these X-ray images as either "normal" or indicative of "pneumonia." Think of it as an expert diagnostic tool, fine-tuned to detect subtle hints and signs on the X-rays. Such a system can drastically elevate diagnostic accuracy.

The importance of timely pneumonia detection cannot be overstated, given its health implications. While the conventional diagnostic approach is heavily reliant on radiologists' keen eyes, our system aims to automate this process. In doing so, it not only expedites diagnosis but also minimizes human errors, making the entire process more efficient and precise.

------------------------------------------------------------------------------------------------------------------------------------------------------------


# CNN for Pneumonia Detection from Chest X-Rays

## Overview

**Convolutional Neural Network (CNN)** for the detection of pneumonia from chest X-ray images. The CNN is designed to differentiate between normal and pneumonia-affected X-ray images, where pneumonia cases include those caused by both bacteria and viruses.

## Data Preprocessing

The dataset is structured into **training and testing sets**, each containing subdirectories labeled 'NORMAL' and 'PNEUMONIA'. Data augmentation techniques such as rotation, width and height shifts, shear, zoom, and horizontal flipping are applied to the training data to enhance the model's generalization capabilities. This is implemented using the `ImageDataGenerator` class in TensorFlow, which also automatically scales the pixel values to the [0, 1] range.

A subset of 20% from the augmented training data is held out for validation purposes to monitor the model's performance and to mitigate overfitting.

## Model Architecture

The CNN architecture consists of the following layers:

- Three convolutional layers with *ReLU activation*, each followed by max-pooling layers to extract features from the X-ray images.
- A flattening step to convert the 2D feature maps into a 1D feature vector.
- A fully connected layer with 512 units and *ReLU activation*, accompanied by a dropout layer to reduce overfitting.
- The output layer with a single neuron employing a *sigmoid activation function*, providing the probability of pneumonia presence.

The model is compiled with the Adam optimizer and binary cross-entropy loss function, which is suitable for binary classification tasks.

## Training

The model is trained on the processed dataset using the `fit` method, with *early stopping* employed to halt the training if the validation loss ceases to decrease for five consecutive epochs.

## Evaluation

Post-training, the model is evaluated on the unseen test set, which is processed without augmentation to maintain the integrity of the test data. The evaluation metrics include accuracy and a *classification report*, which provides insights into precision, recall, and F1-score for each class. Here's the performance report for the CNN trained on imbalanced data:

```
                precision    recall  f1-score   support

      NORMAL       0.84      0.80      0.82       234
   PNEUMONIA       0.89      0.91      0.90       390

    accuracy                           0.87       624
   macro avg       0.86      0.86      0.86       624
weighted avg       0.87      0.87      0.87       624
```

## Visualization

Training and validation accuracies, as well as losses, are plotted against epochs to visually assess the learning process. These plots are critical for identifying overfitting, underfitting, and for determining if the model is learning effectively.

_Training VS Validation Accuracy:_

![image](https://github.com/Khurdhula-Harshavardhan/Pneumonia-Diagnosis-Using-AI/assets/60458750/4d4ab35f-b2ea-40f6-9a25-74b461d5fe99)

_Training VS Validation  Loss:_

 ![image](https://github.com/Khurdhula-Harshavardhan/Pneumonia-Diagnosis-Using-AI/assets/60458750/cfdc3994-94ae-41e8-84f0-c76ed8124aa9)

_Confusion Matrix:_

![image](https://github.com/Khurdhula-Harshavardhan/Pneumonia-Diagnosis-Using-AI/assets/60458750/0b70bda2-8d17-4feb-a514-9739bd561de6)

_ROC-AUC curve:_

![image](https://github.com/Khurdhula-Harshavardhan/Pneumonia-Diagnosis-Using-AI/assets/60458750/0329895e-8b0e-4be4-bf79-641b9912828a)



## Conclusion

The implemented CNN model demonstrates the potential of deep learning in medical imaging by automating the pneumonia detection process. The ability to discern between normal and pneumonia cases can greatly assist in early diagnosis and treatment. Future work could involve fine-tuning the model, expanding the dataset, and potentially using transfer learning to improve accuracy and reliability.

- However, there can be some enhancements that can be made: 
- Fix class im-balance.
- Increase EPOCHs for training.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Addressing Class Imbalance in Pneumonia Detection with CNNs
## Balanced-IncreasedEpochs-CNN:

## Introduction
As an Author tasked with enhancing a CNN for pneumonia detection from chest X-rays, I identified class imbalance as a significant challenge from previously trained CNN. The dataset consists of 3,800 pneumonia samples compared to only 1,400 normal samples. Such a disparity can bias the CNN towards predicting pneumonia, which could increase the false positive rate.

## Why Class Imbalance Matters
Class imbalance can greatly affect the predictive performance of machine learning models. In medical diagnostics, the repercussions of this can be severe, as it may lead to a high misdiagnosis rate. Specifically, for pneumonia detection, the model might incorrectly identify healthy patients as having the condition due to the imbalance.

## Approach to Address Class Imbalance
To address the class imbalance issue, I implemented **class weights** during the training of the CNN. This technique adjusts the model's focus towards underrepresented classes, increasing the penalty for misclassifying the minority class to incentivize the model to pay more attention to the 'NORMAL' samples.

## Implementation
Class weights were calculated using the `compute_class_weight` function from `scikit-learn`. This method ensures a balanced representation of classes by adjusting weights inversely proportional to class frequencies. These calculated weights were integrated into the training process via the `fit` method of the Keras model.

## Expected Outcomes
By applying class weights, the author aims to:

- **Enhance Sensitivity**: Improve the model's ability to correctly identify true negative cases, which are actual normal X-rays.
- **Improve Specificity**: Decrease the number of false positives, where normal cases are wrongly labeled as pneumonia.
- **Balance Precision and Recall**: Find a better balance between the accuracy of positive predictions and the ability to identify all actual positives.

## Conclusion
The inclusion of class weights in the CNN model is designed to yield more reliable and balanced diagnostic predictions. 

_Confusion Matrix of the bieCNN:_
![image](https://github.com/Khurdhula-Harshavardhan/Pneumonia-Diagnosis-Using-AI/assets/60458750/410f1a3f-e5df-42f9-b14a-5cacf40645d6)

The model is now better equipped to discern patterns indicative of pneumonia while correctly identifying normal X-rays, there's an additional gain in performance from 87% to 89~90% stable. Here are the bieCNN's metrics:

![image](https://github.com/Khurdhula-Harshavardhan/Pneumonia-Diagnosis-Using-AI/assets/60458750/49abc398-7650-4022-b3ea-00635a9bd708)


These modifications are crucial for the model's performance, ensuring its utility in clinical settings and also addressing previously issues from Vanilla CNN.

---------------------------------------------------------------------------------------------------------------------------------


_**By Author,
@Khurdhula-Harshavardhan, 
Purdue University,
Fort Wayne,
(hkhurdul@pfw.edu)**_
