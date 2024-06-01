# PRodigy_ML_03
Implementing a support vector machine (SVM) to classify images of cats and dogs.

#### Overview
The code aims to implement a Support Vector Machine (SVM) for classifying images of cats and dogs from a Kaggle dataset.
The SVM model is trained and evaluated, and a prediction is demonstrated on a sample image.

#### Dataset
The training archive comprises 25,000 images of dogs and cats. The project focuses on training the SVM model using these files to enable accurate classification. Subsequently, 
the trained model predicts labels for test1.zip, differentiating between dogs (1) and cats (0).
A part of 25000 images were used for traingi and a part of test1.zip was used for testing.

**Dataset**: https://www.kaggle.com/c/dogs-vs-cats/data

#### Packages and its uee case:
1.Python: The primary programming language for the entire solution.

2.TensorFlow and Keras: These are used for building and training deep learning models, specifically Convolutional Neural Networks (CNNs) in this case.

3.Matplotlib: Used for plotting and visualizing data, including displaying images and training history graphs

4.OpenCV: Used for image processing tasks such as reading and resizing images.

5.Zipfile: Used for handling zip files, such as extracting dataset files.

#### Technologies and Tools:
1.**Kaggle API**:
These are used to download datasets from Kaggle.

2. **Google Colab**:
The notebook contains metadata suggesting it is intended for use in Google Colab, which supports GPU acceleration.

3.**Downloading and Extracting Data from zip file**:

4.**Image Processing with OpenCV**:

5.**Building a CNN Model with Keras**:

6.**Data Loading and Preparation**:
The code uses the **tensorflow.keras.utils.image_dataset_from_directory** function to load and prepare the dataset.
This function helps in loading images from a directory and creates a dataset ready for training.

7.**Data Processing**:
A process function is used to preprocess the images. This typically involves operations like normalization.

8.**Training the Model**:
The model is trained using the fit method. The training data (train_ds) and validation data (validation_ds) are used for this purpose.
The number of epochs is set to 10.

9.**Plotting Training History**:
After training, the history of training and validation accuracy and loss is plotted using Matplotlib.

10**Loading and Preprocessing the Image**
The image is loaded using OpenCV and preprocessed to match the input dimensions and scale expected by the model.

11.**Displaying the Image**:
plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)):
Converts the image from BGR to RGB format (since Matplotlib expects RGB) and displays it using Matplotlib.

12.**Interpreting the Result**:
The prediction result is printed out, which typically needs to be interpreted in the context of the problem (e.g., a value close to 0 or 1 indicating the class in a binary classification).

#### Acuuracy: ~80.40%

#### Best Parameters

1.**Learning Rate**: The default learning rate for Adam optimizer.
2.**Batch Size**: A batch size of 32, which is a common choice and balances memory usage and training speed.
3.**Epochs**: Training for 10 epochs, which should be adjusted based on the performance on the validation set to avoid overfitting or underfitting.

