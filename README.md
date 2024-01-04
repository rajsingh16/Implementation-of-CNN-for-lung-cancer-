# Implementation-of-CNN-for-lung-cancer
it contains analysis on the dataset of IQ-OTH/NCCD - Lung Cancer Dataset 
Model Training (Not Shown):

It seems that a machine learning model (possibly a neural network using TensorFlow's Keras API) has been trained using the model.fit() function on some training data (training_data) for 5 epochs with validation on testing_data.
Image Prediction:

The code uses paths to three specific images (prediction_img) representing different cases (Benign, Malignant, Normal) of lung conditions.
It then creates a plot with three subplots (fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))) to display these images.
Prediction Loop:

For each image path in prediction_img, it loads the image using OpenCV (cv2.imread) and displays it in the corresponding subplot using Matplotlib.
The code uses Keras' image.load_img() to load each image in a standardized size (target_size=(img_width, img_height)).
It converts the loaded image to a numpy array (img_to_array) and performs some preprocessing, scaling pixel values between 0 and 1.
Then, it uses the trained model (model.predict) to predict the class of the image (Benign, Malignant, or Normal) based on the image content.
The predicted class index (predicted_class) is obtained using np.argmax(predictions).
Output:

It prints the predicted class for each image path based on the model's predictions.
Optimization:

The optimized version combines the image loading and prediction steps into a single loop, reducing redundancy and improving efficiency by eliminating separate loops for loading and predicting.
