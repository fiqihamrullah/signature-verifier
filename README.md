# Signature-Verifier

This application train images which are transformed into binary by thresholding as feature with convolutional neural network (CNN) using the Keras library. CNN creates a sequential model and 2D convolutional layer with 64 filters, each with a size of 3x3. The shape of the input data is a 50x50 pixel binary image and a max pooling layer with a pool size of 2x2 with ReLU activation function. CNN trains and tests images using split validation in order to evaluate performance.
