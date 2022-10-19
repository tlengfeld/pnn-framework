# pnn framework - Lengfeld - Seitz
## Getting started
use the `requiremnts.txt` to install the dependencies.

## project structure
The project is structured as follows:

### network-examples

the network examples folder contains two networks.

Run the `MnistFclNetwork.py`-script to see how a simple deep neuronal network with 2 hidden layers performs on the MNIST-Dataset.

The `MnistCnnNetwork.py`-script contains a convolutional neuronal network with one convolutional layer followed by a pooling layer, a flattening layer and two fully connected layer including activations.

### NeuralNetwork.py

In the `NeuralNetwork.py` the tensors are generated.

After all the data is passed to the NeuralNetwork-class the training can be started. In the function `train_one_epoch` the model is trained using one forward and one backward pass. The `test_one_epoch`-function is called with the test data to get the test loss.

The `get_correct_classified_percentage`-class returns the correct classified images in percent for an array of input-data and the corresponding ground truth.

### layers
In the `layers`-folder there is one file for each different layer-class. They are implemented according to the formulas presented in the lecture. Have a look at the networks in the `network-examples`-folder to see examples of how to use them.

### optimizers
we use GradientDescent as optimizer. This class is static and is called from the layers.

### tests
In the `TestGradientDescent.py`-file you can find a simple visual test for the gradient descent. 

All tests from the exercieses and some more tests are written using the `unittest`-package. Run the `TestNNLayers.py`-script to see how the tests perform.
