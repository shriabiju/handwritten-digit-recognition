# AI-ML Assignment 8 — Handwritten Digit Recognition using Artificial Neural Networks (ANN)

## Objective
A postal service organization wants to automate the recognition of handwritten digits
on postal codes. This project develops an Artificial Neural Network (ANN) to classify
handwritten digits (0–9) using the MNIST dataset.

## Dataset Link
MNIST Handwritten Digits dataset, CSV format (Kaggle):
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

> The dataset (`mnist_train.csv`, `mnist_test.csv`) is **not** included in this repo —
> download it from the Kaggle link above and place both files in the project folder
> before running the notebook.

## Libraries Used
- pandas
- numpy
- matplotlib
- scikit-learn (train_test_split, confusion_matrix, classification_report)
- tensorflow / keras (Sequential, Dense, to_categorical)

## Methodology
1. **Data Understanding** — Loaded `mnist_train.csv` and `mnist_test.csv` with Pandas
   and combined them, inspected the first five records, identified the 784 pixel
   columns as input features and `label` as the target variable, reviewed dataset
   dimensions/summary info, and displayed one sample digit image reshaped to 28×28.
2. **Data Preprocessing** — Checked for missing values (none expected), separated
   features and target, normalized pixel values to the 0–1 range, split the data into
   80% training / 20% testing with stratification, and one-hot encoded the labels
   with `to_categorical`.
3. **Model Development** — Built a Sequential ANN with an input layer (784 features),
   two hidden layers (128 and 64 neurons, ReLU activation), and a 10-neuron softmax
   output layer. Compiled with the Adam optimizer and categorical crossentropy loss,
   trained for 10 epochs, and predicted digit labels on the test set.
4. **Model Evaluation** — Evaluated test accuracy, plotted a confusion matrix and
   printed a classification report, and plotted accuracy vs. epoch and loss vs. epoch
   curves for both training and validation.

## Model Architecture
| Layer | Type | Units | Activation |
|---|---|---|---|
| Input | — | 784 (28×28 pixels) | — |
| Hidden Layer 1 | Dense | 128 | ReLU |
| Hidden Layer 2 | Dense | 64 | ReLU |
| Output Layer | Dense | 10 | Softmax |

**Compile settings:** Optimizer = Adam, Loss = Categorical Crossentropy, Metric = Accuracy
**Training:** 10 epochs, batch size 128, 20% validation split

## Results
See `Assignment-8.ipynb` for exact metric values from your run. In general, this
architecture is expected to reach high test accuracy (typically well above 95%) on
MNIST, with most misclassifications concentrated between visually similar digit pairs
(e.g. 4/9, 3/5, 7/1) rather than spread evenly across all digits.

## Conclusion
This project built an Artificial Neural Network to classify handwritten digits from
the MNIST dataset, using two hidden layers of 128 and 64 ReLU neurons and a 10-neuron
softmax output layer. After normalizing pixel values to the 0–1 range and one-hot
encoding the labels, the model was trained for 10 epochs with the Adam optimizer and
categorical crossentropy loss, achieving strong test accuracy with most confusion
concentrated between visually similar digit pairs. Hidden layers are what give an ANN
its power: without them, the network could only learn a linear mapping from raw pixels
to digit classes, but the hidden layers let the model progressively combine simple
pixel patterns into more abstract, non-linear representations of stroke shapes and
digit structure, which is essential for separating classes that are not linearly
separable in raw pixel space. Compared to traditional machine learning approaches such
as KNN or a single Decision Tree, one advantage of this deep learning approach is that
it automatically learns useful feature representations directly from raw pixel data,
rather than requiring manual feature engineering. A limitation of ANNs, however, is
that a fully connected network like this one treats each pixel independently and
ignores the 2D spatial structure of the image, meaning it cannot recognize shifted,
rotated, or scaled versions of a digit as easily as an architecture designed for
images (such as a Convolutional Neural Network) would.
