### FIRE171 - ASN4

- Used deep learning techniques to predict if an individual is going to pay their bank balance next month.
- Created multi-layer neural network.

The script written for training the model is stored in model.py. 
Running model.py from the command line allows you to input properties of the model, and then chose whether or not to store the weights and architecture of the model created (.h5 and .json files).
After training the model, predictions can be made from the stored files through the predict_from_model.py file, which should be run from the command line.

#### Folders:

**Data:** Folder containing all provided data
- [Descriptions of Labels](data/data-description.csv)
- [Training data](data/train.csv)
- [Sample Submission](data/sample-submission.csv)
- [Test data](data/test.csv)

**Models:** Folder containing all saved models
- .h5 files store the weights of the model
- .json files store model architecture

**Predictions:** Folder containing csv's with predictions
- files are named based on test accuracy percentages and/or network architecture
