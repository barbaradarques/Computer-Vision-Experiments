# Computer Vision Experiments

This repository relates to 2 different sets of experiments ran during my time as undergraduate researcher:

1) Analysis of the role of mid-level representations on transfer learning. It was also explored the impact of different SVM kernels on the usage of such represetations for classification tasks.

2) Investigation of the usage of different autoencoder architectures as feature extractors, including the attempt of tying coding and decoding layers by the Moore–Penrose inverse of their weights.

Relevant files:
- **output2file.py** <br/>Saves the output of selected layers of a given CNN following the format below:
  <br/>\<image name> \<output values> \<image ground truth label>
- **dim_reduction.py** <br/> T-NSE treatment over results
- **custom_layers.py** <br/> Implements TiedDenseLayer, which can have it's weights tied to either a  transpose or a Moore–Penrose inverse of the target layer's weight.
- **boxplot.py** <br/> Module responsible for organizing SVM results in comparative boxplot graphs.
- **svm_tests.py** <br/> Tests ran testing different SVM kernels on the output of different intermediate layers.
- **autoenconders_tests.py** <br/> Performance tests ran on different autoencoders configurations.
