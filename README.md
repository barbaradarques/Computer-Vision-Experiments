# IC_ImageRecognition

Relevant files:
- **output2file.py** <br/>Saves the output of selected layers of a given CNN following the format below:
  <br/>\<image name> \<output values> \<image ground truth label>
- **dim_reduction.py** <br/> T-NSE treatment over results
- **custom_layers.py** <br/> Implements TiedDenseLayer, which can have it's weights tied to either a  transpose or a Moore–Penrose inverse of the target layer's weight.
- **boxplot.py** <br/> Module responsible for organizing SVM results in comparative boxplot graphs.
- **svm_tests.py** <br/>
- **autoenconders_tests.py** <br/>
