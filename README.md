# IC_ImageRecognition

Relevant folders and files:
- **./output2file.py** <br/>Saves the output of selected layers of a given CNN following the format below:
  <br/>\<image name> \<output values> \<image ground truth label>
- **./svm.py** <br/> Contains functions that test different parameters on different SVM kernels.
- **./main.py** <br/> Tests the functionalities regarding the modules above.
- **./Produce_1400/** <br/>*Produce* image database
- **./produce-fc1.txt** and **./produce-fc2.txt** <br/>Contain the outputs of *VGG-16*'s first and second FC layers
- **./svm_performance/<database_name>/** <br/> Contains the accuracy scores of each database when varying SVM parameters and the CNN output layer.  
