#TensorFlowCustomOP
  My practice about building a custom op and gradient in tensorflow.<br>
  The op y = sin(w*x) and its grad are defined in the file sin_wx.cc.<br>
  You can run 'python test_my_layer.py' to test the op and gradient.<br>

##The files:<br>
* gen-so.sh: compile script<br>
* sin_wx.cc: defination of op and its gradient<br>
* testing_my_layer.py: testing file<br>

##Results:
![image](http://github.com/hhappy06/TensorFlowCustomOP/raw/master/result/parameter_fitting_result.png)
