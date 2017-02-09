My practice about building a custom op and gradient in tensorflow.
The op y = sin(w*x) and its grad are defined in the file sin_wx.cc.
You can run 'python test_my_layer.py' to test the op and gradient.
The files:
	gen-so.sh: compile script
	sin_wx.cc: defination of op and its gradient
	testing_my_layer.py: testing file
