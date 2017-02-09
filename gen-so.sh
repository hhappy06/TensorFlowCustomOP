TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared sin_wx.cc -o sin_wx.so -I $TF_INC -L $TF_LIB -fPIC -Wl,-rpath $TF_LIB -undefined dynamic_lookup
