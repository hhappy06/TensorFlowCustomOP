#include <stdio.h>
#include <math.h>

// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

REGISTER_OP("SinWx")
	.Attr("T: {float, double}")
	.Input("x: T")
	.Input("w: T")
	.Output("o: T");

REGISTER_OP("SinWxGrad")
	.Attr("T: {float, double}")
	.Input("x: T")
	.Input("w : T")
	.Input("grad: T")
	.Output("grad_x: T")
	.Output("grad_w: T");


using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

// defination of op
template <typename  Device, typename T>
class SinWxOp : public OpKernel {
public:
	explicit SinWxOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
		const Tensor& input_x = context->input(0);
		const Tensor& input_w = context->input(1);
		auto input_x_flat = input_x.flat<T>();
		auto input_w_flat = input_w.flat<T>();

		OP_REQUIRES(context, input_x.dims() == 1,
					errors::InvalidArgument("x must be 1-dimension"));

		OP_REQUIRES(context, input_w.dims() == 1,
					errors::InvalidArgument("x must be 1-dimension"));

		int length_w = input_w.dim_size(0);
		int length_x = input_x.dim_size(0);

		OP_REQUIRES(context, length_w == length_x,
					errors::InvalidArgument("length of input must equal to lenth of weight"));

		// make the output shape
		int dims[1] = {1};
		TensorShape output_shape;
		TensorShapeUtils::MakeShape(dims, 1, &output_shape);
		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
		auto output = output_tensor->flat<T>();
		output.setZero();

		float sum_val(0.0f);
		for(int i = 0; i < length_x; i++) {
			sum_val += input_w_flat(i) * input_x_flat(i);
		}
		output(0) = sin(sum_val);
	}
};

// defination of gradient op
template<typename Device, typename T>
class SinWxGradOp : public OpKernel {
public:
	explicit SinWxGradOp(OpKernelConstruction* context): OpKernel(context) {
	}
	
	void Compute(OpKernelContext* context) override {
		const Tensor& input_x = context->input(0);
		const Tensor& input_w = context->input(1);
		const Tensor& input_grad = context->input(2);

		OP_REQUIRES(context, input_x.dims() == 1,
					errors::InvalidArgument("x must be 1-dimension"));

		OP_REQUIRES(context, input_w.dims() == 1,
					errors::InvalidArgument("w must be 1-dimension"));

		OP_REQUIRES(context, input_grad.dims() == 1 && input_grad.dim_size(0) == 1,
					errors::InvalidArgument("x must be 1-dimension"));

		OP_REQUIRES(context, input_x.dim_size(0) == input_w.dim_size(0),
					errors::InvalidArgument("input and weight must have the same dimension"));

		auto input_x_flat = input_x.flat<T>();
		auto input_w_flat = input_w.flat<T>();
		auto input_grad_flat = input_grad.flat<T>();

		int dims[1];
		dims[0] = input_x.dim_size(0);
		TensorShape output_shape;
		TensorShapeUtils::MakeShape(dims, 1, &output_shape);

		Tensor* grad_x = NULL;
		Tensor* grad_w = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &grad_x));
		OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &grad_w));
		auto grad_x_flat = grad_x->template flat<T>();
		auto grad_w_flat = grad_w->template flat<T>();

		T sum_val(0.0);

		int nsize = input_x.dim_size(0);
		for (int i = 0; i < nsize; ++i) {
			sum_val += input_x_flat(i) * input_w_flat(i);
		}

		sum_val = input_grad_flat(0) * cos(sum_val);

		for (int i = 0; i < nsize; i++) {
			grad_x_flat(i) = sum_val * input_w_flat(i);
			grad_w_flat(i) = sum_val * input_x_flat(i);
		}
	}
	
};

REGISTER_KERNEL_BUILDER(Name("SinWx").Device(DEVICE_CPU).TypeConstraint<float>("T"), SinWxOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("SinWxGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), SinWxGradOp<CPUDevice, float>);
