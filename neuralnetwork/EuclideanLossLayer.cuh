#ifndef EUCLIDEANLOSSLAYER_CUH_INCLUDED
#define EUCLIDEANLOSSLAYER_CUH_INCLUDED

#include "common.cuh"
#include <stdio.h>


static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}


class EuclideanLossLayer {
public:
    EuclideanLossLayer(int input_size, int output_size, const string& name_) {
        assert(0 == init_empty(&loss, input_size, output_size));
        assert(0 == assign_scalar(&loss, 0));
        name = name_;
    }

    ~EuclideanLossLayer() {
        assert(0 == free_device_memory(&loss));
    }


    Dtype forward(Mat& pred, Mat& target) {
        cout << name << " forward " << endl;
        assert(0 == subtract_elementwise(&pred, &target, &loss));

        // there is some problem unknown about using this function
        int err = 0;
        Dtype l2 = 0.5 * euclid_norm(&loss, &err);
        assert(0 == err);

        //assert(0 == mult_elementwise(&loss, &loss, &loss, 0));
        //Dtype l2 = sum_all(&loss, &err);
        return l2;
    }

    void backward(Mat& pred, Mat& target, Mat& d_b) {
        cout << name << " backward " << endl;
        assert(0 == subtract_elementwise(&pred, &target, &d_b));
    }

    string name;
    Mat loss;
};

#endif // EUCLIDEANLOSSLAYER_CUH_INCLUDED
