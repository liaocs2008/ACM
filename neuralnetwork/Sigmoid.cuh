#ifndef SIGMOID_CUH_INCLUDED
#define SIGMOID_CUH_INCLUDED

#include "common.cuh"

class Sigmoid {
public:
    Sigmoid(const string& name_) {
        name = name_;
    }

    ~Sigmoid() {
    }

    void forward(Mat& x, Mat& a) {
        //cout << name << " forward " << endl;
        assert(0 == apply_sigmoid(&x, &a) );
    }

    // for other layers, usually takes a, but for sigmoid, we take b
    void backward(Mat& b, Mat& d_b, Mat& d_a) {
        //cout << name << " backward " << endl;
        assert(0 == sigmoid_deriv_aux(&b, &d_a)); // d_a = a - a^2
        assert(0 == mult_elementwise(&d_a, &d_b, &d_a, 0)); // d_a = (a - a^2) * d_b
    }

    string name;
};

#endif // SIGMOID_CUH_INCLUDED
