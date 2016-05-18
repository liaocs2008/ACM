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
        cout << name << " forward " << endl;
        assert(0 == apply_sigmoid(&x, &a) );
    }

    void backward(Mat& a, Mat& d_b, Mat& d_a) {
        cout << name << " backward " << endl;
        assert(0 == mult_elementwise(&a, &a, &d_a, 0)); // d_a = a * a
        assert(0 == subtract_elementwise(&a, &d_a, &d_a)); // d_a = a - d_a = a - a^2
        assert(0 == mult_elementwise(&d_a, &d_b, &d_a, 0)); // d_a = (a - a^2) * d_b
    }

    string name;
};

#endif // SIGMOID_CUH_INCLUDED
