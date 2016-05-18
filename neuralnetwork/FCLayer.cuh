#ifndef FCLAYER_CUH_INCLUDED
#define FCLAYER_CUH_INCLUDED

#include "common.cuh"

class FCLayer {
 public:
    FCLayer(size_t input_size, size_t hidden_size, const string& layer_name) {
        I = input_size;
        H = hidden_size;
        name = layer_name;

        // initial weights
        assert(0 == init_empty(&w, I, H));
        assert(0 == fill_with_randn(&rnd, &w));

        assert(0 == init_empty(&d_w, I, H));
        assert(0 == assign_scalar(&d_w, 0));

        // initial biases
        assert(0 == init_empty(&c, 1, H));
        assert(0 == assign_scalar(&c, 0));

        assert(0 == init_empty(&d_c, 1, H));
        assert(0 == assign_scalar(&d_c, 0));
    }

    ~FCLayer() {
        assert(0 == free_device_memory(&w));
        assert(0 == free_device_memory(&d_w));
        assert(0 == free_device_memory(&c));
        assert(0 == free_device_memory(&d_c));
    }

    void forward(Mat& x, Mat& a) {
        assert(x.size[1] == I && a.size[1] == H);
        cout << name << " forward " << endl;
        assert(0 == dot(&x, &w, &a, 0, 1)); // a = 0*a + <x, w>
        assert(0 == add_row_vec(&a, &c, &a)); // a = a + c
    }

    void backward(Mat& x, Mat& d_a, Mat& d_x) {
        // x = (B, I), d_a = (B, H), d_x = (B, I)
        assert(x.size[1]==I && d_a.size[1]==H && d_x.size[0]==x.size[0]);
        cout << name << " backward " << endl;

        set_transpose(&x, 1);
        assert(0 == dot(&x, &d_a, &d_w, 0, 1)); // d_w = <x^T, d_a>
        assert(0 == sum_by_axis(&d_a, &d_c, 0, 1, 0));
        set_transpose(&x, 0);

        set_transpose(&w, 1);
        assert(0 == dot(&d_a, &w, &d_x, 0, 1)); // d_x = <d_a, w^T>
        set_transpose(&w, 0);
    }

    void update(Dtype lr = 0.01) {
        assert(0 == add_mult(&w, &d_w, lr));
        assert(0 == assign_scalar(&d_w, 0));
        assert(0 == add_mult(&c, &d_c, lr));
        assert(0 == assign_scalar(&d_c, 0));
    }

    string name;
    size_t I, H;

    Mat w, d_w;
    Mat c, d_c;
};





#endif // FCLAYER_CUH_INCLUDED
