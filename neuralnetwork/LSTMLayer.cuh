#ifndef LSTMLAYER_CUH_INCLUDED
#define LSTMLAYER_CUH_INCLUDED

class LSTMLayer {
public:
    LSTMLayer(int input_size, int hidden_size, const string& layer_name) {
        I = input_size;
        H = hidden_size;
        name = layer_name;

        // initial weights
        assert(0 == init_empty(&wx, I, 4*H));
        assert(0 == fill_with_randn(&rnd, &wx));

        assert(0 == init_empty(&d_wx, I, 4*H));
        assert(0 == assign_scalar(&d_wx, 0));

        assert(0 == init_empty(&wh, H, 4*H));
        assert(0 == fill_with_randn(&rnd, &wh));

        assert(0 == init_empty(&d_wh, H, 4*H));
        assert(0 == assign_scalar(&d_wh, 0));

        // initial biases
        assert(0 == init_empty(&c, 1, 4*H));
        assert(0 == assign_scalar(&c, 0));

        assert(0 == init_empty(&d_c, 1, 4*H));
        assert(0 == assign_scalar(&d_c, 0));
    }

    ~LSTMLayer() {
        assert(0 == free_device_memory(&wx));
        assert(0 == free_device_memory(&d_wx));

        assert(0 == free_device_memory(&wh));
        assert(0 == free_device_memory(&d_wh));

        assert(0 == free_device_memory(&c));
        assert(0 == free_device_memory(&d_c));
    }

    void forward_t(Mat& x, Mat& sc_prev, Mat& h_prev, Mat& state, Mat& sc, Mat& h) {
        //cout << x.size[0] << ", " << x.size[1] << endl;
        assert(0 == dot(&x, &wx, &state, 0, 1)); // a = 0*a + <x, w>
        assert(0 == add_row_vec(&state, &c, &state)); // a = a + c
        assert(0 == dot(&h_prev, &wh, &state, 1, 1)); // a = a + <h_prev, w>

        {
            Mat part;
            assert(0 == get_slice(&state, &part, 0, 3*H)); // IFO
            assert(0 == apply_sigmoid(&part, &part)); // activate IFO

            assert(0 == get_slice(&state, &part, 3*H, 4*H)); //ac
            assert(0 == apply_tanh(&part, &part)); // activate ac
        }

        {
            Mat bI, bF, gAC;
            assert(0 == get_slice(&state, &bI, 0, 1*H));
            assert(0 == get_slice(&state, &bF, 1*H, 2*H));
            assert(0 == get_slice(&state, &gAC, 3*H, 4*H));
            assert(0 == mult_elementwise(&bF, &sc_prev, &sc, 0)); // sc = bF * sc0
            assert(0 == mult_elementwise(&bI, &gAC, &sc, 1)); // sc += bI * gAC
        }

        {
            assert(0 == apply_tanh(&sc, &h)); // h = h(sc)
            Mat bO;
            assert(0 == get_slice(&state, &bO, 2*H, 3*H));
            assert(0 == mult_elementwise(&bO, &h, &h, 0)); // h = h * bO
        }
    }

    void forward(Mat& x, Mat& states, Mat& h, Mat& sc0, Mat& h0, int T) {
        assert(0 == states.size[1] % (5*H)); // memory layout requires
        //cout << name << " forward " << endl;

        Mat sc_prev = sc0; sc_prev.owns_data = 0; // in case deallocation
        Mat h_prev = h0; h_prev.owns_data = 0;
        int input_size = x.size[1] / T;
        //cout << "input_size=" << input_size << endl;
        for (int t = 0; t < T; ++t) {
            Mat s, sct, state; // st for IFOA, sc for current cell state
            assert(0 == get_slice(&states, &s, t*5*H, (t+1)*5*H));
            assert(0 == get_slice(&s, &state, 0, 4*H));
            assert(0 == get_slice(&s, &sct, 4*H, 5*H));

            Mat xt, ht;
            assert(0 == get_slice(&x, &xt, t*input_size, (t+1)*input_size));
            assert(0 == get_slice(&h, &ht, t*H, (t+1)*H));

            forward_t(xt, sc_prev, h_prev, state, sct, ht);
            sc_prev = sct; sc_prev.owns_data = 0; // in case deallocation
            h_prev = ht; h_prev.owns_data = 0;
        }
    }


    // important notice, this function will overwrite x, h, sc
    void backward_t(Mat& d_sc, Mat& d_h, Mat &grad,
                    Mat &x, Mat &sc_prev, Mat &h_prev,
                    Mat& state, Mat& sc, Mat& h) {
        Mat bI, bF, bO, gac;
        assert(0 == get_slice(&state, &bI, 0, 1*H));
        assert(0 == get_slice(&state, &bF, 1*H, 2*H));
        assert(0 == get_slice(&state, &bO, 2*H, 3*H));
        assert(0 == get_slice(&state, &gac, 3*H, 4*H));

        Mat d_aI;
        assert(0 == get_slice(&grad, &d_aI, 0, 1*H));
        {
            assert(0 == sigmoid_deriv_aux(&bI, &d_aI));
            assert(0 == mult_elementwise(&d_aI, &gac, &d_aI, 0));
        }

        Mat d_aF;
        assert(0 == get_slice(&grad, &d_aF, 1*H, 2*H));
        {
            assert(0 == sigmoid_deriv_aux(&bF, &d_aF));
            assert(0 == mult_elementwise(&d_aF, &sc_prev, &d_aF, 0));
        }

        assert(0 == apply_tanh(&sc, &sc)); // important, overwriting sc

        Mat d_aO;
        assert(0 == get_slice(&grad, &d_aO, 2*H, 3*H));
        {
            assert(0 == sigmoid_deriv_aux(&bO, &d_aO));
            assert(0 == mult_elementwise(&d_aO, &sc, &d_aO, 0));
        }

        Mat d_ac;
        assert(0 == get_slice(&grad, &d_ac, 3*H, 4*H));
        {
            assert(0 == apply_tanh_deriv(&bI, &gac, &d_ac));
        }

        // important, overwriting sc again
        assert(0 == apply_tanh_deriv(&bO, &sc, &sc));
        assert(0 == mult_elementwise(&d_h, &sc, &sc, 0));
        assert(0 == add_elementwise(&sc, &d_sc, &d_sc));

        // update grad
        assert(0 == mult_elementwise(&d_sc, &d_aI, &d_aI, 0));
        assert(0 == mult_elementwise(&d_sc, &d_aF, &d_aF, 0));
        assert(0 == mult_elementwise(&d_h, &d_aO, &d_aO, 0));
        assert(0 == mult_elementwise(&d_sc, &d_ac, &d_ac, 0));

        // update d_wx
        set_transpose(&x, 1);
        assert(0 == dot(&x, &grad, &d_wx, 1, 1));
        set_transpose(&x, 0);

        // update d_wh
        set_transpose(&h_prev, 1);
        assert(0 == dot(&h_prev, &grad, &d_wh, 1, 1));
        set_transpose(&h_prev, 0);

        // update d_c
        assert(0 == sum_by_axis(&grad, &d_c, 0, 1, 1));
    }

    void backward(Mat& d_x, Mat& d_h,
                  Mat& x, Mat& states, Mat& h,
                  Mat& sc0, Mat& h0, int T) {
        //cout << name << " backward " << endl;
        // in fact, we need space for d_sct at each time step, but here just overwriting
        Mat sc_prev, h_prev, grad, d_sct;
        init_empty(&grad, x.size[0], 4*H);
        init_empty(&d_sct, x.size[0], H);
        assert(0 == assign_scalar(&d_sct, 0));
        int input_size = x.size[1] / T;
        for (int t = T-1; t >= 0; --t) {
            assert(0 == assign_scalar(&grad, 0));

            Mat s, sct, state; // st for IFOA, sc for current cell state
            assert(0 == get_slice(&states, &s, t*5*H, (t+1)*5*H));
            assert(0 == get_slice(&s, &state, 0, 4*H));
            assert(0 == get_slice(&s, &sct, 4*H, 5*H));

            if (0 == t) {
                assert(0 == get_slice(&sc0, &sc_prev, 0, sc0.size[1]));
                assert(0 == get_slice(&h0, &h_prev, 0, h0.size[1]));
            } else {
                assert(0 == get_slice(&h, &h_prev, (t-1)*H, t*H));
                Mat t_1;
                assert(0 == get_slice(&states, &t_1, (t-1)*5*H, t*5*H));
                assert(0 == get_slice(&t_1, &sc_prev, 4*H, 5*H));
            }

            Mat xt, ht;
            assert(0 == get_slice(&x, &xt, t*input_size, (t+1)*input_size));
            assert(0 == get_slice(&h, &ht, t*H, (t+1)*H));

            Mat d_ht;
            assert(0 == get_slice(&d_h, &d_ht, t*H, (t+1)*H));

            backward_t(d_sct, d_ht, grad, xt, sc_prev, h_prev, state, sct, ht);

            if (t > 0) { // update d_sc, d_h for next time step
                Mat bF;
                assert(0 == get_slice(&state, &bF, 1*H, 2*H));
                assert(0 == mult_elementwise(&d_sct, &bF, &d_sct, 0)); // overwrite d_sct

                Mat d_ht1;
                assert(0 == get_slice(&d_h, &d_ht1, (t-1)*H, t*H));
                set_transpose(&wh, 1);
                assert(0 == dot(&grad, &wh, &d_ht1, 1, 1));
                set_transpose(&wh, 0);
            }

            Mat d_xt;
            assert(0 == get_slice(&d_x, &d_xt, t*input_size, (t+1)*input_size));
            set_transpose(&wx, 1);
            assert(0 == dot(&grad, &wx, &d_xt, 0, 1));
            set_transpose(&wx, 0);
        }
    }

    void update(Dtype lr = 0.01) {
        assert(0 == add_mult(&wx, &d_wx, lr));
        assert(0 == assign_scalar(&d_wx, 0));
        assert(0 == add_mult(&wh, &d_wh, lr));
        assert(0 == assign_scalar(&d_wh, 0));
        assert(0 == add_mult(&c, &d_c, lr));
        assert(0 == assign_scalar(&d_c, 0));
    }

    int I, H;
    string name;

    Mat wx, wh, c;
    Mat d_wx, d_wh, d_c;
};


#endif // LSTMLAYER_CUH_INCLUDED
