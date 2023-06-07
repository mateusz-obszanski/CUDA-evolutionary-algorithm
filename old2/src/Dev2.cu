#include <iostream>
#include <iterator>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/random.h>

// note: functor inherits from unary_function
struct square_root : public thrust::unary_function<float, float> {
    __host__ __device__ float
    operator()(float x) const {
        return sqrtf(x);
    }
};

int
main() {
    // thrust::device_vector<float>             arr1(8);
    // thrust::device_vector<float>             arr2(8);
    // thrust::default_random_engine            rng(42);
    // thrust::uniform_real_distribution<float> d;

    // thrust::generate(arr1.begin(), arr1.end(), [=] __device__() { return d(rng); });
    // thrust::generate(arr2.begin(), arr2.end(), [=] __device__() { return d(rng); });

    // thrust::copy(arr1.begin(), arr1.end(), std::ostream_iterator<float>(std::cout));
    // thrust::copy(arr2.begin(), arr2.end(), std::ostream_iterator<float>(std::cout));

    thrust::device_vector<float> v(4);
    v[0] = 1.0f;
    v[1] = 4.0f;
    v[2] = 9.0f;
    v[3] = 16.0f;

    typedef thrust::device_vector<float>::iterator FloatIterator;

    thrust::transform_iterator<square_root, FloatIterator> iter(v.begin(), square_root());

    *iter;   // returns 1.0f
    iter[0]; // returns 1.0f;
    iter[1]; // returns 2.0f;
    iter[2]; // returns 3.0f;
    iter[3]; // returns 4.0f;

    // iter[4] is an out-of-bounds error

    std::cout << "trying to see fifth value: ";
    std::cout << *(iter + 4) << '\n'; // this throws

    std::cout << "hi!\n";
}
