// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/convolution.hpp>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/global_control.h>
#include <latch>
#include <oneapi/tbb/blocked_range2d.h>

double mticks() {
    typedef std::chrono::high_resolution_clock clock;
    typedef std::chrono::duration<float, std::milli> duration;

    static clock::time_point start = clock::now();
    duration elapsed = clock::now() - start;
    return elapsed.count();
}

namespace {
    constexpr size_t filter_input_ch_axis = 0;

    template <typename T>
    void extend_with_zeros(const ov::Strides& strides,
                           const ov::Shape& input_shape,
                           const T* in,
                           ov::Shape& output_shape,
                           std::vector<T>& input_zeros) {
        std::vector<size_t> input_3d(3, 1);
        std::vector<size_t> strides_3d(3, 1);
        std::vector<size_t> output_3d(3, 1);

        for (size_t i = 0; i < strides.size(); ++i) {
            output_shape[i + 2] = input_shape[i + 2] + (strides[i] - 1) * (input_shape[i + 2] - 1);
            input_3d[input_3d.size() - strides.size() + i] = input_shape[i + 2];
            strides_3d[strides_3d.size() - strides.size() + i] = strides[i];
            output_3d[output_3d.size() - strides.size() + i] = output_shape[i + 2];
        }

        const size_t input_size = ov::shape_size(input_3d);
        if (input_size == 1) {
            for (size_t i = 0; i < shape_size(input_shape); ++i) {
                input_zeros.push_back(in[i]);
            }
        } else {
            for (size_t batch = 0; batch < input_shape[0]; ++batch) {
                const auto offset_batch = batch * input_size * input_shape[1];
                for (size_t channel = 0; channel < input_shape[1]; ++channel) {
                    const auto offset_channel = offset_batch + channel * input_size;
                    for (int i_z = 0; i_z < input_3d[0]; ++i_z) {
                        const auto offset_i_z = i_z * input_3d[2] * input_3d[1];
                        for (int i_y = 0; i_y < input_3d[1]; ++i_y) {
                            const auto offset_i_y = i_y * input_3d[2];
                            for (int i_x = 0; i_x < input_3d[2]; ++i_x) {
                                input_zeros.push_back(in[offset_channel + i_x + offset_i_y + offset_i_z]);

                                if (i_x < input_3d[2] - 1) {
                                    for (int k = 0; k < strides_3d[2] - 1; k++) {
                                        input_zeros.push_back(0);
                                    }
                                }
                            }

                            if (i_y < input_3d[1] - 1) {
                                const auto new_size = output_3d[2] * (strides_3d[1] - 1);
                                input_zeros.insert(input_zeros.begin() + input_zeros.size(), new_size, 0);
                            }
                        }

                        if (i_z < input_3d[0] - 1) {
                            const auto new_size = output_3d[1] * output_3d[2] * (strides_3d[0] - 1);
                            input_zeros.insert(input_zeros.begin() + input_zeros.size(), new_size, 0);
                        }
                    }
                }
            }
        }
    }

    void infer_forward_convbackprop_output_shape(const ov::Shape& in_spatial_shape,
                                                 const ov::Shape& f_spatial_shape,
                                                 const ov::Shape& out_spatial_shape,
                                                 ov::Shape& infer_spatial_shape,
                                                 const ov::Strides& strides,
                                                 const ov::Strides& dilations,
                                                 const ov::CoordinateDiff& output_padding) {
        for (size_t idx = 0; idx < in_spatial_shape.size(); idx++) {
            // FIXME: Incorrect logic with negative pad
            int total_padding =
                    static_cast<int>(strides[idx] * (in_spatial_shape[idx] - 1) + dilations[idx] * (f_spatial_shape[idx] - 1) +
                                     1 - out_spatial_shape[idx] + output_padding[idx]);
            size_t padded_dim = std::max<size_t>(static_cast<size_t>(total_padding), static_cast<size_t>(0));
            size_t filter_dilated_dim = dilations[idx] * (f_spatial_shape[idx] - 1) + 1;
            size_t out_spatial_dim =
                    (in_spatial_shape[idx] - 1) * strides[idx] + filter_dilated_dim - padded_dim + output_padding[idx];
            infer_spatial_shape.push_back(out_spatial_dim);
        }
    }

    inline void validate_convolution_backprop_parameters(const ov::Shape& in_shape,
                                                         const ov::Shape& f_shape,
                                                         const ov::Shape& out_shape,
                                                         const ov::Strides& strides,
                                                         const ov::Strides& dilations,
                                                         const ov::CoordinateDiff& pads_begin,
                                                         const ov::CoordinateDiff& pads_end,
                                                         const ov::CoordinateDiff& output_padding) {
        // this implementation supports 1D, 2D and 3D convolutions
        NGRAPH_CHECK(in_shape.size() >= 3 && in_shape.size() <= 5, "Unsupported input rank: ", in_shape);

        NGRAPH_CHECK(in_shape.size() == f_shape.size(),
                     "Incompatible input ranks: ",
                     in_shape.size(),
                     " and ",
                     f_shape.size());

        NGRAPH_CHECK(in_shape[ngraph::runtime::reference::in_channel_axis] == f_shape[filter_input_ch_axis],
                     "Incompatible input channels in data batch and filters shapes: ",
                     in_shape[ngraph::runtime::reference::in_channel_axis],
                     " and ",
                     f_shape[filter_input_ch_axis]);

        NGRAPH_CHECK(in_shape.size() == out_shape.size(),
                     "Incompatible input and output ranks: ",
                     in_shape.size(),
                     " and ",
                     out_shape.size());

        const auto spatial_dims = in_shape.size() - 2;
        NGRAPH_CHECK(strides.size() == spatial_dims, "Strides not definied for all and only spatial dimensions.");

        NGRAPH_CHECK(dilations.size() == spatial_dims, "Dilations not defined for all and only spatial dimensions.");

        NGRAPH_CHECK((pads_begin.size() == pads_end.size()) && (pads_begin.size() == spatial_dims),
                     "Pads not defined for all and only spatial dimensions.");

        NGRAPH_CHECK(!output_padding.empty() && output_padding.size() == spatial_dims,
                     "Output padding not defined for all and only spatial dimensions.");

        ov::Shape out_spatial_shape{std::next(out_shape.begin(), 2), std::end(out_shape)};
        ov::Shape infered_out_spatial_shape{};
        infer_forward_convbackprop_output_shape(ov::Shape{std::next(in_shape.begin(), 2), std::end(in_shape)},
                                                ov::Shape{std::next(f_shape.begin(), 2), std::end(f_shape)},
                                                ov::Shape{std::next(out_shape.begin(), 2), std::end(out_shape)},
                                                infered_out_spatial_shape,
                                                strides,
                                                dilations,
                                                output_padding);
        NGRAPH_CHECK(out_spatial_shape == infered_out_spatial_shape, "Incorrect output shape provided");
    }
}  // namespace

//template <typename T>
//void arm_convolve_3D_channels(const ngraph::runtime::reference::ConvolutionParams& p,
//                          const T* batch,
//                          const ov::Shape& batch_shape,
//                          const T* filter,
//                          const ov::Shape& filter_shape,
//                          T*& out) {
//    const int input_size_z = static_cast<int>(batch_shape[1]);
//    const int input_size_y = static_cast<int>(batch_shape[2]);
//    const int input_size_x = static_cast<int>(batch_shape[3]);
//    const int filter_size_z = static_cast<int>(filter_shape[1]);
//    const int filter_size_y = static_cast<int>(filter_shape[2]);
//    const int filter_size_x = static_cast<int>(filter_shape[3]);
//    const int dilated_filter_size_z = static_cast<int>(filter_size_z + (filter_size_z - 1) * (p.dilation[0] - 1));
//    const int dilated_filter_size_y = static_cast<int>(filter_size_y + (filter_size_y - 1) * (p.dilation[1] - 1));
//    const int dilated_filter_size_x = static_cast<int>(filter_size_x + (filter_size_x - 1) * (p.dilation[2] - 1));
//
//    const ov::Shape input_channel_shape(++batch_shape.begin(), batch_shape.end());
//    const size_t input_channel_size = shape_size(input_channel_shape);
//    const ov::Shape filter_channel_shape(++filter_shape.begin(), filter_shape.end());
//    const size_t filter_channel_size = shape_size(filter_channel_shape);
//
//
//    int i_z_begin = static_cast<int>(-p.pads_begin[0]);
//    int i_z_end = static_cast<int>(p.pads_end[0] + input_size_z - dilated_filter_size_z + p.output_padding[0]);
//    int i_z_delta = static_cast<int>(p.strides[0]);
//
//    int i_y_begin = static_cast<int>(-p.pads_begin[1]);
//    int i_y_end = static_cast<int>(p.pads_end[1] + input_size_y - dilated_filter_size_y + p.output_padding[1]);
//    int i_y_delta = static_cast<int>(p.strides[1]);
//
//    int i_x_begin = static_cast<int>(-p.pads_begin[2]);
//    int i_x_end = static_cast<int>(p.pads_end[2] + input_size_x - dilated_filter_size_x + p.output_padding[2]);
//    int i_x_delta = static_cast<int>(p.strides[2]);
////    std::cout << i_x_begin << " " << i_x_end << " " << i_x_delta << std::endl;
////    std::cout << i_y_begin << " " << i_y_end << " " << i_y_delta << std::endl;
////    std::cout << i_z_begin << " " << i_z_end << " " << i_z_delta << std::endl;
//    int i_z = i_z_begin, i_z_count = (i_z_end - i_z_begin) / i_z_delta + 1;
//    int i_y = i_y_begin, i_y_count = (i_y_end - i_y_begin) / i_y_delta + 1;
//    int i_x = i_x_begin, i_x_count = (i_x_end - i_x_begin) / i_x_delta + 1;
//    std::mutex m;
//    std::vector<T> sum(i_x_count * i_y_count * i_z_count);
//
//    std::cout << oneapi::tbb::this_task_arena::max_concurrency() << std::endl;
//    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, i_x_count * i_y_count * i_z_count),
//            [&](oneapi::tbb::blocked_range<int> r) {
//                    for (int i_index = r.begin(); i_index < r.end(); i_index++) {
////                    for (int i_index = 0; i_index < i_x_count * i_y_count * i_z_count; i_index++) {
//                        auto input_channel = batch;
//                        auto filter_channel = filter;
//                        sum[i_index] = 0;
////                        std::mutex m;
////                        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, filter_shape[0]),
////                            [&](oneapi::tbb::blocked_range<int> r) {
////                                for (size_t filter_channels_count = r.begin();  filter_channels_count < r.end(); filter_channels_count++) {
//                                for (size_t filter_channels_count = 0;  filter_channels_count < filter_shape[0]; filter_channels_count++) {
//                                for (int f_z = 0; f_z < filter_size_z; ++f_z) {
//                                    for (int f_y = 0; f_y < filter_size_y; ++f_y) {
//                                        for (int f_x = 0; f_x < filter_size_x; ++f_x) {
//                                            int rel_i_z = i_z + (f_z * static_cast<int>(p.dilation[0]));
//                                            int rel_i_y = i_y + (f_y * static_cast<int>(p.dilation[1]));
//                                            int rel_i_x = i_x + (f_x * static_cast<int>(p.dilation[2]));
//
//                                            bool padding =
//                                                    !(ngraph::runtime::reference::in_range(rel_i_x, {0, input_size_x}) &&
//                                                      ngraph::runtime::reference::in_range(rel_i_y, {0, input_size_y}) &&
//                                                      ngraph::runtime::reference::in_range(rel_i_z, {0, input_size_z}));
//                                            if (!padding) {
//                                                int f_buf_idx =
//                                                        (f_z * filter_size_y * filter_size_x) + (f_y * filter_size_x) + f_x;
//                                                int i_buf_idx = (rel_i_z * input_size_y * input_size_x) +
//                                                                (rel_i_y * input_size_x) + rel_i_x;
//                                                auto tmp_input_channel = input_channel + filter_channels_count * input_channel_size;
//                                                auto tmp_filter_channel = filter_channel + filter_channels_count * filter_channel_size;
//                                                sum[i_index] += static_cast<T>(tmp_input_channel[i_buf_idx]) *
//                                                       static_cast<T>(tmp_filter_channel[f_buf_idx]);
//                                            }
//                                        }
//                                    }
//                                }
//                            }
//                        i_x += i_x_delta;
//                        if (i_x > i_x_end) { i_y += i_y_delta; i_x = i_x_begin; }
//                        if (i_y > i_y_end) { i_z += i_z_delta; i_y = i_y_begin; }
//                    }
//            });
//    for (int i_index = 0; i_index < i_x_count * i_y_count * i_z_count; i_index++) {
//        *out = sum[i_index];
//        ++out;
//    }
////        });
//}

template <typename T>
void arm_convolve_3D_channels(const ngraph::runtime::reference::ConvolutionParams& p,
                          const T* batch,
                          const ov::Shape& batch_shape,
                          const T* filter,
                          const ov::Shape& filter_shape,
                          T*& out) {
    const int input_size_z = static_cast<int>(batch_shape[1]);
    const int input_size_y = static_cast<int>(batch_shape[2]);
    const int input_size_x = static_cast<int>(batch_shape[3]);
    const int filter_size_z = static_cast<int>(filter_shape[1]);
    const int filter_size_y = static_cast<int>(filter_shape[2]);
    const int filter_size_x = static_cast<int>(filter_shape[3]);
    const int dilated_filter_size_z = static_cast<int>(filter_size_z + (filter_size_z - 1) * (p.dilation[0] - 1));
    const int dilated_filter_size_y = static_cast<int>(filter_size_y + (filter_size_y - 1) * (p.dilation[1] - 1));
    const int dilated_filter_size_x = static_cast<int>(filter_size_x + (filter_size_x - 1) * (p.dilation[2] - 1));

    const ov::Shape input_channel_shape(++batch_shape.begin(), batch_shape.end());
    const size_t input_channel_size = shape_size(input_channel_shape);
    const ov::Shape filter_channel_shape(++filter_shape.begin(), filter_shape.end());
    const size_t filter_channel_size = shape_size(filter_channel_shape);
 std::mutex m;
    for (int i_z = static_cast<int>(-p.pads_begin[0]);
            i_z <= static_cast<int>(p.pads_end[0] + input_size_z - dilated_filter_size_z + p.output_padding[0]);
            i_z += static_cast<int>(p.strides[0])) {
        int i_y_begin = static_cast<int>(-p.pads_begin[1]);
        int i_y_end = static_cast<int>(p.pads_end[1] + input_size_y - dilated_filter_size_y + p.output_padding[1]);
        int i_y_delta = static_cast<int>(p.strides[1]);
        int i_y_count = (i_y_end - i_y_begin) / i_y_delta + 1;

        int i_x_begin = static_cast<int>(-p.pads_begin[2]);
        int i_x_end = static_cast<int>(p.pads_end[2] + input_size_x - dilated_filter_size_x + p.output_padding[2]);
        int i_x_delta = static_cast<int>(p.strides[2]);
        int i_x_count = (i_x_end - i_x_begin) / i_x_delta + 1;

        std::vector<std::vector<T>> sum(i_y_count, std::vector<T>(i_x_count, 0));
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range2d<int, int>(i_y_begin, i_y_end + 1, i_x_begin, i_x_end + 1),
            [&](oneapi::tbb::blocked_range2d<int> r) {
        for (int i_y = r.rows().begin(); i_y < r.rows().end(); i_y += i_y_delta) {
//        for (int i_y = i_y_begin; i_y <= i_y_end; i_y += i_y_delta) {
            int i_y_ = (i_y - i_y_begin) / i_y_delta;
            for (int i_x = r.cols().begin(); i_x < r.cols().end(); i_x += i_x_delta) {
//            for (int i_x = i_x_begin; i_x <= i_x_end; i_x += i_x_delta) {
                int i_x_ = (i_x - i_x_begin) / i_x_delta;
                auto input_channel = batch;
                auto filter_channel = filter;
                size_t filter_channels_count = filter_shape[0];
                while (filter_channels_count--) {
                    for (int f_z = 0; f_z < filter_size_z; ++f_z) {
                        for (int f_y = 0; f_y < filter_size_y; ++f_y) {
                            for (int f_x = 0; f_x < filter_size_x; ++f_x) {
                                int rel_i_z = i_z + (f_z * static_cast<int>(p.dilation[0]));
                                int rel_i_y = i_y + (f_y * static_cast<int>(p.dilation[1]));
                                int rel_i_x = i_x + (f_x * static_cast<int>(p.dilation[2]));

                                bool padding =
                                        !(ngraph::runtime::reference::in_range(rel_i_x, {0, input_size_x}) &&
                                          ngraph::runtime::reference::in_range(rel_i_y, {0, input_size_y}) &&
                                          ngraph::runtime::reference::in_range(rel_i_z, {0, input_size_z}));
                                if (padding)
                                    continue;

                                int f_buf_idx = (f_z * filter_size_y * filter_size_x) + (f_y * filter_size_x) + f_x;
                                int i_buf_idx =
                                        (rel_i_z * input_size_y * input_size_x) + (rel_i_y * input_size_x) + rel_i_x;
                                auto tmp_input_channel = input_channel + filter_channels_count * input_channel_size;
                                auto tmp_filter_channel = filter_channel + filter_channels_count * filter_channel_size;
                                sum[i_y_][i_x_] += static_cast<T>(tmp_input_channel[i_buf_idx]) *
                                             static_cast<T>(tmp_filter_channel[f_buf_idx]);
                            }
                        }
                    }
                }
            }
        }
            });
        for (int i_y_ = 0; i_y_ < sum.size(); i_y_++) {
            for (int i_x_ = 0; i_x_ < sum[0].size(); i_x_++) {
                *out = sum[i_y_][i_x_];
                ++out;
            }
        }
    }
}

template <typename T>
void convolution_backprop_impl(const T* in,
                               const T* f,
                               T* out,
                               const ov::Shape& in_shape,
                               const ov::Shape& f_shape,
                               const ov::Shape& out_shape,
                               const ov::Strides& strides,
                               const ov::Strides& dilation,
                               const ov::CoordinateDiff& pads_begin,
                               const ov::CoordinateDiff& pads_end,
                               const ov::CoordinateDiff& output_padding) {
    // here we are converting all param types to int's to avoid arithmetic issues
    // (e.g signed + unsigned) in indexes calculation later
    ngraph::runtime::reference::ConvolutionParams params{strides, dilation, pads_begin, pads_end, output_padding};

    // here we are extending spatial dimensions to 3D, because we are going to use 3D
    // convolution implementation to convolve also in 1D & 2D case
    ov::Shape input_shape{in_shape};
    ov::Shape filters_shape{f_shape};
    if (in_shape.size() < 5) {
        extend_to_3D(params, input_shape, filters_shape);
    }

    for (size_t i = 0; i < input_shape.size() - 2; ++i) {
        if (input_shape[i + 2] > 1 || filters_shape[i + 2] > 1) {
            params.pads_begin[i] = filters_shape[i + 2] - params.pads_begin[i] - 1;
            params.pads_end[i] = filters_shape[i + 2] - params.pads_end[i] - 1;
        } else {
            params.pads_begin[i] = 0;
            params.pads_end[i] = 0;
        }
    }

    // convert output shape to 3D, contains only dimensions
    ov::Shape out_shape_3d{out_shape.begin() + 2, out_shape.end()};

    int out_shape_rank = static_cast<int>(out_shape.size()) - 2;
    if (out_shape_rank < 3) {
        int missing_dims = 3 - out_shape_rank;
        out_shape_3d.insert(std::prev(out_shape_3d.end(), out_shape_rank), missing_dims, 1);
    }

    // modify params.pads_end when output_shape was provided in ctor in order to
    // calculate expected number of output elements
    for (size_t i = 0; i < out_shape_3d.size(); i++) {
        if (out_shape_3d[i] > 1) {
            // expected_dim = (in - 1)* strides + filter - 2*padding + out_padding
            // strides is already applied (through 0's extension in input)
            // padding = pads_begin + pads_end, formula below is using
            // params.pad_begin/params.pads_end:
            const size_t expected_dim =
                    out_shape_3d[i] - ((input_shape[i + 2] - 1) - filters_shape[i + 2] + params.pads_begin[i] +
                                       params.pads_end[i] + 2 + params.output_padding[i]);
            params.pads_end[i] += expected_dim;
        }
    }

    const size_t filters_count = filters_shape[ngraph::runtime::reference::filter_out_ch_axis];
    const ov::Shape filter_shape(++filters_shape.begin(), filters_shape.end());
    const size_t filter_size = shape_size(filter_shape);

    const size_t batches_count = input_shape[ngraph::runtime::reference::in_batch_axis];
    ov::Shape batch_shape(++input_shape.begin(), input_shape.end());
    const size_t batch_size = shape_size(batch_shape);

    auto batch = in;


    std::mutex filterMutex;
//    auto t0_0 = mticks();


    for (size_t batch_idx = 0; batch_idx < batches_count; ++batch_idx) {
//        filterMutex.lock();
//        auto filter = f;
//        filterMutex.unlock();
//        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, filters_count),
//                          [&](oneapi::tbb::blocked_range<int> r) {
//                              for (size_t f_idx = r.begin(); f_idx < r.end(); ++f_idx) {
//                                  convolve_3D_channels(params, batch, batch_shape, filter, filter_shape, out);
//                                  filter += filter_size;
//                              }
//                          });
        auto filter = f;
//        oneapi::tbb::task_group gr;
        for (size_t f_idx = 0; f_idx < filters_count; ++f_idx) {
            arm_convolve_3D_channels(params, batch, batch_shape, filter, filter_shape, out);
            filter += filter_size;
        }
        batch += batch_size;
    }
//    std::cout  << "t0_0 = " << mticks() - t0_0 << std::endl;
}

template <typename T>
void convolution_backprop_in(const T* delta_in,
                             const T* filter,
                             T* delta_out,
                             const ov::Shape& in_shape,
                             const ov::Shape& filter_shape,
                             const ov::Shape& out_shape,
                             const ov::Strides& in_dilation,
                             const ov::Strides& filter_dilation,
                             const ov::CoordinateDiff& forward_in_pad_bellow,
                             const ov::CoordinateDiff& forward_in_pad_above,
                             const ov::Strides& stride,
                             const ov::CoordinateDiff& output_padding) {
//    auto t0 = mticks();
//    auto t3 = mticks();
    std::vector<T> extended_input;
    std::vector<T> extended_filter;
    ov::AxisSet reverse_axes;

    ov::Shape conv_input_shape = in_shape;
    ov::Shape conv_filter_shape = filter_shape;
    ov::Strides conv_stride = stride;
    ov::Strides conv_filter_dilation = filter_dilation;
    auto conv_input_data = delta_in;

    validate_convolution_backprop_parameters(in_shape,
                                             filter_shape,
                                             out_shape,
                                             stride,
                                             filter_dilation,
                                             forward_in_pad_bellow,
                                             forward_in_pad_above,
                                             output_padding);

    // Note that we only reverse the spatial dimensions here (loop
    // starts at 2)
    std::vector<T> reversed(shape_size(filter_shape));
    for (size_t i = 2; i < filter_shape.size(); ++i) {
        reverse_axes.insert(i);
    }
    ngraph::runtime::reference::reverse(reinterpret_cast<const char*>(filter),
            reinterpret_cast<char*>(&reversed[0]),
            filter_shape,
            filter_shape,
            reverse_axes,
            sizeof(T));
//    std::cout << "t3 = " << mticks() - t3 << std::endl;
//    auto t2 = mticks();
    auto conv_filter_data = &reversed[0];

    // if channel number for output is > 1 then reverse layout of filter coefficients as
    // it is required by convolve_3D_channels() function.
    // Current layout:
    // batch0_ch0|batch0_ch1|...|batch0_chN|...|batch1_ch0|batch1_ch1|...|batch1_chN|...
    // Expected layout:
    // batch0_ch0|batch1_ch0|...|batchN_ch0|...|batch0_ch1|batch1_ch1|...|batch1_chN|...
    if (filter_shape[1] > 1) {
        std::vector<T> temp_reversed(reversed);
        const ov::Shape filter_dim_shape(filter_shape.begin() + 2, filter_shape.end());
        const size_t filter_size = shape_size(filter_dim_shape);

        for (size_t i = 0; i < filter_shape[1]; i++) {
            for (size_t j = 0; j < filter_shape[0]; j++) {
                const auto delta = temp_reversed.begin() + j * filter_shape[1] * filter_size + i * filter_size;
                const auto out = reversed.begin() + i * filter_shape[0] * filter_size + j * filter_size;
                std::copy(delta, delta + filter_size, out);
            }
        }
    }
//    std::cout << "t2 = " << mticks() - t2 << std::endl;

//    auto t1 = mticks();
    // swap filter batch and channels
    std::iter_swap(conv_filter_shape.begin(), conv_filter_shape.begin() + 1);

    // extend stride and filter inputs with zero padding for stride and filter_dilation
    // > 1, after that set stride and filter params to 1.
    const size_t stride_dim = std::accumulate(stride.begin(), stride.end(), int64_t(1), std::multiplies<int64_t>());
    if (stride_dim >= 2) {
        extend_with_zeros(stride, in_shape, delta_in, conv_input_shape, extended_input);
        std::fill(conv_stride.begin(), conv_stride.end(), 1);
        conv_input_data = &extended_input[0];
    }
    const size_t dilation_dim =
            std::accumulate(filter_dilation.begin(), filter_dilation.end(), uint64_t(1), std::multiplies<size_t>());
    if (dilation_dim >= 2) {
        extend_with_zeros<T>(filter_dilation,
                             filter_shape,
                             reinterpret_cast<const T*>(&reversed[0]),
                             conv_filter_shape,
                             extended_filter);
        std::fill(conv_filter_dilation.begin(), conv_filter_dilation.end(), 1);
        conv_filter_data = &extended_filter[0];
    }
//    std::cout << "t1 = " << mticks() - t1 << std::endl;

//    auto t0 = mticks();
    convolution_backprop_impl(conv_input_data,
                              conv_filter_data,
                              delta_out,
                              conv_input_shape,
                              conv_filter_shape,
                              out_shape,
                              conv_stride,
                              conv_filter_dilation,
                              forward_in_pad_bellow,
                              forward_in_pad_above,
                              output_padding);
//    std::cout  << "t0 = " << mticks() - t0 << std::endl;
}

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ConvolutionBackpropData& node) {
    auto make = [&] (auto refFunction) {
        auto out_shape = node.get_shape();
        ngraph::Strides in_dilation(std::vector<size_t>(node.get_input_shape(0).size() - 2));
        std::fill(in_dilation.begin(), in_dilation.end(), 1);
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    out_shape,
                                    in_dilation,
                                    node.get_dilations(),
                                    node.get_pads_begin(),
                                    node.get_pads_end(),
                                    node.get_strides(),
                                    node.get_output_padding());
    };

    return CallSwitch(
        AP_WRAP(make, convolution_backprop_in),
        node.input(0), floatTypes);
}
}  //  namespace ArmPlugin
