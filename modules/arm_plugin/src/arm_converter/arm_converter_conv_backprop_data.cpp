// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <arm_compute/runtime/NEON/functions/NEDeconvolutionLayer.h>
#include <arm_compute/runtime/NEON/NEScheduler.h>
#include "arm_converter/arm_converter.hpp"
#include <utility>

namespace ArmPlugin {

enum DeconvInput {Features, Weights, Bias};
template<typename Deconv>
static auto DeconvParameters(const Deconv& node) {
    unsigned int pad_l    = node.get_pads_begin().at(D2::W);
    unsigned int pad_r    = node.get_pads_end().at(D2::W);
    unsigned int pad_t    = node.get_pads_begin().at(D2::H);
    unsigned int pad_b    = node.get_pads_end().at(D2::H);
    unsigned int stride_x = node.get_strides().at(D2::W);
    unsigned int stride_y = node.get_strides().at(D2::H);

    return std::make_pair(
            arm_compute::PadStrideInfo {stride_x, stride_y, pad_l, pad_r, pad_t, pad_b, arm_compute::DimensionRoundingType::FLOOR},
            arm_compute::Size2D {node.get_dilations().at(D2::W), node.get_dilations().at(D2::H)});
}

struct NEDeconvolutionLayerQI final: public arm_compute::IFunction {
public:
    explicit NEDeconvolutionLayerQI(std::shared_ptr<arm_compute::IMemoryManager> memory_manager = nullptr):
        _memory_manager(std::move(memory_manager)),
        _input(nullptr), _weights(nullptr), _bias(nullptr), _output(nullptr) {}
    NEDeconvolutionLayerQI(const NEDeconvolutionLayerQI &) = delete;
    NEDeconvolutionLayerQI(NEDeconvolutionLayerQI &&) = delete;
    NEDeconvolutionLayerQI &operator=(const NEDeconvolutionLayerQI &) = delete;
    NEDeconvolutionLayerQI &operator=(NEDeconvolutionLayerQI &&) = delete;
    ~NEDeconvolutionLayerQI() override = default;
    void configure(arm_compute::ITensor *input, arm_compute::ITensor *weights, arm_compute::ITensor *biases, arm_compute::ITensor *output,
                   const arm_compute::PadStrideInfo &conv_info, const arm_compute::WeightsInfo &weights_info = arm_compute::WeightsInfo(),
                   const arm_compute::Size2D &dilation = arm_compute::Size2D(1U, 1U),
                   const arm_compute::ActivationLayerInfo &act_info = arm_compute::ActivationLayerInfo(),
                   bool enable_fast_math = false, unsigned int num_groups = 1) {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
        ARM_COMPUTE_ERROR_THROW_ON(NEDeconvolutionLayerQI::validate(input->info(), weights->info(),
                                                                    ((biases != nullptr) ? biases->info() : nullptr),
                                                                    output->info()));
        _input = input;
        _weights = weights;
        _bias = biases;
        _output = output;
        _deconv = std::make_unique<arm_compute::NEDeconvolutionLayer>(_memory_manager);
        _deconv->configure(const_cast<arm_compute::ITensor *>(_input), _weights, _bias, _output, conv_info);
    }

    static arm_compute::Status validate(const arm_compute::ITensorInfo *input, const arm_compute::ITensorInfo *weights,
                                        const arm_compute::ITensorInfo *biases, const arm_compute::ITensorInfo *output,
                                        const arm_compute::PadStrideInfo &conv_info,
                                        const arm_compute::WeightsInfo &weights_info = arm_compute::WeightsInfo(),
                                        const arm_compute::Size2D &dilation = arm_compute::Size2D(1U, 1U),
                                        const arm_compute::ActivationLayerInfo &act_info = arm_compute::ActivationLayerInfo(),
                                        bool enable_fast_math = false, unsigned int num_groups = 1) {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, biases, output);
        return arm_compute::NEDeconvolutionLayer::validate(input, weights, biases, output, conv_info);
    }
    void run() override {
        ARM_COMPUTE_ERROR_ON_MSG(!_deconv.get(), "Kernel didn't configured");
        _deconv->run();
    }

protected:
    std::shared_ptr<arm_compute::IMemoryManager>        _memory_manager;
    const arm_compute::ITensor                          *_input;
    arm_compute::ITensor                                *_weights;
    arm_compute::ITensor                                *_bias;
    arm_compute::ITensor                                *_output;
    std::unique_ptr<arm_compute::NEDeconvolutionLayer>  _deconv;
};

//template<> Converter::Conversion::Ptr Converter::Convert(const opset::ConvolutionBackpropData& node) {
//    auto make = [&] (auto refFunction) {
//        auto out_shape = node.get_shape();
//        ngraph::Strides in_dilation(std::vector<size_t>(node.get_input_shape(0).size() - 2));
//        std::fill(in_dilation.begin(), in_dilation.end(), 1);
//        return this->MakeConversion(refFunction,
//                                    node.input(0),
//                                    node.input(1),
//                                    node.output(0),
//                                    node.get_input_shape(0),
//                                    node.get_input_shape(1),
//                                    out_shape,
//                                    in_dilation,
//                                    node.get_dilations(),
//                                    node.get_pads_begin(),
//                                    node.get_pads_end(),
//                                    node.get_strides(),
//                                    node.get_output_padding());
//    };
//
//    return CallSwitch(
//        AP_WRAP(make, ngraph::runtime::reference::convolution_backprop_in),
//        node.input(0), floatTypes);
//}

static arm_compute::ActivationLayerInfo GetActivationInfo(const ngraph::Node& node) {
    auto itInfo = node.get_rt_info().find("ActivationLayerInfo");
    if (itInfo != node.get_rt_info().end()) {
        return itInfo->second.as<arm_compute::ActivationLayerInfo>();
    } else {
        return {};
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmDeconvolution& node) {
    arm_compute::PadStrideInfo conv_info;
    arm_compute::Size2D dilation;
    std::tie(conv_info, dilation) = DeconvParameters(node);

    return MakeConversion<NEDeconvolutionLayerQI>(
            node.input(Features), node.input(Weights), nullptr, node.output(0),
            conv_info, arm_compute::WeightsInfo{}, dilation, GetActivationInfo(node));
}

}  //  namespace ArmPlugin
