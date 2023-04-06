// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "extensions/bi3d/Bi3D.hpp"

#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp" 
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace cvcore {

namespace detail {

// Function to bind a cuda stream with cid into downstream message
gxf_result_t BindCudaStream(gxf::Entity& entity, gxf_uid_t cid) {
  if (cid == kNullUid) {
    GXF_LOG_ERROR("stream_cid is null");
    return GXF_FAILURE;
  }
  auto output_stream_id = entity.add<gxf::CudaStreamId>("stream");
  if (!output_stream_id) {
    GXF_LOG_ERROR("failed to add cudastreamid.");
    return output_stream_id.error();
  }
  output_stream_id.value()->stream_cid = cid;
  return GXF_SUCCESS;
}

// Function to record a new cuda event
gxf_result_t BindCudaEvent(gxf::Entity & entity,
                           gxf::Handle<gxf::CudaStream> & stream) {

  cudaEvent_t event_id;
  cudaEventCreateWithFlags(&event_id, 0);
  gxf::CudaEvent event;
  event.initWithEvent(event_id, stream->dev_id(), [](auto){});

  auto ret = stream->record(event.event().value(),
                            [event = event_id, entity = entity.clone().value()](auto) {
                              cudaEventDestroy(event);
                            });
  if (!ret) {
    GXF_LOG_ERROR("record event failed");
    return ret.error();
  }

  return GXF_SUCCESS;
}

} // namespace detail

gxf_result_t Bi3D::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(left_image_name_, "left_image_name", "The name of the left image to be received", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &=
    registrar->parameter(right_image_name_, "right_image_name", "The name of the right image to be received", "",
                         gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(output_name_, "output_name",
                                 "The name of the tensor to be passed to next node", "");
  result &= registrar->parameter(stream_pool_, "stream_pool", "cuda stream pool", "cuda stream pool object",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(pool_, "pool", "Memory pool for allocating output data", "");
  result &= registrar->parameter(left_image_receiver_, "left_image_receiver",
                                 "Receiver to get the left image", "");
  result &= registrar->parameter(right_image_receiver_, "right_image_receiver",
                                 "Receiver to get the right image", "");
  result &= registrar->parameter(disparity_receiver_, "disparity_receiver",
                                 "Optional receilver for dynamic disparity input", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(output_transmitter_, "output_transmitter", "Transmitter to send the data", 
                                 "");

  result &= registrar->parameter(image_type_, "image_type", "Type of input image: BGR_U8 or RGB_U8", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(pixel_mean_, "pixel_mean", "The mean for each channel", "");
  result &= registrar->parameter(normalization_, "normalization", "The normalization for each channel", "");
  result &=
    registrar->parameter(standard_deviation_, "standard_deviation",
                         "The standard deviation for each channel", "");

  result &= registrar->parameter(max_batch_size_, "max_batch_size", "The max batch size to run inference on",
                                 "");
  result &= registrar->parameter(input_layer_width_, "input_layer_width", "The model input layer width", "");
  result &= registrar->parameter(input_layer_height_, "input_layer_height", "The model input layer height",
                                 "");
  result &= registrar->parameter(model_input_type_, "model_input_type",
                                 "The model input image: BGR_U8 or RGB_U8", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

  result &= registrar->parameter(featnet_engine_file_path_, "featnet_engine_file_path",
                                 "The path to the serialized TRT engine of featnet", "");
  result &=
    registrar->parameter(featnet_input_layers_name_, "featnet_input_layers_name",
                         "The names of the input layers", "");
  result &= registrar->parameter(featnet_output_layers_name_, "featnet_output_layers_name",
                                 "The names of the output layers", "");

  result &= registrar->parameter(featnet_hr_engine_file_path_, "featnet_hr_engine_file_path",
                                 "The path to the serialized TRT engine of high-resolution featnet", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(featnet_hr_input_layers_name_, "featnet_hr_input_layers_name",
                                 "The names of the input layers", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(featnet_hr_output_layers_name_, "featnet_hr_output_layers_name",
                                 "The names of the output layers", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

  result &= registrar->parameter(refinenet_engine_file_path_, "refinenet_engine_file_path",
                                 "The path to the serialized TRT engine of refinenet", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(refinenet_input_layers_name_, "refinenet_input_layers_name",
                                 "The names of the input layers", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(refinenet_output_layers_name_, "refinenet_output_layers_name",
                                 "The names of the output layers", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

  result &= registrar->parameter(segnet_engine_file_path_, "segnet_engine_file_path",
                                 "The path to the serialized TRT engine of segnet", "");
  result &=
    registrar->parameter(segnet_input_layers_name_, "segnet_input_layers_name",
                         "The names of the input layers", "");
  result &=
    registrar->parameter(segnet_output_layers_name_, "segnet_output_layers_name",
                         "The names of the output layers", "");

  result &= registrar->parameter(engine_type_, "engine_type", "The type of engine to be run", "");
  result &= registrar->parameter(apply_sigmoid_, "apply_sigmoid", "Whether to apply sigmoid operation", "");
  result &= registrar->parameter(apply_thresholding_, "apply_thresholding",
                                 "Whether to apply threshold operation", "");
  result &= registrar->parameter(threshold_value_low_, "threshold_value_low",
                                 "Low value set by thresholding operation", "");
  result &= registrar->parameter(threshold_value_high_, "threshold_value_high",
                                 "High value set by thresholding operation", "");
  result &= registrar->parameter(threshold_, "threshold",
                                 "Pixel value used by thresholding operation that casts to low or high", "");
  result &=
    registrar->parameter(apply_edge_refinement_, "apply_edge_refinement",
                         "Whether to apply edge refinement", "", false);

  result &= registrar->parameter(max_disparity_levels_, "max_disparity_levels",
                                 "The maximum number of output disparity levels", "");
  result &= registrar->parameter(disparity_values_, "disparity_values", "Input disparity values array", "",
                                 gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(timestamp_policy_, "timestamp_policy",
                                 "Input channel to get timestamp 0(left)/1(right)", "", 0);

  return gxf::ToResultCode(result);
}

gxf_result_t Bi3D::start() {

  // Allocate cuda stream using stream pool if necessary
  if (stream_pool_.try_get()) {
    auto stream = stream_pool_.try_get().value()->allocateStream();
    if (!stream) {
      GXF_LOG_ERROR("allocating stream failed.");
      return GXF_FAILURE;
    }
    cuda_stream_ = std::move(stream.value());
    if (!cuda_stream_->stream()) {
      GXF_LOG_ERROR("allocated stream is not initialized.");
      return GXF_FAILURE;
    }
  }

  // Setting image pre-processing params for Bi3D
  const auto& pixel_mean_vec         = pixel_mean_.get();
  const auto& normalization_vec      = normalization_.get();
  const auto& standard_deviation_vec = standard_deviation_.get();
  if (pixel_mean_vec.size() != 3 || normalization_vec.size() != 3 || standard_deviation_vec.size() != 3) {
    GXF_LOG_ERROR("Invalid preprocessing params.");
    return GXF_FAILURE;
  }
  std::copy(pixel_mean_vec.begin(), pixel_mean_vec.end(), preProcessorParams.pixelMean);
  std::copy(normalization_vec.begin(), normalization_vec.end(), preProcessorParams.normalization);
  std::copy(standard_deviation_vec.begin(), standard_deviation_vec.end(), preProcessorParams.stdDev);

  // Setting model input params for Bi3D
  modelInputParams.maxBatchSize     = max_batch_size_.get();
  modelInputParams.inputLayerWidth  = input_layer_width_.get();
  modelInputParams.inputLayerHeight = input_layer_height_.get();

  // Setting inference params for Bi3D
  auto toComputeEngine = [](std::string type) -> gxf::Expected<int> {
    if (type == "GPU") {
      return -1;
    } else if (type == "DLA_CORE_0") {
      return 0;
    } else if (type == "DLA_CORE_1") {
      return 1;
    } else {
      GXF_LOG_ERROR("Invalid compute engine type.");
      return gxf::Unexpected{GXF_FAILURE};
    }
  };
  auto engineType = toComputeEngine(engine_type_.get());
  if (!engineType) {
    return engineType.error();
  }
  featureInferenceParams = ::cvcore::inferencer::TensorRTInferenceParams{
                              ::cvcore::inferencer::TRTInferenceType::TRT_ENGINE,
                              nullptr,
                              featnet_engine_file_path_.get(),
                              1,
                              featnet_input_layers_name_.get(),
                              featnet_output_layers_name_.get(),
                              engineType.value()};
  featureHRInferenceParams = ::cvcore::inferencer::TensorRTInferenceParams{
                                ::cvcore::inferencer::TRTInferenceType::TRT_ENGINE,
                                nullptr,
                                featnet_hr_engine_file_path_.try_get().value_or(""),
                                1,
                                featnet_hr_input_layers_name_.try_get().value_or(std::vector<std::string>{""}),
                                featnet_hr_output_layers_name_.try_get().value_or(std::vector<std::string>{""}),
                                engineType.value()};
  refinementInferenceParams = ::cvcore::inferencer::TensorRTInferenceParams{
                                ::cvcore::inferencer::TRTInferenceType::TRT_ENGINE,
                                nullptr,
                                refinenet_engine_file_path_.try_get().value_or(""),
                                1,
                                refinenet_input_layers_name_.try_get().value_or(std::vector<std::string>{""}),
                                refinenet_output_layers_name_.try_get().value_or(std::vector<std::string>{""}),
                                engineType.value()};
  segmentationInferenceParams = ::cvcore::inferencer::TensorRTInferenceParams{
                                  ::cvcore::inferencer::TRTInferenceType::TRT_ENGINE,
                                  nullptr,
                                  segnet_engine_file_path_.get(),
                                  1,
                                  segnet_input_layers_name_.get(),
                                  segnet_output_layers_name_.get(),
                                  engineType.value()};

  // Setting extra params for Bi3D
  auto toProcessingControl = [](bool flag) {
    return flag ? ::cvcore::bi3d::ProcessingControl::ENABLE : ::cvcore::bi3d::ProcessingControl::DISABLE;
  };
  extraParams = {max_disparity_levels_.get(),
                 toProcessingControl(apply_edge_refinement_.get()),
                 toProcessingControl(apply_sigmoid_.get()), 
                 toProcessingControl(apply_thresholding_.get()),
                 threshold_value_low_.get(),
                 threshold_value_high_.get(),
                 threshold_.get()};

  // Setting Bi3D object with the provided params
  objBi3D.reset(new ::cvcore::bi3d::Bi3D(preProcessorParams, modelInputParams, featureInferenceParams,
                                         featureHRInferenceParams, refinementInferenceParams,
                                         segmentationInferenceParams, extraParams));

  const auto& disparity_values_vec = disparity_values_.try_get();
  if (disparity_values_vec) {
    if (disparity_values_vec.value().empty() || disparity_values_vec.value().size() > max_disparity_levels_.get()) {
      GXF_LOG_ERROR("Invalid disparity values.");
      return GXF_FAILURE;
    }
    disparityValues = ::cvcore::Array<int>(disparity_values_vec.value().size(), true);
    disparityValues.setSize(disparity_values_vec.value().size());

    for (size_t i = 0; i < disparity_values_vec.value().size(); i++) {
      disparityValues[i] = disparity_values_vec.value()[i];
    }
  }
  return GXF_SUCCESS;
}

gxf_result_t Bi3D::tick() {

  // Get a CUDA stream for execution
  cudaStream_t cuda_stream = 0;
  if(cuda_stream_.try_get())
  {
    cuda_stream = cuda_stream_->stream().value();
  }

  // Receiving the data synchronously across streams
  auto inputLeftMessage = left_image_receiver_->receive();
  if (!inputLeftMessage) {
    return inputLeftMessage.error();
  } else if (cuda_stream != 0) {
    detail::BindCudaEvent(inputLeftMessage.value(), cuda_stream_);
    auto inputLeft_stream_id = inputLeftMessage.value().get<gxf::CudaStreamId>("stream");
    if(inputLeft_stream_id) {
      auto inputLeft_stream = gxf::Handle<gxf::CudaStream>::Create(inputLeft_stream_id.value().context(),
                                                                   inputLeft_stream_id.value()->stream_cid);
      // NOTE: This is an expensive call. It will halt the current CPU thread until all events
      //   previously associated with the stream are cleared
      inputLeft_stream.value()->syncStream();
    }
  }
  auto inputRightMessage = right_image_receiver_->receive();
  if (!inputRightMessage) {
    return inputRightMessage.error();
  } else if (cuda_stream != 0) {
    detail::BindCudaEvent(inputRightMessage.value(), cuda_stream_);
    auto inputRight_stream_id = inputRightMessage.value().get<gxf::CudaStreamId>("stream");
    if(inputRight_stream_id) {
      auto inputRight_stream = gxf::Handle<gxf::CudaStream>::Create(inputRight_stream_id.value().context(),
                                                                    inputRight_stream_id.value()->stream_cid);
      // NOTE: This is an expensive call. It will halt the current CPU thread until all events
      //   previously associated with the stream are cleared
      inputRight_stream.value()->syncStream();
    }
  }

  auto maybeLeftName = left_image_name_.try_get();
  auto inputLeftBuffer = inputLeftMessage.value().get<gxf::VideoBuffer>(
    maybeLeftName ? maybeLeftName.value().c_str() : nullptr);
  if (!inputLeftBuffer) {
    return inputLeftBuffer.error();
  }
  auto maybeRightName = right_image_name_.try_get();
  auto inputRightBuffer = inputRightMessage.value().get<gxf::VideoBuffer>(
    maybeRightName ? maybeRightName.value().c_str() : nullptr);
  if (!inputRightBuffer) {
    return inputRightBuffer.error();
  }
  if (inputLeftBuffer.value()->storage_type() != gxf::MemoryStorageType::kDevice ||
      inputRightBuffer.value()->storage_type() != gxf::MemoryStorageType::kDevice) {
    GXF_LOG_ERROR("input images must be on GPU.");
    return GXF_FAILURE;
  }
  const auto & left_info = inputLeftBuffer.value()->video_frame_info();
  const auto & right_info = inputRightBuffer.value()->video_frame_info();
  if (left_info.color_format != right_info.color_format ||
      left_info.height != right_info.height || left_info.width != right_info.width ||
      left_info.surface_layout != left_info.surface_layout ||
      (left_info.color_format != gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB &&
       left_info.color_format != gxf::VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32 &&
       left_info.color_format != gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR &&
       left_info.color_format != gxf::VideoFormat::GXF_VIDEO_FORMAT_B32_G32_R32)) {
    return GXF_INVALID_DATA_FORMAT;
  }

  // Create output message
  gxf::Expected<gxf::Entity> outputMessage = gxf::Entity::New(context());
  if (!outputMessage) {
    GXF_LOG_ERROR("Bi3D::tick ==> Failed to create output message.");
    return outputMessage.error();
  }

  // Receive the disparity array if necessary
  const auto& maybeDisparityReceiver = disparity_receiver_.try_get();
  if (maybeDisparityReceiver) {
    auto disparityMessage = maybeDisparityReceiver.value()->receive();
    if (!disparityMessage) {
      return disparityMessage.error();
    }
    auto disparityTensor = disparityMessage.value().get<gxf::Tensor>();
    if (!disparityTensor) {
      return disparityTensor.error();
    }
    if (disparityTensor.value()->element_count() > max_disparity_levels_.get() ||
        disparityTensor.value()->element_type() != gxf::PrimitiveType::kInt32 ||
        disparityTensor.value()->storage_type() == gxf::MemoryStorageType::kDevice) {
      GXF_LOG_ERROR("invalid input disparity values.");
      return GXF_FAILURE;
    }
    const auto disparityCount = disparityTensor.value()->element_count();
    disparityValues           = ::cvcore::Array<int>(disparityCount, true);
    disparityValues.setSize(disparityCount);

    for (size_t i = 0; i < disparityCount; i++) {
      disparityValues[i] = disparityTensor.value()->data<int32_t>().value()[i];
    }

    auto forwardedDisparity = outputMessage.value().add<gxf::Tensor>(disparityTensor->name());
    if (strcmp(disparityTensor->name(), output_name_.get().c_str()) == 0) {
      GXF_LOG_ERROR("Bi3D::tick ==> Forwarded disparity array name and network output name must differ.");
      return GXF_FAILURE;
    }
    if (!forwardedDisparity) {
      GXF_LOG_ERROR("Bi3D::tick ==> Failed to forward input disparity tensor in output message.");
      return forwardedDisparity.error();
    }
    *forwardedDisparity.value() = std::move(*disparityTensor.value());
  }

  // Creating GXF tensor to hold the data to be transmitted
  auto outputImage = outputMessage.value().add<gxf::Tensor>(output_name_.get().c_str());
  if (!outputImage) {
    GXF_LOG_ERROR("Bi3D::tick ==> Failed to create tensor in output message.");
    return outputImage.error();
  }
  auto result = outputImage.value()->reshape<float>({static_cast<int>(disparityValues.getSize()),
                                                     static_cast<int>(left_info.height),
                                                     static_cast<int>(left_info.width)},
                                                    gxf::MemoryStorageType::kDevice, pool_);
  if (!result) {
    GXF_LOG_ERROR("Bi3D::tick ==> Failed to allocate tensor in output message.");
    return result.error();
  }

  // Creating CVCore Tensors to hold the input and output data
  ::cvcore::Tensor<::cvcore::CHW, ::cvcore::CX, ::cvcore::F32> outputImageDevice(
    left_info.width, left_info.height, disparityValues.getSize(),
    const_cast<float*>(outputImage.value()->data<float>().value()), false);

  // Running the inference
  if (left_info.color_format == gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB ||
      left_info.color_format == gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR) {
    ::cvcore::Tensor<::cvcore::HWC, ::cvcore::C3, ::cvcore::U8> leftImageDevice(
      left_info.width, left_info.height,
      reinterpret_cast<uint8_t*>(inputLeftBuffer.value()->pointer()), false);
    ::cvcore::Tensor<::cvcore::HWC, ::cvcore::C3, ::cvcore::U8> rightImageDevice(
      right_info.width, right_info.height,
      reinterpret_cast<uint8_t*>(inputRightBuffer.value()->pointer()), false);
    objBi3D->execute(outputImageDevice,
                     leftImageDevice, rightImageDevice,
                     disparityValues, cuda_stream);
  } else if (left_info.color_format == gxf::VideoFormat::GXF_VIDEO_FORMAT_R32_G32_B32 ||
             left_info.color_format == gxf::VideoFormat::GXF_VIDEO_FORMAT_B32_G32_R32) {
    ::cvcore::Tensor<::cvcore::CHW, ::cvcore::C3, ::cvcore::F32> leftImageDevice(
      left_info.width, left_info.height,
      reinterpret_cast<float*>(inputLeftBuffer.value()->pointer()), false);
    ::cvcore::Tensor<::cvcore::CHW, ::cvcore::C3, ::cvcore::F32> rightImageDevice(
      right_info.width, right_info.height,
      reinterpret_cast<float*>(inputRightBuffer.value()->pointer()), false);
    objBi3D->execute(outputImageDevice,
                     leftImageDevice, rightImageDevice,
                     disparityValues, cuda_stream);
  } else {
    return GXF_FAILURE;
  }

  // Allocate a cuda event that can be used to record on each tick
  if(cuda_stream_.try_get())
  {
    detail::BindCudaStream(outputMessage.value(), cuda_stream_.cid());
    detail::BindCudaEvent(outputMessage.value(), cuda_stream_);
  }

  // Pass down timestamp if necessary
  auto maybeDaqTimestamp = timestamp_policy_.get() == 0 ? inputLeftMessage.value().get<gxf::Timestamp>()
                                                        : inputRightMessage.value().get<gxf::Timestamp>();
  if (maybeDaqTimestamp) {
    auto outputTimestamp = outputMessage.value().add<gxf::Timestamp>(maybeDaqTimestamp.value().name());
    if (!outputTimestamp) {
      return outputTimestamp.error();
    }
    *outputTimestamp.value() = *maybeDaqTimestamp.value();
  }

  // Send the data
  output_transmitter_->publish(outputMessage.value());
  return GXF_SUCCESS;
}

gxf_result_t Bi3D::stop() {
  objBi3D.reset(nullptr);
  return GXF_SUCCESS;
}

} // namespace cvcore
} // namespace nvidia
