#include "config_loader.h"
#include "image_io.h"
#include "trt_helper.h"

#include <vector>

std::vector<Caption> processCaption(const Config &config,
                                    const std::vector<Image> &inputs,
                                    nvinfer1::IExecutionContext *&context);
