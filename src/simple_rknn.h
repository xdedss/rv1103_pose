#ifndef SIMPLE_RKNN_H
#define SIMPLE_RKNN_H

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

#include "rknn_api.h"

typedef struct
{
    char *dma_buf_virt_addr;
    int dma_buf_fd;
    int size;
} rknn_dma_buf;

class SimpleRKNN {
private:
    rknn_context ctx = 0;
    std::string model_path;
    rknn_dma_buf img_dma_buf; // currently unused, required by rga, see from_zoo/main.cc
    bool is_quant;

public:
    rknn_input_output_num io_num;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    std::vector<rknn_tensor_mem*> input_mems;
    std::vector<rknn_tensor_mem*> output_mems;

    SimpleRKNN();
    ~SimpleRKNN();
    int init(const std::string &model_path);
    int run(const cv::Mat &input);
    int destroy();
};

#endif /* SIMPLE_RKNN_H */
