#include "simple_rknn.h"

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    std::stringstream ss;
    for (size_t i = 0; i < attr->n_dims; ++i) {
        if (i != 0)
            ss << ", "; // Separate elements by a comma
        ss << attr->dims[i];
    }
    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, ss.str().c_str(),
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

SimpleRKNN::SimpleRKNN()
{
}

SimpleRKNN::~SimpleRKNN()
{
    destroy();
}

int SimpleRKNN::init(const std::string &model_path)
{

    int ret;
    int model_len = 0;
    char *model;

    this->model_path = model_path;

    printf("init_yolov5_model from %s\n", model_path.c_str());

    ret = rknn_init(&ctx, (char *)this->model_path.c_str(), 0, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    ret = rknn_query(
        ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf(
        "model input num: %d, output num: %d\n",
        io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    input_attrs.clear();
    for (int i = 0; i < io_num.n_input; i++)
    {
        rknn_tensor_attr attr;
        attr.index = i;
        ret = rknn_query(
            ctx,
            RKNN_QUERY_NATIVE_INPUT_ATTR,
            &(attr),
            sizeof(rknn_tensor_attr));

        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }

        input_attrs.push_back(attr);

        dump_tensor_attr(&attr);
    }

    // Get Model Output Info
    printf("output tensors:\n");
    output_attrs.clear();
    for (int i = 0; i < io_num.n_output; i++)
    {
        rknn_tensor_attr attr;
        attr.index = i;
        // When using the zero-copy API interface,
        // query the native output tensor attribute
        ret = rknn_query(
            ctx,
            RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR,
            &(attr),
            sizeof(rknn_tensor_attr));

        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }

        output_attrs.push_back(attr);

        dump_tensor_attr(&(attr));
    }

    // default input type is int8 (normalize and quantize need compute in outside)
    // if set uint8, will fuse normalize and quantize to npu
    input_attrs[0].type = RKNN_TENSOR_UINT8;
    // default fmt is NHWC,1106 npu only support NHWC in zero copy mode
    input_attrs[0].fmt = RKNN_TENSOR_NHWC;
    printf("input_attrs[0].size_with_stride=%d\n", input_attrs[0].size_with_stride);

    input_mems.clear();
    for (uint32_t i = 0; i < io_num.n_input; ++i)
    {
        printf("create input mem with size=%u\n", input_attrs[i].size_with_stride);
        auto *mem_ptr = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
        // Set input tensor memory
        ret = rknn_set_io_mem(ctx, mem_ptr, &input_attrs[0]);
        if (ret < 0)
        {
            printf("input_mems rknn_set_io_mem fail! ret=%d\n", ret);
            return -1;
        }
        input_mems.push_back(mem_ptr);
    }

    // Set output tensor memory
    output_mems.clear();
    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        printf("create mem with size=%u\n", output_attrs[i].size_with_stride);
        auto *mem_ptr = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        ret = rknn_set_io_mem(ctx, mem_ptr, &output_attrs[i]);
        if (ret < 0)
        {
            printf("output_mems rknn_set_io_mem fail! ret=%d\n", ret);
            return -1;
        }
        output_mems.push_back(mem_ptr);
    }

    return 0;

    // TODO:
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
    {
        is_quant = true;
    }
    else
    {
        is_quant = false;
    }

    return 0;
}

int SimpleRKNN::run(const cv::Mat &input)
{

    // Run
    printf("rknn_run\n");
    int ret = rknn_run(ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    return ret;
}

int SimpleRKNN::destroy()
{

    if (ctx != 0)
    {
        printf("rknn_destroy %llu\n", ctx);
        rknn_destroy(ctx);
    }

    ctx = 0;
    return 0;
}
