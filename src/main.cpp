#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <unistd.h>   // sleep()

#include "timer.h"
#include "simple_rknn.h"

int main()
{
    printf("main: running SimpleRKNN \n");
    SimpleRKNN model;
    model.init("yolov5.rknn");

    cv::Mat img = cv::imread("model/bus.jpg");
    printf("img read: %d %d\n", img.rows, img.cols);
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::resize(img, img, cv::Size(640, 640));
    // printf("img resize: %d %d\n", img.rows, img.cols);

    // printf("input sample");
    // for (int i = 0; i < 100; ++i)
    // {
    //     printf("%d ", img.data[i * 10]);
    // }
    // printf("\n");

    // model.run(cv::Mat());

    // return 0;

    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
    cap.open(0);

    const int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    fprintf(stderr, "%d x %d\n", w, h);

    cv::Mat bgr[9];
    cv::Mat input;
    for (int i = 0; i < 9; i++)
    {
        Timer timer;
        timer.start();
        cap >> bgr[i];
        // cv::cvtColor(bgr[i], input, cv::COLOR_BGR2RGB);
        // cv::resize(input, input, cv::Size(640, 640));
        memcpy(model.input_mems[0]->virt_addr, img.data, 640 * 640 * 3);
        model.run(cv::Mat());
        printf("grab: %lf s\n", timer.elapsedSeconds());

        // sleep(1);
    }


    cap.release();

    // combine into big image
    {
        cv::Mat out(h * 3, w * 3, CV_8UC3);
        bgr[0].copyTo(out(cv::Rect(0, 0, w, h)));
        bgr[1].copyTo(out(cv::Rect(w, 0, w, h)));
        bgr[2].copyTo(out(cv::Rect(w * 2, 0, w, h)));
        bgr[3].copyTo(out(cv::Rect(0, h, w, h)));
        bgr[4].copyTo(out(cv::Rect(w, h, w, h)));
        bgr[5].copyTo(out(cv::Rect(w * 2, h, w, h)));
        bgr[6].copyTo(out(cv::Rect(0, h * 2, w, h)));
        bgr[7].copyTo(out(cv::Rect(w, h * 2, w, h)));
        bgr[8].copyTo(out(cv::Rect(w * 2, h * 2, w, h)));

        cv::imwrite("out.jpg", out);
    }

    return 0;
}
