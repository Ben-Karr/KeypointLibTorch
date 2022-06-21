//Help: https://github.com/pytorch/vision/tree/main/test/tracing/frcnn
//      https://github.com/pytorch/vision/tree/main/examples/cpp
#include <torch/script.h>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <torchvision/ops/nms.h>

#include <iostream>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

cv::Mat draw_keypoints(cv::Mat img, torch::Tensor keypoints, torch::Tensor scores, bool with_limbs);
cv::Mat draw_limbs(cv::Mat img, std::vector<cv::Point2f> points);
torch::Tensor mat_to_tensor(cv::Mat img, const int img_size);

int main(){
    const int img_size = 640;
    cv::VideoCapture cap(0);
    cv::Mat img;
    double frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double ratio = frame_width / frame_height;

    torch::jit::script::Module model;
    try {
        model = torch::jit::load("../../models/traced_keypoint_model.pt"); // The model includes normalization
        model.to(torch::kCUDA);
        model.eval();
        std::cout << "Successfully loaded model" <<std::endl;
    }
    catch (const c10::Error& e){
        std::cerr << "Error loading the model\n";
        return -1;
    }
    while(true){
        cap.read(img);

        if(img.empty()){
            std::cerr << "Frame is empty" << '\n';
            break;
        }
        cv::resize(img, img, cv::Size(img_size, img_size), cv::INTER_LINEAR);
        img.convertTo(img, CV_32FC3, 1.0f / 255.0f);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        torch::Tensor img_tensor = mat_to_tensor(img, img_size);

        std::vector<torch::jit::IValue> inputs;
        std::vector<torch::Tensor> images;

        images.push_back(img_tensor);
        inputs.push_back(images);

        auto output = model.forward(inputs);
        auto detections = output.toTuple()->elements().at(1).toList().get(0).toGenericDict();
        torch::Tensor keypoints = detections.at("keypoints").toTensor().to(torch::kCPU);
        torch::Tensor scores = detections.at("scores").toTensor().to(torch::kCPU);

        img = draw_keypoints(img, keypoints, scores, true);
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

        cv::resize(img, img, cv::Size(int(img_size * ratio), img_size), cv::INTER_LINEAR);
        cv::imshow("Keypoints on input", img);

        char c = cv::waitKey(1);
        if (c == 27){
            break;
        }
    }
    
    return 0;
}

cv::Mat draw_keypoints(cv::Mat img, torch::Tensor keypoints, torch::Tensor scores, bool with_limbs){
    for (int i = 0; i < keypoints.sizes()[0]; ++i){
        // Skip instance if model is 'unshure' that it found a person
        if (scores[i].item().toFloat() < 0.7) {
            continue;
        }
        std::vector<cv::Point2f> points;
        for (int j = 0; j < keypoints.sizes()[1]; ++j){
            float x_value, y_value;
            x_value = keypoints[i][j][0].item().toFloat();
            y_value = keypoints[i][j][1].item().toFloat();
            cv::Point2f key_point(x_value, y_value);
            points.push_back(key_point);
            cv::circle(img, key_point, 3, cv::Scalar(0, 0, 255));
        }
        if (with_limbs) {
            img = draw_limbs(img, points);
        }
    }
    return img;
}

cv::Mat draw_limbs(cv::Mat img, std::vector<cv::Point2f> points){
    enum {nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle};
    // froms[i] forms a limb with tos[i]
    std::vector<int> froms{right_eye, right_eye, left_eye, left_eye, right_shoulder, right_elbow, left_shoulder, left_elbow, right_hip, right_knee, left_hip, left_knee, right_shoulder, right_hip, right_shoulder, left_shoulder};
    std::vector<int>   tos{nose, right_ear, nose, left_ear, right_elbow, right_wrist, left_elbow, left_wrist, right_knee, right_ankle, left_knee, left_ankle, left_shoulder, left_hip, right_hip, left_hip};
    cv::Point2f from;
    cv::Point2f to;

    for (int i = 0; i <  16; i++){
        from = points[froms[i]];
        to = points[tos[i]];
        cv::line(img, from, to, cv::Scalar(0, 0, 0), 2); // to see line on light backgrounds
        cv::line(img, from, to, cv::Scalar(10, 255, 10), 1);
    }
    return img;
}

torch::Tensor mat_to_tensor(cv::Mat img, const int img_size){
    torch::Tensor img_tensor = torch::from_blob(img.data, {img_size, img_size, 3});
    img_tensor = img_tensor.permute({2,0,1});
    img_tensor = img_tensor.to(torch::kCUDA);
    img_tensor = img_tensor.contiguous();
    return img_tensor;
}
