#include <iostream>
#include <string>
#include <algorithm>
#include <cstring>

#include </usr/local/include/opencv4/opencv2/opencv.hpp>
#include </usr/local/include/opencv4/opencv2/dnn.hpp>
#include </usr/local/include/opencv4/opencv2/highgui.hpp>
#include </usr/local/include/opencv4/opencv2/imgproc.hpp>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include "tinyexpr.h"

using namespace std;
using namespace cv;
using namespace tesseract;
using namespace cv::dnn;

Mat frame, frameCopy, frameGrayScale, thresholdFrame, blob, matrix, rotated;
vector<Mat> cropped;

void decode(const Mat& scores, const Mat& geometry, float scoreThresh, 
            vector<RotatedRect>& boxes, vector<float>& confidences) {

    const int height = scores.size[2];
    const int width = scores.size[3];

    for (int y = 0; y < height; ++y) {
        const float* scoresData = scores.ptr<float>(0, 0, y);
        const float* x0_data = geometry.ptr<float>(0, 0, y);
        const float* x1_data = geometry.ptr<float>(0, 1, y);
        const float* x2_data = geometry.ptr<float>(0, 2, y);
        const float* x3_data = geometry.ptr<float>(0, 3, y);
        const float* anglesData = geometry.ptr<float>(0, 4, y);
        
        for (int x = 0; x < width; ++x) {
            float score = scoresData[x];
            if (score < scoreThresh) continue;

            // decode a prediction.
            // Multiple by 4 because feature maps are 4 time less than input image.
            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = cos(angle);
            float sinA = sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];

            Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                           offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
            Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
            RotatedRect box(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            boxes.push_back(box);
            confidences.push_back(score);
        }
    }
}

// draw box around text
void drawBoxesAroundText(const RotatedRect& box, const Point2f ratio) {
    // take each vertice of the text box and draw rectangle
    Point2f vertices[4];
    box.points(vertices);
    for (int j = 0; j < 4; ++j) {
        vertices[j].x *= ratio.x;
        vertices[j].y *= ratio.y;
    }
    for (int j = 0; j < 4; ++j) {
        line(frame, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 2);
    }
}

// crop the text box from
void cropText(RotatedRect& box, const Point2f ratio, int index, Mat& crop, vector<Mat>& cropped) {
    float boxAngle = box.angle;
    Size boxSize = box.size;

    boxSize.width *= ratio.x;
    boxSize.height *= ratio.y;
    box.center.x *= ratio.x;
    box.center.y *= ratio.y;

    if (box.angle < -45) {
        boxAngle += 90.0;
        swap(boxSize.width, boxSize.height);
    }

    matrix = getRotationMatrix2D(box.center, boxAngle, 1.0);
    warpAffine(frameCopy, rotated, matrix, frame.size(), INTER_CUBIC);
    getRectSubPix(rotated, boxSize, box.center, crop);

    // imshow("Rotated", rotated);
    // imshow("Cropped", crop);
    
    cropped[index] = crop;
}

int main() {
    // initialize Tesseract API for text recognition
    TessBaseAPI *tessAPI = new TessBaseAPI();
    if (tessAPI->Init(NULL, "eng", OEM_LSTM_ONLY)) {
        cout << "tessAPI initialize error" << endl;
        return -1;
    }
    tessAPI->SetVariable("tessedit_char_whitelist", "0123456789+/*=.");

    // load the network into memory for text detection
    Net net = readNet("frozen_east_text_detection.pb");

    // vector of Mat for network outputs
    vector<Mat> output;
    vector<String> outputLayers(2);
    // two output of the network 
    // 1st is the confidence score of the detected box 
    // 2nd is the geometry of the Text-box
    outputLayers[0] = "feature_fusion/Conv_7/Sigmoid";
    outputLayers[1] = "feature_fusion/concat_3";

    // the id of video device. 0 is webcam
    VideoCapture vc(0);

    // check if video device has been initialised
    if (!vc.isOpened()) {
        cout << "Cannot open camera.";
        return -1;
    } else {
        cout << "Manage to open camera.\n";
    }

    // unconditional loop
    while (true) {
        
        // get frame from camera
        vc >> frame;

        // create 4-dimensional blob from image, input image for network
        blobFromImage(frame, blob, 1.0, Size(128, 128), Scalar(123.68, 116.78, 103.94), true, false);

        // pass the input image through the network
        net.setInput(blob);
        net.forward(output, outputLayers);

        // output of the network
        Mat scores = output[0];
        Mat geometry = output[1];

        // decode the positions and orientations of the text boxes and filter out the most probable text box
        vector<RotatedRect> boxes;
        vector<float> confidences;
        decode(scores, geometry, 0.5, boxes, confidences);

        // filter out the false positives and get the final predictions using Non-Maximum Suppression
        vector<int> indices;
        NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

        // Mat crop for cropping the text boxes and vector<Mat> for storing them
        Mat crop;
        cropped.resize(indices.size());
        
        // create copy of the original frame. The cropping process will be perform on copied frame
        frame.copyTo(frameCopy);

        // ratio of detected text box and frame 
        Point2f ratio((float)frame.cols / 128, (float)frame.rows / 128);
        // draw rectangle for each text box and crop the text area
        for (size_t i = 0; i < indices.size(); ++i) {
            RotatedRect& box = boxes[indices[i]];
            drawBoxesAroundText(box, ratio);
            cropText(box, ratio, i, crop, cropped);
        }

        // work with each cropped images with text
        for (int i = 0; i < cropped.size(); ++i) {

            // recognized expression, position of unnecessary chars
            string recognizedExpression;
            // string prevExpr;
            size_t foundCharPosition;

            // image preprocessing: convert to grayscale -> apply blur -> apply threshold
            cvtColor(cropped[i], frameGrayScale, COLOR_RGB2GRAY);
            // imshow("Gray", frameGrayScale);
            GaussianBlur(frameGrayScale, frameGrayScale, Size(5, 5), 0, 0);
            // imshow("Blur", frameGrayScale);
            adaptiveThreshold(frameGrayScale, thresholdFrame, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 75, 10);
            // imshow("Threshold", thresholdFrame);

            // send preprocessed cropped image with text into Tesseract API for recognizing the text and getting output
            tessAPI->SetImage(thresholdFrame.data, thresholdFrame.size().width, thresholdFrame.size().height, thresholdFrame.channels(), thresholdFrame.step1());
            tessAPI->Recognize(0);
            recognizedExpression = tessAPI->GetUTF8Text();
            cout << "Recognized text: " << recognizedExpression << endl;

            // solve the math expression if the output of the tessAPI is not empty
            if (recognizedExpression != "") {
            
                // find and delet unnecessary chars from the recognized expression: ' ', '='
                foundCharPosition = recognizedExpression.find(' ');
                if (foundCharPosition!=string::npos)  recognizedExpression.erase(recognizedExpression.begin()+foundCharPosition);
                foundCharPosition = recognizedExpression.find('=');
                if (foundCharPosition!=string::npos)  recognizedExpression.erase(recognizedExpression.begin()+foundCharPosition);
                
                // find and change some chars ('x', ':') from the recognized expression to correct ones('*', '/') for solving
                foundCharPosition = recognizedExpression.find('x');
                if (foundCharPosition!=string::npos)  recognizedExpression.at(foundCharPosition) = '*';
                foundCharPosition = recognizedExpression.find('X');
                if (foundCharPosition!=string::npos)  recognizedExpression.at(foundCharPosition) = '*';
                foundCharPosition = recognizedExpression.find(':');
                if (foundCharPosition!=string::npos)  recognizedExpression.substr(foundCharPosition, 1) = '/';

                // convert string to char* for solving using tinyexpr methods                
                char* expression;
                strcpy(expression, recognizedExpression.c_str());
                double finalResult = te_interp(expression, 0);

                // put result of solving under the box of the text in the video
                putText(frame, to_string(finalResult), Point2f(boxes[indices[i]].center.x, 
                        boxes[indices[i]].center.y+boxes[indices[i]].size.height*3), FONT_HERSHEY_SIMPLEX, 
                        int(boxes[indices[i]].size.height/10), Scalar(0, 0, 255), 3);
            }

            // empty recognizedExpression
            recognizedExpression = "";  
        }

        // empty the vector of cropped images after 
        cropped.clear();

        // Display the resulting frame
        imshow("Frame", frame);

        // Press  ESC on keyboard to exit
        if (waitKey(30) == 27) break;
    }

    // release the video capture object
    vc.release();

    // Close all the frames
    destroyAllWindows();

    // empty tessAPI and delete pointer of tessAPI
    tessAPI->Clear();
    tessAPI->End();

    return 0;
}
