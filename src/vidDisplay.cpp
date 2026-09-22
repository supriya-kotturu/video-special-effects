/*
  Author:  Sai Supriya Kotturu
  Date:    2026-09-21
  Purpose: Tasks 2+ - capture live video, display it, and apply the effects from
          filters.h according to the key the user last pressed.
*/

#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "filters.h"

enum class Mode { RGB, GREY };

int saveImage(cv::Mat frame, int currFrameId) {
  const std::string outDir = "data/frames";
  std::filesystem::create_directories(outDir);  // imwrite won't do this itself

  std::string fileName =
      outDir + "/captured_frame_" + std::to_string(currFrameId) + ".jpg";

  if (cv::imwrite(fileName, frame)) {
    std::cout << "Saved frame to " << fileName << std::endl;
    return 1;
  } else {
    std::cerr << "Failed to save frame." << std::endl;
  }

  return (0);
}

int applyGreyFilter(cv::Mat& src, cv::Mat& dest) {
  cv::cvtColor(src, dest, cv::COLOR_BGR2GRAY);
  return (0);
}

int main(int argc, char* argv[]) {
  cv::VideoCapture* capdev;

  // open the video device
  capdev = new cv::VideoCapture(0);
  if (!capdev->isOpened()) {
    printf("Unable to open video device\n");
    return (-1);
  }

  // force a lower resolution
  capdev->set(cv::CAP_PROP_FRAME_WIDTH, 640);
  capdev->set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  // get some properties of the image
  cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
  printf("Expected size: %d %d\n", refS.width, refS.height);

  cv::namedWindow("Video", 1);  // identifies a window
  cv::Mat frame;
  cv::Mat displayFrame;
  int frameCounter = 1;
  Mode mode = Mode::RGB;

  while (true) {
    *capdev >> frame;  // get a new frame from the camera, treat as a stream

    if (frame.empty()) {
      printf("frame is empty\n");
      break;
    }

    // see if there is a waiting keystroke
    char key = cv::waitKey(10);

    switch (key) {
      case 'g':
        mode = Mode::GREY;
      case 'c':
        mode = Mode::RGB;
      case 's':
        saveImage(displayFrame, frameCounter);
        frameCounter++;
        break;
      case 'q':
        return (0);
    }

    if (mode == Mode::GREY) {
      applyGreyFilter(frame, displayFrame);
    } else {
      displayFrame = frame;
    }

    cv::imshow("Live!", displayFrame);
  }

  cv::destroyAllWindows();
  delete capdev;
  return (0);
}
