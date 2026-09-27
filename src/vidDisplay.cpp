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

enum class Mode { RGB, GRAY, CUSTOM_GRAY, SEPIA, X_RAY, BLUR };

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

  std::string windowName = "Live!!";

  cv::namedWindow(windowName, 1);  // identifies a window
  cv::Mat frame;
  cv::Mat displayFrame;
  int frameCounter = 1;
  Mode mode = Mode::RGB;
  bool vignette = false;

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
        std::cout << "Changing mode to GRAYSCALE" << std::endl;
        mode = Mode::GRAY;
        break;
      case 'h':
        std::cout << "Changing mode to CUSTOM_GRAYSCALE" << std::endl;
        mode = Mode::CUSTOM_GRAY;
        break;
      case 'x':
        std::cout << "Changing mode to X_RAY" << std::endl;
        mode = Mode::X_RAY;
        break;
      case 'c':
        std::cout << "Changing mode to RGB" << std::endl;
        mode = Mode::RGB;
        break;
      case 'p':
        std::cout << "Changing mode to Sepia" << std::endl;
        mode = Mode::SEPIA;
        break;
      case 'b':
        std::cout << "Changing mode to Blur" << std::endl;
        mode = Mode::BLUR;
        break;
      case 'v':
        std::cout << "Swithing vignette" << std::endl;
        vignette = !vignette;
        break;
      case 'q':
        return (0);
    }

    switch (mode) {
      case Mode::GRAY:
        applyGrayscaleFilter(frame, displayFrame);
        break;
      case Mode::CUSTOM_GRAY:
        applyCustomGrayscaleFilter(frame, displayFrame);
        break;
      case Mode::X_RAY:
        applyXRayFilter(frame, displayFrame);
        break;
      case Mode::SEPIA:
        applySepiaFilter(frame, displayFrame);
        break;
      case Mode::BLUR:
        applyBlur(frame, displayFrame);
        break;
      default:
        displayFrame = frame;
        break;
    }

    if (vignette) {
      applyVignette(displayFrame, displayFrame);
    }

    if (key == 's') {
      saveImage(displayFrame, frameCounter);
      frameCounter++;
    }

    cv::imshow(windowName, displayFrame);
  }

  cv::destroyAllWindows();
  delete capdev;
  return (0);
}
