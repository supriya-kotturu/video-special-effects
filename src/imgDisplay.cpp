/*
  Author:  Sai Supriya Kotturu
  Date:    2026-09-21
  Purpose: Task 1 - read an image from a file, display it in a window, and loop
          until the user presses 'q'.
*/

#include <filters.h>

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

/*
  Loads the image named on the command line and shows its Sobel X and Sobel Y
  outputs (absolute value) in two windows; used to check the task 7 filters on
  a still image. Blocks until the user presses 'q'.

  argv[1]: path to the image file
  returns: 0 on quit, -1 on missing argument, 1 if the image can't be read
*/
int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_image>" << std::endl;
    return -1;
  }

  std::string filePath = argv[1];
  cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);

  if (image.empty()) {
    std::cerr << "Error: Could not open or find the image." << std::endl;
    return 1;
  }

  std::string windowName = "YAY...";

  cv::namedWindow(windowName, cv::WINDOW_NORMAL);
  cv::resizeWindow(windowName, 600, 800);
  cv::imshow(windowName, image);

  // cv::Mat blurred;
  // blur5x5_1(image, blurred);
  // cv::imshow(windowName, blurred);

  cv::Mat sobelX;
  cv::Mat sobelXDisplay;
  sobelX3x3(image, sobelX);
  cv::convertScaleAbs(sobelX, sobelXDisplay);
  cv::imshow("X", sobelXDisplay);

  cv::Mat sobelY;
  cv::Mat sobelYDisplay;
  sobelY3x3(image, sobelY);
  cv::convertScaleAbs(sobelY, sobelYDisplay);
  cv::imshow("Y", sobelYDisplay);

  while (true) {
    char key = (char)cv::waitKey(0);
    if (key == 'q' || key == 'Q') {
      return 0;
    }
  }

  return 0;
}
