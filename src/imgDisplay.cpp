/*
  Author:  Sai Supriya Kotturu
  Date:    2026-09-21
  Purpose: Task 1 - read an image from a file, display it in a window, and loop
           until the user presses 'q'.
*/

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

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

  while (true) {
    char key = (char)cv::waitKey(0);
    if (key == 'q' || key == 'Q') {
      return 0;
    }
  }

  return 0;
}
