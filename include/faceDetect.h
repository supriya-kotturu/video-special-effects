/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Include file for faceDetect.cpp, face detection and drawing functions

  Modified by Sai Supriya Kotturu, 2026-09-27: added the includes this header
  needs on its own (it's included before OpenCV in faceDetect.cpp), and
  pointed the cascade path at data/ (relative to the repo root, where the
  programs are run from).
*/
#ifndef FACEDETECT_H
#define FACEDETECT_H

#include <opencv2/core.hpp>
#include <vector>

// put the path to the haar cascade file here
#define FACE_CASCADE_FILE "data/haarcascade_frontalface_alt2.xml"

// prototypes
int detectFaces(cv::Mat& grey, std::vector<cv::Rect>& faces);
int drawBoxes(cv::Mat& frame, std::vector<cv::Rect>& faces, int minWidth = 50,
              float scale = 1.0);

#endif
