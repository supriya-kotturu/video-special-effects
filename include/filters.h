/*
  Author:  Sai Supriya Kotturu
  Date:    2026-09-21
  Purpose: Declarations of the image-manipulation functions implemented in
          filters.cpp (grayscale, sepia, blurs, Sobel, ...). Included by
          vidDisplay.cpp so it can call them.
*/

#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/core.hpp>

/**
 * Converts a BGR frame to grayscale using OpenCV's standard luminance weights
 * (0.299 R + 0.587 G + 0.114 B).
 *
 * @param src   input frame, CV_8UC3 (BGR)
 * @param dest  output frame, CV_8UC1; (re)allocated to match src
 * @return 0 on success
 */
int applyGrayscaleFilter(cv::Mat& src, cv::Mat& dest);

/**
 * Custom grayscale: per pixel, (255 - G) + mean(B, G, R), truncated to 8 bits.
 *
 * The sum is deliberately NOT saturated. For near-neutral pixels it lands
 * right around 255, so saturating would give an almost uniformly white frame;
 * letting it wrap is what produces the visible, darker, high-contrast result.
 *
 * @param src   input frame, CV_8UC3 (BGR)
 * @param dest  output frame, CV_8UC1; (re)allocated to match src
 * @return 0 on success
 */
int applyCustomGrayscaleFilter(cv::Mat& src, cv::Mat& dest);

/**
 * X-ray style negative built from the green channel only: 255 - G.
 *
 * @param src   input frame, CV_8UC3 (BGR)
 * @param dest  output frame, CV_8UC1; (re)allocated to match src
 * @return 0 on success
 */
int applyXRayFilter(cv::Mat& src, cv::Mat& dest);

/**
 * Classic sepia tone. Each output channel is a weighted sum of the ORIGINAL
 * B, G, R values (not already-modified ones), clamped to 255:
 *
 *   B' = 0.131 B + 0.534 G + 0.272 R
 *   G' = 0.168 B + 0.686 G + 0.349 R
 *   R' = 0.189 B + 0.769 G + 0.393 R
 *
 * @param src   input frame, CV_8UC3 (BGR)
 * @param dest  output frame, CV_8UC3 (BGR); (re)allocated to match src
 * @return 0 on success
 */
int applySepiaFilter(cv::Mat& src, cv::Mat& dest);

/**
 *  Darkens pixels by distance from center
 *
 * @param src   input frame, CV_8UC3 (BGR)
 * @param dest  output frame, CV_8UC3 (BGR); (re)allocated to match src
 * @param inner inner untouched radius of the image (0..1)
 * @param strength how dark the corners get (0..1)
 * @return 0 on success
 */
int applyVignette(cv::Mat& src, cv::Mat& dest, double inner = 0.3,
                  double strength = 0.8);

/** Applies Gaussian Blur on the scr image
 *
 * @param src input frame, CV_8UC3 (BGR)
 * @param dest  output frame, CV_8UC3 (BGR); (re)allocated to match src
 */
int applyBlur(cv::Mat& src, cv::Mat& dest);

// prototypes for the functions to test
int blur5x5_1(cv::Mat& src, cv::Mat& dst);
int blur5x5_2(cv::Mat& src, cv::Mat& dst);

#endif  // FILTERS_H
