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
 * Vignette: darkens pixels by their normalized distance from the frame center,
 * with a smoothstep fade so there is no visible ring. Works on any 8-bit image
 * (1 or 3 channels), so it can be layered on top of any other filter.
 *
 * @param src       input frame, CV_8UC1 or CV_8UC3
 * @param dest      output frame, same type as src; may be the same Mat as src
 * @param inner     radius (0..1 of center-to-corner distance) left untouched
 * @param strength  how dark the corners get (0 = none, 1 = black)
 * @return 0 on success
 */
int applyVignette(cv::Mat& src, cv::Mat& dest, double inner = 0.3,
                  double strength = 0.8);

/**
 * Blur used by the live video stream; delegates to the fast separable
 * blur5x5_2.
 *
 * @param src   input frame, CV_8UC3 (BGR); not modified
 * @param dest  output frame, CV_8UC3 (BGR)
 * @return 0 on success
 */
int applyBlur(cv::Mat& src, cv::Mat& dest);

/**
 * Naive 5x5 Gaussian blur (task 6A). Full 5x5 integer kernel
 * [1 2 4 2 1] x [1 2 4 2 1] (weights sum to 100), applied per channel using
 * at() for pixel access. The outer 2 rows/cols are copied from src.
 *
 * @param src   input image, CV_8UC3 (BGR); not modified
 * @param dst   output image, CV_8UC3 (BGR), same size as src
 * @return 0 on success
 */
int blur5x5_1(cv::Mat& src, cv::Mat& dst);

/**
 * Separable 5x5 Gaussian blur (task 6B). Same result as blur5x5_1, computed as
 * a horizontal then a vertical [1 2 4 2 1] pass through a 16-bit temporary,
 * with row pointers instead of at(). 10 multiply-adds per pixel instead of 25.
 * The outer 2 rows/cols are copied from src.
 *
 * @param src   input image, CV_8UC3 (BGR); not modified
 * @param dst   output image, CV_8UC3 (BGR), same size as src
 * @return 0 on success
 */
int blur5x5_2(cv::Mat& src, cv::Mat& dst);

/**
 * 3x3 Sobel X (horizontal gradient) as separable 1x3 filters: [-1 0 1]
 * across columns, then [1 2 1] down rows. Positive where brightness increases
 * to the right; highlights vertical edges. The 1-pixel border is left at 0.
 *
 * @param src   input frame, CV_8UC3 (BGR); not modified
 * @param dest  output gradient, CV_16SC3, values in [-255, 255]; use
 *              cv::convertScaleAbs to display it
 * @return 0 on success
 */
int sobelX3x3(cv::Mat& src, cv::Mat& dest);

/**
 * 3x3 Sobel Y (vertical gradient) as separable 1x3 filters: [1 2 1] across
 * columns, then [1 0 -1] down rows (row above minus row below). Positive where
 * brightness increases upward; highlights horizontal edges. The 1-pixel
 * border is left at 0.
 *
 * @param src   input frame, CV_8UC3 (BGR); not modified
 * @param dest  output gradient, CV_16SC3, values in [-255, 255]; use
 *              cv::convertScaleAbs to display it
 * @return 0 on success
 */
int sobelY3x3(cv::Mat& src, cv::Mat& dest);

/**
 * Gradient magnitude from the X and Y Sobel images: per pixel and channel,
 * sqrt(sx^2 + sy^2). Edge strength in any direction, independent of sign.
 * Values above 255 (up to ~360 on strong diagonal edges) are clamped.
 *
 * @param sx    Sobel X output, CV_16SC3
 * @param sy    Sobel Y output, CV_16SC3, same size as sx
 * @param dest  output image, CV_8UC3, suitable for imshow
 * @return 0 on success
 */
int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dest);

/**
 * Blurs the image (blur5x5_2) and then quantizes each color channel into a
 * fixed number of levels: with b = 255 / levels, each value x becomes
 * (x / b) * b. The result has at most levels^3 colors, giving a posterized,
 * cartoon-like look. Blurring first keeps noise from causing speckles.
 *
 * @param src     input image, CV_8UC3 (BGR); not modified
 * @param dest    output image, CV_8UC3 (BGR)
 * @param levels  number of levels per channel, 1..255 (10 is a good default)
 * @return 0 on success
 */
int blurQuantize(cv::Mat& src, cv::Mat& dest, int levels = 10);

#endif  // FILTERS_H
