/*
  Author:  Sai Supriya Kotturu
  Date:    2026-09-21
  Purpose: Definitions of the image-manipulation functions declared in
          filters.h. All pixel-level work lives in this file.
*/

#include "filters.h"

#include <numeric>
#include <opencv2/imgproc.hpp>

// sepia helpers are internal to this translation unit
namespace {

uchar getBlueCoefficient(const cv::Vec3b& px) {
  return static_cast<uchar>(
      MIN(0.131 * px[0] + 0.534 * px[1] + 0.272 * px[2], 255));
}

uchar getGreenCoefficient(const cv::Vec3b& px) {
  return static_cast<uchar>(
      MIN(0.168 * px[0] + 0.686 * px[1] + 0.349 * px[2], 255));
}

uchar getRedCoefficient(const cv::Vec3b& px) {
  return static_cast<uchar>(
      MIN(0.189 * px[0] + 0.769 * px[1] + 0.393 * px[2], 255));
}

}  // namespace

int applyGrayscaleFilter(cv::Mat& src, cv::Mat& dest) {
  cv::cvtColor(src, dest, cv::COLOR_BGR2GRAY);
  return 0;
}

int applyCustomGrayscaleFilter(cv::Mat& src, cv::Mat& dest) {
  dest.create(src.rows, src.cols, CV_8UC1);

  for (int r = 0; r < src.rows; r++) {
    const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(r);
    uchar* destRow = dest.ptr<uchar>(r);

    for (int c = 0; c < src.cols; c++) {
      const cv::Vec3b& px = srcRow[c];

      int delta = 255 - px[1];
      int avg = (px[0] + px[1] + px[2]) / 3;

      // wraparound is intentional - see filters.h
      destRow[c] = static_cast<uchar>(delta + avg);
    }
  }

  return 0;
}

int applyXRayFilter(cv::Mat& src, cv::Mat& dest) {
  dest.create(src.rows, src.cols, CV_8UC1);

  for (int r = 0; r < src.rows; r++) {
    const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(r);
    uchar* destRow = dest.ptr<uchar>(r);

    for (int c = 0; c < src.cols; c++) {
      destRow[c] = static_cast<uchar>(255 - srcRow[c][1]);
    }
  }

  return 0;
}

int applySepiaFilter(cv::Mat& src, cv::Mat& dest) {
  dest.create(src.rows, src.cols, CV_8UC3);

  for (int r = 0; r < src.rows; r++) {
    const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(r);
    cv::Vec3b* destRow = dest.ptr<cv::Vec3b>(r);

    for (int c = 0; c < src.cols; c++) {
      // copy, not reference: src and dest may be the same Mat, and all three
      // channels must be computed from the original values
      cv::Vec3b px = srcRow[c];

      destRow[c][0] = getBlueCoefficient(px);
      destRow[c][1] = getGreenCoefficient(px);
      destRow[c][2] = getRedCoefficient(px);
    }
  }

  return 0;
}

int applyVignette(cv::Mat& src, cv::Mat& dest, double inner /* = 0.3 */,
                  double strength /* = 0.8*/) {
  dest.create(src.rows, src.cols, src.type());

  // get the center position of the frame
  const double cx = src.cols / 2.0;
  const double cy = src.rows / 2.0;

  // The farthest pixel from the center is a corner.
  // From the center (cy, cx) to the corner (0, 0)
  const double maxDist = std::sqrt(cx * cx + cy * cy);
  const int chans = src.channels();

  for (int r = 0; r < src.rows; r++) {
    // const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(r);
    // cv::Vec3b* destRow = dest.ptr<cv::Vec3b>(r);

    // use uchar* instead of Vec3b to handle grayscale, Xray filters which are
    // single channel
    const uchar* srcRow = src.ptr<uchar>(r);
    uchar* destRow = dest.ptr<uchar>(r);

    for (int c = 0; c < src.cols; c++) {
      // offset of this pixel from the frame center (cy, cx)
      double dy = r - cy;
      double dx = c - cx;

      // norm: distance as a fraction of the center-to-corner distance, 0 at
      // the center and 1 at the corners. Being unitless, the same inner and
      // strength give the same look at any resolution.
      double norm = std::sqrt(dx * dx + dy * dy) / maxDist;

      // t: how far into the fade band [inner, 1] this pixel is. 0 means
      // inside the untouched inner radius, 1 means at a corner.
      double t = std::clamp((norm - inner) / (1.0 - inner), 0.0, 1.0);

      // factor: brightness multiplier, 1 = unchanged, 1 - strength = darkest.
      // smoothstep t*t*(3 - 2t) has zero slope at both ends, so the fade
      // starts without the visible ring that a linear ramp leaves at inner.
      double factor = 1.0 - strength * t * t * (3.0 - 2.0 * t);

      for (int k = 0; k < chans; k++)
        destRow[c * chans + k] =
            cv::saturate_cast<uchar>(srcRow[c * chans + k] * factor);
    }
  }

  return 0;
}

int blur5x5_1(cv::Mat& src, cv::Mat& dest) {
  // check if the image has 3 channels (BGR)
  CV_Assert((src.type() == CV_8UC3));

  src.copyTo(dest);

  const int rows = src.rows;
  const int cols = src.cols;
  const int chans = src.channels();

  // weights sum to 100; static const so it's built once, not per call
  static const int filter[5][5] = {
      {1, 2, 4, 2, 1}, {2, 4, 8, 4, 2}, {4, 8, 16, 8, 4},
      {2, 4, 8, 4, 2}, {1, 2, 4, 2, 1},
  };

  static const int* begin = &filter[0][0];
  static const int* end = &filter[0][0] + 25;

  static const int filterSum = std::accumulate(begin, end, 0);

  for (int r = 2; r < rows - 2; r++) {
    for (int c = 2; c < cols - 2; c++) {
      for (int k = 0; k < chans; k++) {
        int corr = 0;

        for (int i = 0; i < 5; i++) {
          for (int j = 0; j < 5; j++) {
            corr += filter[i][j] * src.at<cv::Vec3b>(r + i - 2, c + j - 2)[k];
          }
        }

        dest.at<cv::Vec3b>(r, c)[k] = static_cast<uchar>(corr / filterSum);
      }
    }
  }

  return 0;
}

int blur5x5_2(cv::Mat& src, cv::Mat& dest) {
  // check if the image has 3 channels (BGR)
  CV_Assert((src.type() == CV_8UC3));

  // border rows/cols keep the original pixels so they're non-zero
  src.copyTo(dest);

  const int rows = src.rows;
  const int cols = src.cols;

  // one 1D kernel used in both directions; direction comes from which index
  // (row or column) gets the offset. 5x5 kernel = outer product w x w.
  static const int w[5] = {1, 2, 4, 2, 1};

  // horizontal sums reach 255 * 10 = 2550, too big for uchar, so the
  // intermediate is signed 16-bit. Division is deferred to the end (by 100)
  // so rounding happens once and the result matches blur5x5_1 exactly.
  cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);

  // horizontal pass: every row, because the vertical pass reads temp rows
  // r-2..r+2 for dest rows 2..rows-3, i.e. all rows 0..rows-1
  for (int r = 0; r < rows; r++) {
    const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(r);
    cv::Vec3s* tRow = temp.ptr<cv::Vec3s>(r);

    for (int c = 2; c < cols - 2; c++) {
      for (int k = 0; k < 3; k++) {
        int sum = 0;
        for (int j = 0; j < 5; j++) {
          sum += w[j] * srcRow[c + j - 2][k];
        }
        tRow[c][k] = static_cast<short>(sum);
      }
    }
  }

  // vertical pass: 5 row pointers set once per row, not per pixel
  for (int r = 2; r < rows - 2; r++) {
    const cv::Vec3s* tRows[5];
    for (int i = 0; i < 5; i++) {
      tRows[i] = temp.ptr<cv::Vec3s>(r + i - 2);
    }
    cv::Vec3b* destRow = dest.ptr<cv::Vec3b>(r);

    for (int c = 2; c < cols - 2; c++) {
      for (int k = 0; k < 3; k++) {
        int sum = 0;  // max 2550 * 10 = 25500, needs int
        for (int i = 0; i < 5; i++) {
          sum += w[i] * tRows[i][c][k];
        }
        destRow[c][k] = static_cast<uchar>(sum / 100);
      }
    }
  }

  return 0;
}

int applyBlur(cv::Mat& src, cv::Mat& dest) {
  blur5x5_2(src, dest);
  return 0;
}