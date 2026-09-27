/*
  Author:  Sai Supriya Kotturu
  Date:    2026-09-21
  Purpose: Definitions of the image-manipulation functions declared in
          filters.h. All pixel-level work lives in this file.
*/

#include "filters.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/imgproc.hpp>

// sepia helpers are internal to this translation unit. Each takes one
// original BGR pixel and returns the new sepia value for one channel,
// clamped to 255 because the green/red weights sum to more than 1.
namespace {

// new blue = 0.272 R + 0.534 G + 0.131 B
uchar getBlueCoefficient(const cv::Vec3b& px) {
  return static_cast<uchar>(
      MIN(0.131 * px[0] + 0.534 * px[1] + 0.272 * px[2], 255));
}

// new green = 0.349 R + 0.686 G + 0.168 B
uchar getGreenCoefficient(const cv::Vec3b& px) {
  return static_cast<uchar>(
      MIN(0.168 * px[0] + 0.686 * px[1] + 0.349 * px[2], 255));
}

// new red = 0.393 R + 0.769 G + 0.189 B
uchar getRedCoefficient(const cv::Vec3b& px) {
  return static_cast<uchar>(
      MIN(0.189 * px[0] + 0.769 * px[1] + 0.393 * px[2], 255));
}

}  // namespace

// OpenCV grayscale (task 3). Returns 0 on success.
int applyGrayscaleFilter(cv::Mat& src, cv::Mat& dest) {
  cv::cvtColor(src, dest, cv::COLOR_BGR2GRAY);
  return 0;
}

// Custom grayscale (task 4): (255 - G) + mean(B,G,R), wrapped to 8 bits.
// src is CV_8UC3, dest becomes CV_8UC1. Returns 0 on success.
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

// X-ray negative from the green channel: 255 - G. src is CV_8UC3, dest
// becomes CV_8UC1. Returns 0 on success.
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

// Sepia tone (task 5) computed from the original BGR values of each pixel.
// src and dest are CV_8UC3. Returns 0 on success.
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

// Vignette extension: darkens by normalized distance from center. Works on
// 1- or 3-channel 8-bit frames; inner/strength in 0..1. Returns 0 on success.
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

// Naive 5x5 Gaussian blur (task 6A) using at(). src/dest are CV_8UC3; border
// copied from src. Returns 0 on success.
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

  // for each interior pixel and channel: weighted sum of the 5x5
  // neighborhood, accumulated in an int (up to 25500) then divided by 100
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

// Separable 5x5 Gaussian blur (task 6B) using row pointers. src/dest are
// CV_8UC3; border copied from src. Returns 0 on success.
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

// Blur for the live stream; uses the fast separable version. Returns 0.
int applyBlur(cv::Mat& src, cv::Mat& dest) {
  blur5x5_2(src, dest);
  return 0;
}

// Sobel X (task 7) as separable 1x3 filters, positive to the right. src is
// CV_8UC3, dest becomes CV_16SC3 in [-255, 255]. Returns 0 on success.
int sobelX3x3(cv::Mat& src, cv::Mat& dest) {
  // check if the image has 3 channels (BGR)
  CV_Assert((src.type() == CV_8UC3));

  const int rows = src.rows;
  const int cols = src.cols;

  // separable Sobel X = [1 2 1]^T (vertical smooth) x [-1 0 1] (horizontal
  // difference). right - left, so positive when brightness increases rightward
  static const int h[3] = {-1, 0, 1};
  static const int v[3] = {1, 2, 1};

  // output is signed: edges can be dark->bright (+) or bright->dark (-).
  // zeros leaves the 1-pixel border at 0 (not required by the spec)
  dest = cv::Mat::zeros(src.size(), CV_16SC3);

  // horizontal difference results are in [-255, 255]
  cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);

  // horizontal pass: every row, because the vertical pass reads temp rows
  // r-1..r+1 for dest rows 1..rows-2, i.e. all rows 0..rows-1
  for (int r = 0; r < rows; r++) {
    const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(r);
    cv::Vec3s* tRow = temp.ptr<cv::Vec3s>(r);

    for (int c = 1; c < cols - 1; c++) {
      for (int k = 0; k < 3; k++) {
        int sum = 0;
        for (int j = 0; j < 3; j++) {
          sum += h[j] * srcRow[c + j - 1][k];
        }
        tRow[c][k] = static_cast<short>(sum);
      }
    }
  }

  // vertical pass: 3 row pointers set once per row, not per pixel
  for (int r = 1; r < rows - 1; r++) {
    const cv::Vec3s* tRows[3];
    for (int i = 0; i < 3; i++) {
      tRows[i] = temp.ptr<cv::Vec3s>(r + i - 1);
    }
    cv::Vec3s* destRow = dest.ptr<cv::Vec3s>(r);

    for (int c = 1; c < cols - 1; c++) {
      for (int k = 0; k < 3; k++) {
        int sum = 0;  // [1 2 1] sums to 4, so range is [-1020, 1020]
        for (int i = 0; i < 3; i++) {
          sum += v[i] * tRows[i][c][k];
        }
        // divide by the smoothing weight to bring it back to [-255, 255]
        destRow[c][k] = static_cast<short>(sum / 4);
      }
    }
  }

  return 0;
}

// Sobel Y (task 7) as separable 1x3 filters, positive up. src is
// CV_8UC3, dest becomes CV_16SC3 in [-255, 255]. Returns 0 on success.
int sobelY3x3(cv::Mat& src, cv::Mat& dest) {
  // check if the image has 3 channels (BGR)
  CV_Assert((src.type() == CV_8UC3));

  const int rows = src.rows;
  const int cols = src.cols;

  // separable Sobel Y = [1 0 -1]^T (vertical difference) x [1 2 1]
  // (horizontal smooth). Row indices grow downward, so "positive up" means
  // above - below: the row above (tRows[0]) gets +1.
  static const int h[3] = {1, 2, 1};
  static const int v[3] = {1, 0, -1};

  // output is signed: brighter above (+) or brighter below (-).
  // zeros leaves the 1-pixel border at 0 (not required by the spec)
  dest = cv::Mat::zeros(src.size(), CV_16SC3);

  // horizontal smoothing results are in [0, 1020] (weights sum to 4)
  cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);

  // horizontal pass: every row, because the vertical pass reads temp rows
  // r-1..r+1 for dest rows 1..rows-2, i.e. all rows 0..rows-1
  for (int r = 0; r < rows; r++) {
    const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(r);
    cv::Vec3s* tRow = temp.ptr<cv::Vec3s>(r);

    for (int c = 1; c < cols - 1; c++) {
      for (int k = 0; k < 3; k++) {
        int sum = 0;
        for (int j = 0; j < 3; j++) {
          sum += h[j] * srcRow[c + j - 1][k];
        }
        tRow[c][k] = static_cast<short>(sum);
      }
    }
  }

  // vertical pass: 3 row pointers set once per row, not per pixel
  for (int r = 1; r < rows - 1; r++) {
    const cv::Vec3s* tRows[3];
    for (int i = 0; i < 3; i++) {
      tRows[i] = temp.ptr<cv::Vec3s>(r + i - 1);
    }
    cv::Vec3s* destRow = dest.ptr<cv::Vec3s>(r);

    for (int c = 1; c < cols - 1; c++) {
      for (int k = 0; k < 3; k++) {
        int sum = 0;  // above - below of values in [0, 1020]: [-1020, 1020]
        for (int i = 0; i < 3; i++) {
          sum += v[i] * tRows[i][c][k];
        }
        // divide by the horizontal smoothing weight (4) -> [-255, 255]
        destRow[c][k] = static_cast<short>(sum / 4);
      }
    }
  }

  return 0;
}

// Gradient magnitude (task 8): per channel sqrt(sx^2 + sy^2), clamped to 255.
// sx/sy are CV_16SC3 Sobel outputs, dest becomes CV_8UC3. Returns 0 on success.
int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dest) {
  // both inputs must be signed Sobel outputs of the same size
  CV_Assert(sx.type() == CV_16SC3 && sy.type() == CV_16SC3);
  CV_Assert(sx.size() == sy.size());

  dest.create(sx.size(), CV_8UC3);

  for (int r = 0; r < sx.rows; r++) {
    const cv::Vec3s* sxRow = sx.ptr<cv::Vec3s>(r);
    const cv::Vec3s* syRow = sy.ptr<cv::Vec3s>(r);
    cv::Vec3b* destRow = dest.ptr<cv::Vec3b>(r);

    for (int c = 0; c < sx.cols; c++) {
      for (int k = 0; k < 3; k++) {
        // squares reach 2 * 255^2 = 130050, fine in int
        int x = sxRow[c][k];
        int y = syRow[c][k];

        // max is 255 * sqrt(2) ~ 360. Clamping (not scaling) keeps typical
        // edges at full brightness; only strong diagonals saturate at 255.
        destRow[c][k] = cv::saturate_cast<uchar>(
            std::sqrt(static_cast<float>(x * x + y * y)));
      }
    }
  }

  return 0;
}

// Blur + quantize (task 9): blurs with blur5x5_2, then snaps each channel to
// one of `levels` buckets. src/dest are CV_8UC3. Returns 0 on success.
int blurQuantize(cv::Mat& src, cv::Mat& dest, int levels) {
  // levels <= 0 divides by zero; levels > 255 makes the bucket size 0
  CV_Assert(levels > 0 && levels <= 255);

  // blur first so sensor noise doesn't flip pixels between adjacent buckets
  // (speckles in still images, flicker in video)
  blur5x5_2(src, dest);

  // integer division drops the remainder, so x / b * b is the bottom of x's
  // bucket; e.g. levels = 10 -> b = 25 -> 125..149 all become 125
  const int b = 255 / levels;

  // in place: each pixel only reads its own value
  for (int r = 0; r < dest.rows; r++) {
    cv::Vec3b* row = dest.ptr<cv::Vec3b>(r);
    for (int c = 0; c < dest.cols; c++) {
      for (int k = 0; k < 3; k++) {
        row[c][k] = static_cast<uchar>(row[c][k] / b * b);
      }
    }
  }

  return 0;
}
