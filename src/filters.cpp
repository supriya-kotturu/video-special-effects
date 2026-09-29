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
#include <vector>

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

// Soft face mask shared by the face-aware effects: 1 away from faces, 0 over
// them. Builds an oval slightly larger than each face box with a smoothstep
// falloff (a hard box looked like a cut-out), then folds it into `smooth`, a
// running average owned by the caller, so the oval doesn't wobble with the
// detector's per-frame jitter. smooth becomes CV_32F of the given size.
static void faceOvalMask(const cv::Size& size,
                         const std::vector<cv::Rect>& faces, cv::Mat& smooth) {
  const int rows = size.height;
  const int cols = size.width;
  cv::Mat mask(rows, cols, CV_32F, cv::Scalar(1.0f));
  for (const cv::Rect& f : faces) {
    const float cx = f.x + f.width * 0.5f;
    const float cy = f.y + f.height * 0.5f;
    const float ax = f.width * 0.75f;   // half-width incl. hair/ears
    const float ay = f.height * 0.95f;  // half-height incl. forehead/chin
    const cv::Rect area =
        cv::Rect(cvRound(cx - ax * 1.3f), cvRound(cy - ay * 1.3f),
                 cvRound(ax * 2.6f), cvRound(ay * 2.6f)) &
        cv::Rect(0, 0, cols, rows);
    for (int r = area.y; r < area.y + area.height; r++) {
      float* mRow = mask.ptr<float>(r);
      for (int c = area.x; c < area.x + area.width; c++) {
        // normalized ellipse distance: 1.0 on the oval's edge
        const float dx = (c - cx) / ax;
        const float dy = (r - cy) / ay;
        const float d = std::sqrt(dx * dx + dy * dy);
        float t = std::clamp((d - 0.8f) / (1.2f - 0.8f), 0.0f, 1.0f);
        t = t * t * (3.0f - 2.0f * t);
        mRow[c] = std::min(mRow[c], t);
      }
    }
  }

  if (smooth.size() != mask.size()) {
    mask.copyTo(smooth);
  }
  cv::addWeighted(smooth, 0.7, mask, 0.3, 0, smooth);
}

// "Seattle rain" (task 12 effect, uses face detection): semi-transparent rain
// streaks over the frame that fade out over detected faces through a soft
// oval. src is CV_8UC3, faces are detector boxes (may be empty), dest becomes
// CV_8UC3. Returns 0 on success.
int rainEffect(cv::Mat& src, cv::Mat& dest,
               const std::vector<cv::Rect>& faces) {
  CV_Assert(src.type() == CV_8UC3);

  const int rows = src.rows;
  const int cols = src.cols;

  // ---- 1. rain mask: 1 = rain shows, 0 = keep the face clear ---------------
  static cv::Mat smoothMask;
  faceOvalMask(src.size(), faces, smoothMask);

  // ---- 2. rain streaks, persistent across frames ---------------------------
  // z in [0.3, 1] is a pseudo-distance per drop: nearer drops fall faster and
  // are longer and brighter, which reads as parallax
  struct Drop {
    float x, y, z;
  };
  static std::vector<Drop> drops;
  static cv::RNG rng(5330);
  static cv::Size dropArea;
  const float wind = 0.15f;  // slight slant
  if (dropArea != src.size()) {
    dropArea = src.size();
    drops.assign(350, Drop{});
    for (Drop& d : drops) {
      d = {rng.uniform(0.0f, (float)cols), rng.uniform(0.0f, (float)rows),
           rng.uniform(0.3f, 1.0f)};
    }
  }

  cv::Mat rain = cv::Mat::zeros(src.size(), CV_8UC1);
  for (Drop& d : drops) {
    const float speed = 8.0f + 14.0f * d.z;
    const float len = 8.0f + 18.0f * d.z;
    d.y += speed;
    d.x += speed * wind;
    if (d.y - len > rows || d.x > cols) {  // off screen: start again at the top
      d.y = rng.uniform(-40.0f, 0.0f);
      d.x = rng.uniform(-0.2f * rows * wind, (float)cols);
    }
    // brightness added on top of the frame: faint enough to stay see-through
    const int v = static_cast<int>(40 + 90 * d.z);
    cv::line(rain, cv::Point2f(d.x, d.y),
             cv::Point2f(d.x - len * wind, d.y - len), cv::Scalar(v), 1,
             cv::LINE_AA);
  }

  // ---- 3. composite: add the streaks, scaled by the mask -------------------
  // additive with a slight blue-white tint, so the original frame shows
  // through every streak (transparent rain) rather than being covered
  const float TINT[3] = {1.0f, 0.95f, 0.9f};  // BGR
  dest.create(src.size(), CV_8UC3);
  for (int r = 0; r < rows; r++) {
    const cv::Vec3b* sRow = src.ptr<cv::Vec3b>(r);
    const uchar* rRow = rain.ptr<uchar>(r);
    const float* mRow = smoothMask.ptr<float>(r);
    cv::Vec3b* dRow = dest.ptr<cv::Vec3b>(r);
    for (int c = 0; c < cols; c++) {
      const float add = rRow[c] * mRow[c];
      for (int k = 0; k < 3; k++) {
        dRow[c][k] = cv::saturate_cast<uchar>(sRow[c][k] + add * TINT[k]);
      }
    }
  }

  return 0;
}

// Depth fog (task 11): blends each pixel toward a fog color by
// 1 - exp(-density * distance). src is CV_8UC3, depth is CV_8UC1 (bright =
// near), dest becomes CV_8UC3. Returns 0 on success.
int fogEffect(cv::Mat& src, cv::Mat& depth, cv::Mat& dest, float density) {
  CV_Assert(src.type() == CV_8UC3 && depth.type() == CV_8UC1);
  CV_Assert(src.size() == depth.size());

  // DA2's per-frame normalization makes raw depth flicker; a running average
  // keeps the fog from pulsing
  static cv::Mat smoothDepth;
  cv::Mat depthF;
  depth.convertTo(depthF, CV_32F, 1.0 / 255.0);
  if (smoothDepth.size() != depthF.size()) {
    depthF.copyTo(smoothDepth);
  }
  cv::addWeighted(smoothDepth, 0.7, depthF, 0.3, 0, smoothDepth);

  // cool light grey, like marine-layer fog (BGR)
  const float FOG[3] = {215.0f, 208.0f, 200.0f};

  dest.create(src.size(), CV_8UC3);
  for (int r = 0; r < src.rows; r++) {
    const cv::Vec3b* sRow = src.ptr<cv::Vec3b>(r);
    const float* zRow = smoothDepth.ptr<float>(r);
    cv::Vec3b* dRow = dest.ptr<cv::Vec3b>(r);
    for (int c = 0; c < src.cols; c++) {
      // depth is closeness, so distance is its complement, in [0, 1]
      const float distance = 1.0f - zRow[c];

      // Beer-Lambert: light surviving through fog decays exponentially with
      // distance, so fog amount rises quickly at first and then saturates
      const float f = 1.0f - std::exp(-density * distance);

      for (int k = 0; k < 3; k++) {
        dRow[c][k] =
            cv::saturate_cast<uchar>((1.0f - f) * sRow[c][k] + f * FOG[k]);
      }
    }
  }

  return 0;
}

// Neon duotone (task 12 effect, area-based): green/dark-green duotone on
// detailed regions (faces, hands, edges) and magenta/pale-yellow duotone on
// smooth regions (walls). src/dest are CV_8UC3. Returns 0 on success.
int neonCartoon(cv::Mat& src, cv::Mat& dest) {
  CV_Assert(src.type() == CV_8UC3);

  const int rows = src.rows;
  const int cols = src.cols;

  // ---- 1. detail map from our own Sobel + magnitude -------------------------
  // raw magnitude is only the outlines; a light blur spreads it into textured
  // areas (face, hands, fabric) so whole regions switch palette, while smooth
  // walls stay near 0
  cv::Mat sx, sy, mag, magGrey, detail;
  sobelX3x3(src, sx);
  sobelY3x3(src, sy);
  magnitude(sx, sy, mag);
  cv::cvtColor(mag, magGrey, cv::COLOR_BGR2GRAY);
  magGrey.convertTo(detail, CV_32F, 1.0 / 255.0);
  cv::GaussianBlur(detail, detail, cv::Size(0, 0), 3.0);

  // smoothstep between lo and hi: soft transitions instead of hard contours
  auto ramp = [](float v, float lo, float hi) {
    float t = std::clamp((v - lo) / (hi - lo), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
  };

  // BGR duotone palettes: {dark end, light end}
  const cv::Vec3f MAGENTA(190, 50, 205), PALE_YELLOW(165, 228, 238);
  const cv::Vec3f GREEN_DARK(25, 70, 30), GREEN_LIGHT(60, 215, 75);

  dest.create(src.size(), CV_8UC3);
  for (int r = 0; r < rows; r++) {
    const cv::Vec3b* sRow = src.ptr<cv::Vec3b>(r);
    const uchar* mRow = magGrey.ptr<uchar>(r);
    const float* detRow = detail.ptr<float>(r);
    cv::Vec3b* dRow = dest.ptr<cv::Vec3b>(r);

    for (int c = 0; c < cols; c++) {
      const cv::Vec3b& px = sRow[c];
      const float lum =
          (0.114f * px[0] + 0.587f * px[1] + 0.299f * px[2]) / 255.0f;

      // background: brightness picks the point between magenta and yellow
      const float tb = ramp(lum, 0.30f, 0.80f);
      const cv::Vec3f bg = MAGENTA * (1.0f - tb) + PALE_YELLOW * tb;

      // subject: same idea in green; strong raw edges push toward the light
      // end, which draws a bright outline rim
      const float ts =
          std::min(1.0f, ramp(lum, 0.25f, 0.75f) + 1.2f * mRow[c] / 255.0f);
      const cv::Vec3f fg = GREEN_DARK * (1.0f - ts) + GREEN_LIGHT * ts;

      // how much this pixel belongs to the detailed "subject" palette
      const float s = ramp(detRow[c], 0.04f, 0.09f);
      const cv::Vec3f out = bg * (1.0f - s) + fg * s;

      for (int k = 0; k < 3; k++) {
        dRow[c][k] = cv::saturate_cast<uchar>(out[k]);
      }
    }
  }

  return 0;
}

// Cartoonize (task 12 effect, area-based): blurQuantize colors plus black
// outlines where the gradient magnitude exceeds magThreshold. src/dest are
// CV_8UC3. Returns 0 on success.
int cartoonize(cv::Mat& src, cv::Mat& dest, int levels, int magThreshold) {
  CV_Assert(src.type() == CV_8UC3);

  // Sobel on the raw frame fires on sensor noise, which shows up as flickering
  // black dots in flat areas; blurring first leaves only real edges
  cv::Mat smooth, sx, sy, mag;
  blur5x5_2(src, smooth);
  sobelX3x3(smooth, sx);
  sobelY3x3(smooth, sy);
  magnitude(sx, sy, mag);

  blurQuantize(src, dest, levels);

  for (int r = 0; r < dest.rows; r++) {
    const cv::Vec3b* mRow = mag.ptr<cv::Vec3b>(r);
    cv::Vec3b* dRow = dest.ptr<cv::Vec3b>(r);
    for (int c = 0; c < dest.cols; c++) {
      // max over channels so an edge between two colors of similar
      // brightness (e.g. red shirt on green wall) still gets a line
      const int strength = std::max({mRow[c][0], mRow[c][1], mRow[c][2]});
      if (strength > magThreshold) {
        dRow[c] = cv::Vec3b(0, 0, 0);
      }
    }
  }

  return 0;
}

// Median filter extension: per-channel median over a
// ksize x ksize window. src/dest are CV_8UC3; border copied from src.
// Returns 0 on success.
int medianFilter(cv::Mat& src, cv::Mat& dest, int ksize) {
  CV_Assert(src.type() == CV_8UC3);
  // even sizes have no center pixel
  CV_Assert(ksize >= 3 && ksize % 2 == 1);

  // copying first fills the border the window can't reach; also means dest
  // must not alias src, since the loop reads src neighbors
  src.copyTo(dest);

  const int half = ksize / 2;
  const int mid = ksize * ksize / 2;
  std::vector<uchar> window(ksize * ksize);

  for (int r = half; r < src.rows - half; r++) {
    cv::Vec3b* dRow = dest.ptr<cv::Vec3b>(r);
    for (int c = half; c < src.cols - half; c++) {
      for (int k = 0; k < 3; k++) {
        int n = 0;
        for (int dr = -half; dr <= half; dr++) {
          const cv::Vec3b* sRow = src.ptr<cv::Vec3b>(r + dr);
          for (int dc = -half; dc <= half; dc++) {
            window[n++] = sRow[c + dc][k];
          }
        }
        // only the middle element must be in place, not a full sort:
        // O(n) instead of O(n log n) per pixel
        std::nth_element(window.begin(), window.begin() + mid, window.end());
        dRow[c][k] = window[mid];
      }
    }
  }

  return 0;
}

// Disco lights extension (uses face detection): brightness from src, color
// from a sweeping neon rainbow, with a strobe. Faces stay in natural color
// while the room flashes, or the reverse when discoOnFace is set. t is in
// seconds. src/dest are CV_8UC3. Returns 0 on success.
int discoEffect(cv::Mat& src, cv::Mat& dest, double t,
                const std::vector<cv::Rect>& faces, bool discoOnFace,
                double strobeHz) {
  CV_Assert(src.type() == CV_8UC3);

  // one fully saturated, full-value color per OpenCV hue step (0..179);
  // built once so the per-pixel loop is a table lookup, not an HSV conversion
  static cv::Mat palette;
  if (palette.empty()) {
    cv::Mat hsv(1, 180, CV_8UC3);
    for (int h = 0; h < 180; h++) {
      hsv.at<cv::Vec3b>(0, h) = cv::Vec3b(static_cast<uchar>(h), 255, 255);
    }
    cv::cvtColor(hsv, palette, cv::COLOR_HSV2BGR);
  }
  const cv::Vec3b* pal = palette.ptr<cv::Vec3b>(0);

  // full hue cycle every 2 s; fmod keeps it small however large t gets
  const int shift = static_cast<int>(std::fmod(t * 90.0, 180.0));
  // pixels per hue step: 180 * 4 = 720 px per rainbow, about one per frame
  const int BAND = 4;

  // square wave: bright half of each period, dim the other. Dim rather than
  // black keeps the scene readable and the flash contrast lower
  const bool dim = strobeHz > 0 && std::fmod(t * strobeHz, 1.0) >= 0.5;
  // gain / 256: >1 on the bright phase because tinting darkens everything
  const int gain = dim ? 90 : 320;

  // 1 away from faces, 0 over them, soft at the oval's edge
  static cv::Mat faceMask;
  faceOvalMask(src.size(), faces, faceMask);

  dest.create(src.size(), CV_8UC3);
  for (int r = 0; r < src.rows; r++) {
    const cv::Vec3b* sRow = src.ptr<cv::Vec3b>(r);
    const float* mRow = faceMask.ptr<float>(r);
    cv::Vec3b* dRow = dest.ptr<cv::Vec3b>(r);
    for (int c = 0; c < src.cols; c++) {
      const cv::Vec3b& px = sRow[c];
      // w = how much disco this pixel gets
      const float w = discoOnFace ? 1.0f - mRow[c] : mRow[c];
      // integer luminance, weights x256
      const int lum = (29 * px[0] + 150 * px[1] + 77 * px[2]) >> 8;
      const cv::Vec3b& color = pal[((r + c) / BAND + shift) % 180];
      for (int k = 0; k < 3; k++) {
        // lum * color * gain <= 255 * 255 * 320, fits in int; >> 16 is the
        // two /256 normalizations (lum/255 and gain/256, close enough)
        const int disco = (lum * color[k] * gain) >> 16;
        dRow[c][k] =
            cv::saturate_cast<uchar>(w * disco + (1.0f - w) * px[k]);
      }
    }
  }

  return 0;
}
