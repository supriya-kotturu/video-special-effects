/*
  Author:  Sai Supriya Kotturu
  Date:    2026-10-08
  Purpose: Compare two textures by the average energy of their Sobel gradient
          magnitude (project 1 filters), showing the magnitude images.
*/

#include <filters.h>

#include <cmath>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

// Energy = mean over pixels and channels of the magnitude (and its square).
// The square weights strong edges more, which separates "few sharp edges"
// from "many weak ones" better than the plain mean.
void report(const std::string& name, cv::Mat& texture, cv::Mat& mag,
            cv::Mat& sx, cv::Mat& sy) {
  cv::Mat m32;
  mag.convertTo(m32, CV_32F);
  cv::Scalar ch = cv::mean(m32);
  double meanMag = (ch[0] + ch[1] + ch[2]) / 3.0;

  cv::Mat sq = m32.mul(m32);
  cv::Scalar chSq = cv::mean(sq);
  double meanSq = (chSq[0] + chSq[1] + chSq[2]) / 3.0;

  // intensity spread, to expose the contrast confound
  cv::Mat gray;
  cv::cvtColor(texture, gray, cv::COLOR_BGR2GRAY);
  cv::Scalar mu, sigma;
  cv::meanStdDev(gray, mu, sigma);

  // Gx vs Gy split: magnitude hides orientation, this ratio recovers it
  cv::Scalar ax = cv::mean(cv::abs(sx));
  cv::Scalar ay = cv::mean(cv::abs(sy));
  double meanAx = (ax[0] + ax[1] + ax[2]) / 3.0;
  double meanAy = (ay[0] + ay[1] + ay[2]) / 3.0;

  std::cout << name << ": mean|grad|=" << meanMag << "  mean|grad|^2=" << meanSq
            << "  gray stddev=" << sigma[0] << "  mean|Gx|=" << meanAx
            << "  mean|Gy|=" << meanAy << std::endl;
}

// Loads path, crops to the ROI (w == 0 keeps the whole image) and writes the
// crop to dstPath so the exact patch can be reused in the report.
bool load(const char* path, int x, int y, int w, int h, const char* dstPath,
          cv::Mat& out) {
  cv::Mat img;
  const std::string p(path);
  if (p == "synth:v" || p == "synth:h") {
    // sine stripes, 8 px period; same pattern rotated 90 deg for "h"
    img.create(100, 100, CV_8UC3);
    for (int r = 0; r < img.rows; r++) {
      for (int c = 0; c < img.cols; c++) {
        int t = (p == "synth:v") ? c : r;
        uchar v =
            cv::saturate_cast<uchar>(128 + 100 * std::sin(t * 2 * CV_PI / 8));
        img.at<cv::Vec3b>(r, c) = cv::Vec3b(v, v, v);
      }
    }
  } else {
    img = cv::imread(path, cv::IMREAD_COLOR);
  }
  if (img.empty()) {
    std::cerr << "Error: cannot read " << path << std::endl;
    return false;
  }
  if (w > 0) {
    cv::Rect roi(x, y, w, h);
    if ((roi & cv::Rect(0, 0, img.cols, img.rows)) != roi) {
      std::cerr << "Error: ROI outside " << path << std::endl;
      return false;
    }
    img = img(roi).clone();
  }
  cv::imwrite(dstPath, img);
  out = img;
  return true;
}

}  // namespace

/*
  argv: <img1> x y w h <img2> x y w h   (w = 0 keeps the whole image)
  Shows texture + gradient magnitude for each, prints the energies, saves a
  side-by-side to out/texture_compare.png. Blocks until 'q'.
*/
int main(int argc, char* argv[]) {
  if (argc < 11) {
    std::cerr << "Usage: " << argv[0] << " <img1> x y w h <img2> x y w h"
              << std::endl;
    return -1;
  }

  cv::Mat tex[2];
  cv::Mat mag[2];
  const char* dst[2] = {"data/images/texture_a.png",
                        "data/images/texture_b.png"};

  for (int i = 0; i < 2; i++) {
    const int base = 1 + i * 5;
    if (!load(argv[base], std::atoi(argv[base + 1]), std::atoi(argv[base + 2]),
              std::atoi(argv[base + 3]), std::atoi(argv[base + 4]), dst[i],
              tex[i])) {
      return 1;
    }
    cv::Mat sx, sy;
    sobelX3x3(tex[i], sx);
    sobelY3x3(tex[i], sy);
    magnitude(sx, sy, mag[i]);
    cv::imwrite(i == 0 ? "data/images/texture_a_mag.png"
                       : "data/images/texture_b_mag.png",
                mag[i]);
    report(i == 0 ? "A" : "B", tex[i], mag[i], sx, sy);
  }

  cv::Mat top, bottom, all;
  cv::hconcat(tex[0], mag[0], top);
  cv::hconcat(tex[1], mag[1], bottom);
  cv::vconcat(top, bottom, all);
  cv::imwrite("out/texture_compare.png", all);

  cv::namedWindow("texture | gradient magnitude", cv::WINDOW_NORMAL);
  cv::imshow("texture | gradient magnitude", all);
  while (true) {
    char key = (char)cv::waitKey(0);
    if (key == 'q' || key == 'Q') return 0;
  }
}
