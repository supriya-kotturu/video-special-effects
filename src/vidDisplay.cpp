/*
  Author:  Sai Supriya Kotturu
  Date:    2026-09-21
  Purpose: Tasks 2+ - capture live video, display it, and apply the effects from
          filters.h according to the key the user last pressed.
*/

#include <filesystem>
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "DA2Network.hpp"
#include "faceDetect.h"
#include "filters.h"

// which filter the main loop applies; set by the last mode key pressed
enum class Mode {
  RGB,
  GRAY,
  CUSTOM_GRAY,
  SEPIA,
  X_RAY,
  BLUR,
  SOBEL_X,
  SOBEL_Y,
  MAGNITUDE,
  BLUR_QUANTIZE,
  DEPTH,
  RAIN,
  FOG,
  NEON,
  CARTOON,
  MEDIAN,
  DISCO
};

/*
  Saves a frame as data/frames/captured_frame_<id>.jpg, creating the folder
  if needed.

  frame:       image to write (what is currently displayed)
  currFrameId: number used in the file name so saves don't overwrite each other
  returns:     1 if the file was written, 0 on failure
*/
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

/*
  Opens the default camera and shows the live stream. Keys:
    c = color, g = OpenCV grayscale, h = custom grayscale, x = X-ray,
    p = sepia, b = blur, 1 = Sobel X, 2 = Sobel Y, 3 = gradient magnitude,
    l = blur + quantize (10 levels), d = depth map (DA2),
    o = depth fog, r = Seattle rain (clear over faces),
    n = neon duotone, t = cartoonize (flat colors + black outlines),
    m = median filter (5x5), k = disco lights (strobes at 2.5 Hz; face stays
    natural, press k again to swap face/background),
    v = toggle vignette, f = toggle face boxes (both stack on any mode),
    s = save current frame, q = quit
  argc/argv are unused. Returns 0 on normal exit, -1 if the camera can't open.
*/
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
  // raw signed (CV_16SC3) Sobel output, kept separate from the displayable
  // image so later steps (gradient magnitude) can use the signed values
  cv::Mat sobel;
  cv::Mat sobelY;  // second gradient, only needed for magnitude
  int frameCounter = 1;
  Mode mode = Mode::RGB;
  bool vignette = false;
  bool showFaces = false;
  bool discoOnFace = false;
  cv::Mat grey;                 // detector input
  std::vector<cv::Rect> faces;  // detector output, reused each frame

  // Depth Anything V2, created on first use so the other modes don't pay the
  // model load time. The scale must stay constant: run_network keeps a static
  // buffer sized from its first call.
  std::unique_ptr<DA2Network> daNet;
  const float DEPTH_SCALE = 0.5f;  // 640x480 -> 320x240 network input
  // DA2 is slow on the CPU, so refresh depth every few frames and reuse it;
  // people move little between refreshes
  const int DEPTH_EVERY = 3;
  cv::Mat depth;  // CV_8UC1, 255 = nearest
  cv::Mat depthVis;
  int depthTick = 0;

  while (true) {
    *capdev >> frame;  // get a new frame from the camera, treat as a stream

    if (frame.empty()) {
      printf("frame is empty\n");
      break;
    }

    // see if there is a waiting keystroke
    char key = cv::waitKey(10);

    // mode keys change the active filter; the mode persists across frames
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
      case '1':
        std::cout << "Changing mode to SOBEL_X" << std::endl;
        mode = Mode::SOBEL_X;
        break;
      case '2':
        std::cout << "Changing mode to SOBEL_Y" << std::endl;
        mode = Mode::SOBEL_Y;
        break;
      case '3':
        std::cout << "Changing mode to MAGNITUDE" << std::endl;
        mode = Mode::MAGNITUDE;
        break;
      case 'l':
        std::cout << "Changing mode to BLUR_QUANTIZE" << std::endl;
        mode = Mode::BLUR_QUANTIZE;
        break;
      case 'd':
        std::cout << "Changing mode to DEPTH" << std::endl;
        mode = Mode::DEPTH;
        break;
      case 'r':
        std::cout << "Changing mode to RAIN" << std::endl;
        mode = Mode::RAIN;
        break;
      case 'o':
        std::cout << "Changing mode to FOG" << std::endl;
        mode = Mode::FOG;
        break;
      case 'n':
        std::cout << "Changing mode to NEON" << std::endl;
        mode = Mode::NEON;
        break;
      case 't':
        std::cout << "Changing mode to CARTOON" << std::endl;
        mode = Mode::CARTOON;
        break;
      case 'm':
        std::cout << "Changing mode to MEDIAN" << std::endl;
        mode = Mode::MEDIAN;
        break;
      // first press enters disco; pressing again swaps which area flashes
      case 'k':
        if (mode == Mode::DISCO) {
          discoOnFace = !discoOnFace;
          std::cout << "Disco on " << (discoOnFace ? "face" : "background")
                    << std::endl;
        } else {
          std::cout << "Changing mode to DISCO" << std::endl;
          mode = Mode::DISCO;
        }
        break;
      case 'v':
        std::cout << "Toggling vignette" << std::endl;
        vignette = !vignette;
        break;
      case 'f':
        std::cout << "Toggling face detection" << std::endl;
        showFaces = !showFaces;
        break;
      case 'q':
        return (0);
    }

    // depth modes: load the network once, refresh depth every DEPTH_EVERY
    // frames (and immediately if there is no depth yet)
    if (mode == Mode::DEPTH || mode == Mode::FOG) {
      if (!daNet) {
        std::cout << "Loading Depth Anything V2..." << std::endl;
        daNet = std::make_unique<DA2Network>("data/model_fp16.onnx");
      }
      if (depth.empty() || depthTick % DEPTH_EVERY == 0) {
        daNet->set_input(frame, DEPTH_SCALE);
        daNet->run_network(depth, frame.size());
      }
      depthTick++;
    }

    // apply the active filter to this frame
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
      // imshow can't show signed 16-bit correctly; convertScaleAbs takes |v|
      // and converts to CV_8UC3 so both edge directions show as bright
      case Mode::SOBEL_X:
        sobelX3x3(frame, sobel);
        cv::convertScaleAbs(sobel, displayFrame);
        break;
      case Mode::SOBEL_Y:
        sobelY3x3(frame, sobel);
        cv::convertScaleAbs(sobel, displayFrame);
        break;
      // magnitude already produces CV_8UC3, so no convertScaleAbs
      case Mode::MAGNITUDE:
        sobelX3x3(frame, sobel);
        sobelY3x3(frame, sobelY);
        magnitude(sobel, sobelY, displayFrame);
        break;
      case Mode::BLUR_QUANTIZE:
        blurQuantize(frame, displayFrame, 10);
        break;
      // colormap makes relative depth readable: bright/yellow = near
      case Mode::DEPTH:
        cv::applyColorMap(depth, displayFrame, cv::COLORMAP_INFERNO);
        break;
      // rain fades out over detected faces
      case Mode::RAIN:
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
        detectFaces(grey, faces);
        rainEffect(frame, displayFrame, faces);
        break;
      case Mode::FOG:
        fogEffect(frame, depth, displayFrame);
        break;
      case Mode::NEON:
        neonCartoon(frame, displayFrame);
        break;
      case Mode::CARTOON:
        cartoonize(frame, displayFrame);
        break;
      case Mode::MEDIAN:
        medianFilter(frame, displayFrame);
        break;
      // wall-clock time, so the animation speed doesn't depend on the fps
      case Mode::DISCO:
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
        detectFaces(grey, faces);
        discoEffect(frame, displayFrame,
                    cv::getTickCount() / cv::getTickFrequency(), faces,
                    discoOnFace);
        break;
      default:
        displayFrame = frame;
        break;
    }

    // detect on the unfiltered frame (the Haar detector expects a natural
    // greyscale image, not e.g. Sobel output), draw on whatever is displayed
    if (showFaces) {
      cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
      detectFaces(grey, faces);
      drawBoxes(displayFrame, faces);
    }

    // vignette is applied last so it layers over any mode, and before saving
    // so the saved image matches what's on screen
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
