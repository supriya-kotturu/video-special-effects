# Video Special Effects

CS 5330 Project 1: OpenCV filters, face detection and Depth Anything V2 depth
estimation on still images and a live webcam feed.

## Executables

- `vid`: live camera feed with keyboard-driven effects (see below)
- `img`: task 1, load an image file, display it, quit on `q`
- `timeBlur`: task 6, times `blur5x5_1` vs `blur5x5_2`
- `da2example` / `da2video`: course-provided Depth Anything V2 demos
- `main`: scratch/sanity check target

Filters live in [src/filters.cpp](src/filters.cpp) /
[include/filters.h](include/filters.h); face detection (course-provided) in
[src/faceDetect.cpp](src/faceDetect.cpp).

## Key bindings (`vid`)

One effect mode is active at a time. `v` and `f` are toggles that stack on
top of any mode.

| Key | Effect | Task | Description |
| --- | --- | --- | --- |
| `c` | Color | 2 | Unmodified camera feed |
| `g` | Greyscale | 3 | OpenCV `cvtColor` |
| `h` | Custom greyscale | 4 | (255 − G) + mean(B, G, R), wrapped to 8 bits |
| `x` | X-ray | — | Green-channel negative, 255 − G |
| `p` | Sepia | 5 | Antique-camera tone from the original BGR values |
| `b` | Blur | 6 | Separable 5×5 Gaussian |
| `1` | Sobel X | 7 | Vertical edges (absolute value shown) |
| `2` | Sobel Y | 7 | Horizontal edges (absolute value shown) |
| `3` | Gradient magnitude | 8 | Color edge strength, sqrt(sx² + sy²) |
| `l` | Blur + quantize | 9 | Posterized, 10 levels per channel |
| `d` | Depth map | 11 | Depth Anything V2, INFERNO colormap (bright = near) |
| `o` | Depth fog | 11 | Exponential fog that thickens with distance |
| `r` | Seattle rain | 12 | Animated see-through rain that stays off your face |
| `n` | Neon duotone | 12 | Green on detailed regions, magenta/yellow on smooth ones |
| `t` | Cartoonize | 12 | Flat quantized colors with black edge outlines |
| `m` | Median | ext. | 5×5 median: denoises while keeping edges sharp |
| `k` | Disco lights | ext. | Neon strobe on the background, face natural; press again to swap |
| `v` | Vignette (toggle) | ext. | Darkens toward the corners |
| `f` | Face boxes (toggle) | 10 | Haar-cascade face detection |
| `s` | Save frame | — | Writes `data/frames/captured_frame_<n>.jpg` |
| `q` | Quit | — | |

Depth modes (`d`, `o`) load the DA2 model on first use, which takes a moment.

**Photosensitivity warning:** disco mode (`k`) strobes at 2.5 Hz.

## Build

Two build paths are supported:

**Windows (CMake + MSVC, local OpenCV):**

```powershell
cmake --preset windows-msvc
cmake --build build --config Debug --target vid
./out/vid.exe
```

**Linux/macOS (make + pkg-config):**

```sh
make run P=vid
```

Run from the repo root: `vid` loads `data/haarcascade_frontalface_alt2.xml`
and `data/model_fp16.onnx` by relative path. Requires OpenCV 4 (OpenCV 5
dropped `CascadeClassifier`) and ONNX Runtime.

See [BUILD.md](BUILD.md) for OpenCV setup, the CMake vs. `out/` vs. `build/`
distinction, the DLL-copy step on Windows, and troubleshooting. CMake expects
ONNX Runtime at `C:/onnxruntime/...`; override with `-DONNXRUNTIME_ROOT=...`.

## Layout

```text
src/        source files (one .cpp per executable, plus filters.cpp and faceDetect.cpp)
include/    filters.h, faceDetect.h, DA2Network.hpp
data/       Haar cascade, DA2 model, test images; data/frames/ holds saved frames (gitignored)
build/      CMake-generated project files + intermediate objects (gitignored)
out/        final executables + runtime DLLs (gitignored)
```
