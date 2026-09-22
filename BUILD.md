# Build Details

## Prerequisites

- **Windows:** Visual Studio 2022+ (MSVC, C++ workload), CMake 3.15+, OpenCV
  prebuilt for Windows extracted to `C:/opencv` (so `C:/opencv/build` has
  `x64/vc16/...`). Adjust `OpenCV_DIR` in [CMakePresets.json](CMakePresets.json)
  if yours lives elsewhere.
- **Linux/macOS:** g++ with C++17, `libopencv-dev` (or equivalent) providing
  `pkg-config opencv4`.

## Windows: CMake + MSVC

```powershell
# configure once, or whenever CMakeLists.txt itself changes
cmake --preset windows-msvc

# build (repeat this on every code change — no need to reconfigure)
cmake --build build --config Debug --target vid   # or img / main

# run
./out/vid.exe
```

`build/` vs `out/`:

- `build/` is CMake's own workspace — `CMakeCache.txt`, generated `.vcxproj`
  files, and intermediate `.obj`s. Created/refreshed by *configure*
  (`cmake --preset ...` or "Delete cache and reconfigure" in VS Code). This
  step never invokes the compiler — it just regenerates project files from
  `CMakeLists.txt`.
- `out/` is where actual build products land — set by
  `CMAKE_RUNTIME_OUTPUT_DIRECTORY` in [CMakeLists.txt](CMakeLists.txt).
  Populated by *build* (`cmake --build`), which compiles/links and drops
  `vid.exe`, `img.exe`, `main.exe` there.

Reconfigure only when `CMakeLists.txt` changes (new source file, new target,
new `find_package`, etc.) — not on every ordinary edit-compile loop.

### DLL copy step

OpenCV is linked dynamically (`opencv_world4xxd.dll`). Windows has no rpath,
so the loader only checks the exe's own folder + `PATH`. Since `PATH` isn't
guaranteed to include your OpenCV install, `CMakeLists.txt` adds a
`POST_BUILD` command per target that copies the runtime DLL from
`OpenCV_DIR/x64/vc16/bin` into `out/` next to the exe on every build. If
`vid.exe` fails to launch with a missing-DLL error, check that this copy step
ran (rebuild) and that `OpenCV_DIR` points at a valid install.

You may also see benign `INFO` lines at startup like:

```
global plugin_loader.impl.hpp:67 cv::plugin::impl::DynamicLib::libraryLoad load opencv_core_parallel_openmp500_64d.dll => FAILED
```

This is OpenCV failing to load an *optional* parallel-backend plugin and
falling back automatically — not an error, safe to ignore.

## Linux/macOS: make

```sh
make            # builds out/main, out/vid, out/img
make run P=vid  # builds (if needed) and runs out/vid
make clean
```

Uses `pkg-config opencv4` for flags/libs instead of CMake — no `OpenCV_DIR`
needed as long as OpenCV is installed system-wide.

## Troubleshooting

- **"Unable to open video device"** — another app is holding the camera, or
  index `0` isn't your camera; try `cv::VideoCapture(1)` in
  [src/vidDisplay.cpp](src/vidDisplay.cpp).
- **Window opens but effect doesn't seem to apply** — click the video window
  so it has keyboard focus before pressing `g`/`c`/`s`/`q`; `cv::waitKey`
  only sees keystrokes sent to that window.
- **Nothing happens after "Delete cache and reconfigure"** — that step only
  regenerates `build/`, it doesn't compile. Run the `cmake --build` step above.
