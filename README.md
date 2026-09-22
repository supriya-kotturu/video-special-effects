# Video Special Effects

OpenCV project for still-image and live-video display with pixel-level effects
(greyscale, and more added per task). Built around three executables:

- `main` — scratch/sanity check target
- `img` — Task 1: load an image file, display it, quit on `q`
- `vid` — Task 2+: live camera feed with keyboard-driven effect modes
  (`g` greyscale, `c` color, `s` save frame, `q` quit)

Effect implementations live in [src/filters.cpp](src/filters.cpp) /
[include/filters.h](include/filters.h), shared by `img` and `vid`.

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

See [BUILD.md](BUILD.md) for OpenCV setup, the CMake vs. `out/` vs. `build/`
distinction, the DLL-copy step on Windows, and troubleshooting.

## Layout

```
src/        source files (one .cpp per executable, plus filters.cpp)
include/    filters.h
data/       saved frames / captured images (gitignored)
build/      CMake-generated project files + intermediate objects (gitignored)
out/        final executables + runtime DLLs (gitignored)
```
