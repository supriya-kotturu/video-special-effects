cmake_minimum_required(VERSION 3.15)
project(video_special_effects CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# put binaries in out/ to mirror the makefile's BINDIR, one dir per config
# on multi-config generators (VS)
foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${cfg} cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} ${CMAKE_SOURCE_DIR}/out)
endforeach()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

find_package(OpenCV REQUIRED)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall)
endif()

include_directories(${OpenCV_INCLUDE_DIRS} include)

# filters.o is shared between vid/img in the makefile; a static lib does the
# same job here without recompiling filters.cpp per target. faceDetect.cpp
# (course-provided) rides along so every program can use face detection.
add_library(filters STATIC src/filters.cpp src/faceDetect.cpp)
target_link_libraries(filters PUBLIC ${OpenCV_LIBS})

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE ${OpenCV_LIBS})

add_executable(vid src/vidDisplay.cpp)
target_link_libraries(vid PRIVATE filters)

add_executable(img src/imgDisplay.cpp)
target_link_libraries(img PRIVATE filters)

add_executable(timeBlur src/timeBlur.cpp)
target_link_libraries(timeBlur PRIVATE filters)

# ONNX Runtime (task 11, Depth Anything V2): Microsoft's prebuilt release zip,
# extracted outside the repo. Override with -DONNXRUNTIME_ROOT=... if moved.
set(ONNXRUNTIME_ROOT "C:/onnxruntime/onnxruntime-win-x64-1.30.0"
    CACHE PATH "Extracted ONNX Runtime release")
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")

add_executable(da2example src/da2-example.cpp)
target_link_libraries(da2example PRIVATE ${OpenCV_LIBS} onnxruntime)

add_executable(da2video src/da2-video.cpp)
target_link_libraries(da2video PRIVATE ${OpenCV_LIBS} onnxruntime)

# vid uses DA2 for the depth and fog modes
target_link_libraries(vid PRIVATE onnxruntime)

# Windows ships an older onnxruntime.dll in System32; the copy next to the exe
# is found first, so the version we compiled against is the one that loads
foreach(tgt da2example da2video vid)
    add_custom_command(TARGET ${tgt} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
            "$<TARGET_FILE_DIR:${tgt}>"
        COMMENT "Copying ONNX Runtime DLLs next to ${tgt}")
endforeach()

# Windows has no rpath; the exe's own folder is checked before PATH, so copy
# the runtime DLL there on every build instead of relying on PATH/session state.
# Only needed for a prebuilt opencv_world install: the vcpkg toolchain already
# copies each exe's DLL dependencies next to it (VCPKG_APPLOCAL_DEPS).
# OpenCV 4 is required: OpenCV 5 dropped CascadeClassifier (task 10).
if(WIN32 AND NOT VCPKG_TOOLCHAIN)
    set(OPENCV_RUNTIME_DLL
        "${OpenCV_DIR}/x64/vc16/bin/opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}$<$<CONFIG:Debug>:d>.dll")
    foreach(tgt main vid img timeBlur)
        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENCV_RUNTIME_DLL}" "$<TARGET_FILE_DIR:${tgt}>"
            COMMENT "Copying OpenCV runtime DLL next to ${tgt}")
    endforeach()
endif()

# texture comparison by gradient-magnitude energy
add_executable(textureEnergy src/textureEnergy.cpp)
target_link_libraries(textureEnergy PRIVATE filters)
