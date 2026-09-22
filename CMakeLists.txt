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
# same job here without recompiling filters.cpp per target
add_library(filters STATIC src/filters.cpp)
target_link_libraries(filters PUBLIC ${OpenCV_LIBS})

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE ${OpenCV_LIBS})

add_executable(vid src/vidDisplay.cpp)
target_link_libraries(vid PRIVATE filters)

add_executable(img src/imgDisplay.cpp)
target_link_libraries(img PRIVATE filters)

# Windows has no rpath; the exe's own folder is checked before PATH, so copy
# the runtime DLL there on every build instead of relying on PATH/session state
if(WIN32)
    set(OPENCV_RUNTIME_DLL
        "${OpenCV_DIR}/x64/vc16/bin/opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}$<$<CONFIG:Debug>:d>.dll")
    foreach(tgt main vid img)
        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENCV_RUNTIME_DLL}" "$<TARGET_FILE_DIR:${tgt}>"
            COMMENT "Copying OpenCV runtime DLL next to ${tgt}")
    endforeach()
endif()
