cmake_minimum_required(VERSION 3.5)
project(triangle)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLINUX -Wall -Wextra -Werror -g -Wno-unused-parameter")

find_package(X11 REQUIRED)
set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(kissfft REQUIRED)
find_package(RtAudio REQUIRED)

get_property(importTargetsAfter DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
list(REMOVE_ITEM importTargetsAfter ${importTargets})
message("${importTargetsAfter}")

add_executable(triangle main.cpp)

include_directories( ${OpenCV_INCLUDE_DIRS} )
#include_directories(${X11_INCLUDE_DIR})
#include_directories(${OPENGL_INCLUDE_DIRS})
#include_directories(${GLEW_INCLUDE_DIRS})

target_link_libraries(triangle PRIVATE ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} glfw RtAudio::rtaudio "/usr/lib/x86_64-linux-gnu/libkissfft-float.so.131.1.0")