cmake_minimum_required(VERSION 3.20)
project(fftrip)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

IF (!WIN32)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLINUX -Wall -Wextra -Werror -g -Wno-unused-parameter")
ENDIF()

find_package(X11 REQUIRED)
set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)
find_package(OpenCL REQUIRED)

find_package(OpenCV REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Threads REQUIRED)
IF (WIN32)
find_package(unofficial-kissfft REQUIRED)
find_package(freetype REQUIRED)
ELSE()
find_package(kissfft REQUIRED)
find_package(Freetype REQUIRED)
ENDIF()
find_package(RtAudio REQUIRED)


add_compile_definitions(_USE_MATH_DEFINES)
add_executable(fftrip main.cpp audioanalyzer.cpp)

include_directories( ${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS} ${FREETYPE_INCLUDE_DIRS} )


IF (WIN32)
target_link_libraries(fftrip ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} ${OpenCL_LIBS} ${FREETYPE_LIBRARIES} glfw RtAudio::rtaudio unofficial::kissfft::kissfft)
ELSE()
target_link_libraries(fftrip ${X11_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${OpenCV_LIBS} ${OpenCL_LIBS} ${FREETYPE_LIBRARIES} glfw RtAudio::rtaudio "/usr/lib/x86_64-linux-gnu/libkissfft-float.so.131.1.0")
ENDIF()
