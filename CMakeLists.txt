cmake_minimum_required(VERSION 3.0)

project(sparse-stereo-refinement)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


find_package(OpenCV REQUIRED)

add_executable(sparse-stereo-refinement main.cpp)

target_link_libraries(sparse-stereo-refinement PRIVATE ${OpenCV_LIBRARIES})

