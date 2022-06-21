#General
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(torch_keypoints)
#Torch
set(CMAKE_PREFIX_PATH /home/tigalilly/libtorch;/usr/local/include/torchvision/)
#Torch
find_package(TorchVision REQUIRED)
find_package(Python3 COMPONENTS Development)
#OpenCV
find_package(OpenCV REQUIRED)
#General
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
#Torch
add_executable(${PROJECT_NAME} main.cpp)
#Torch
target_compile_features(${PROJECT_NAME} PUBLIC cxx_range_for)
target_link_libraries(${PROJECT_NAME} TorchVision::TorchVision)
#OpenCV
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
#General
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)