cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(torch_keypoints)

set(CMAKE_PREFIX_PATH /home/tigalilly/libtorch;/usr/local/include/torchvision/)
find_package(TorchVision REQUIRED)
find_package(Python3 COMPONENTS Development)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} main.cpp)
#add_executable(${PROJECT_NAME} single_image.cpp)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_range_for)
target_link_libraries(${PROJECT_NAME} TorchVision::TorchVision)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)