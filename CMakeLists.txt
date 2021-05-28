cmake_minimum_required(VERSION 2.8.3)
project(freeture_matcher)

add_definitions(-std=c++11 -Werror)

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(catkin_simple REQUIRED)
find_package(Open3D REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

include_directories(${Open3D_INCLUDE_DIRS})

cs_add_library(${PROJECT_NAME} src/freeture_extractor.cpp)
target_link_libraries(${PROJECT_NAME} ${Open3D_LIBRARIES})

cs_install()
cs_export()
