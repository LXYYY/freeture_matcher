cmake_minimum_required(VERSION 2.8.3)
project(freeture_matcher)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-std=c++11 -Wall -Wextra)

cs_add_library(${PROJECT_NAME} src/freeture_extractor.cpp)

cs_install()
cs_export()
