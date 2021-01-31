cmake_minimum_required(VERSION 2.8.3)
project(freeture_matcher)

add_definitions(-std=c++14 -Werror)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(Open3D REQUIRED)

include_directories(${Open3D_INCLUDE_DIRS})

cs_add_library(${PROJECT_NAME}
        src/freeture_extractor.cpp
    )

target_link_libraries(${PROJECT_NAME} ${Open3D_LIBRARIES})

cs_export()
