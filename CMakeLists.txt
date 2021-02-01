cmake_minimum_required(VERSION 2.8.3)
project(freeture_matcher)

add_definitions(-std=c++11 -Werror)

find_package(catkin_simple REQUIRED)
find_package(Open3D REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

include_directories(include ${Open3D_INCLUDE_DIRS})

cs_add_library(${PROJECT_NAME} src/freeture_extractor.cpp)
target_link_libraries(${PROJECT_NAME} ${Open3D_LIBRARIES})

cs_add_executable(${PROJECT_NAME}_node src/freeture_matcher_node.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})

cs_install()
cs_export(LIBRARIES ${Open3D_LIBRARIES})
