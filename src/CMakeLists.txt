cmake_minimum_required (VERSION 3.8)

project ("virtual-referee")

set(OpenCV_DIR C:/opencv/build )
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable (virtual-referee "virtual-referee.cpp" "virtual-referee.h" )
target_link_libraries( virtual-referee ${OpenCV_LIBS} )

set_property(TARGET virtual-referee PROPERTY CXX_STANDARD 20)