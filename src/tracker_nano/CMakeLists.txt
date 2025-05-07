cmake_minimum_required(VERSION 3.8)
project(tracker_nano)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# install python package
ament_python_install_package(${PROJECT_NAME})

set(NODES
  tracker_nano/tracker.py
)


install(PROGRAMS
    ${NODES}
DESTINATION lib/${PROJECT_NAME}
)


ament_package()
