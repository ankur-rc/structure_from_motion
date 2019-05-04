cmake_minimum_required(VERSION 3.10)
project(sfm VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wpedantic -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting default build type to: " ${default_build_type})
    set(CMAKE_BUILD_TYPE ${default_build_type})
endif()


# add the executable
add_executable(${CMAKE_PROJECT_NAME} "${PROJECT_SOURCE_DIR}/src/main.cpp" "${PROJECT_SOURCE_DIR}/src/sfm.cpp")

# include directories
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/include")

# Boost
set(BOOST_ROOT "/usr/local")
# set(BOOST_INCLUDEDIR "${BOOST_ROOT}/include/boost")
# set(BOOST_LIBRARYDIR "${BOOST_ROOT}/lib")
set(Boost_NO_SYSTEM_PATHS TRUE)
set(Boost_USE_MULTITHREADED OFF)
# set(Boost_DEBUG TRUE)
set(Boost_DETAILED_FAILURE_MSG TRUE)

find_package(Boost 1.69.0 REQUIRED COMPONENTS system filesystem)   
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${Boost_LIBRARIES})

# OpenCV
find_package(OpenCV 3.3 REQUIRED core highgui imgproc features2d calib3d imgcodecs)  
message(STATUS "OpenCV_INCLUDE_DIRS:    " ${OpenCV_INCLUDE_DIRS})
message(STATUS "Opencv_LIBS:            " ${OpenCV_LIBS})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULES)
target_link_libraries(${CMAKE_PROJECT_NAME} Eigen3::Eigen)

# Ceres
find_package(Ceres 1.14 REQUIRED)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CERES_INCLUDE_DIRS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${CERES_LIBRARIES})

# PCL
find_package(PCL 1.7 REQUIRED COMPONENTS common io visualization)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${PCL_LIBRARIES})


