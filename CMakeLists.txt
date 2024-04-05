

SET(SDK_ROOT "/home/lzr/projects/rv/luckfox-pico")

# put this before the first project(...)
SET(CMAKE_C_COMPILER "${SDK_ROOT}/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc")
SET(CMAKE_CXX_COMPILER "${SDK_ROOT}/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-g++")

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

project(opencv-mobile-test)
cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)

add_definitions(-DRV1106_1103)

# this will set vars for 3rd party directories
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/from_zoo/3rdparty/ 3rdparty.out)

# this will compile lib fileutils imagedrawing imageutils
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/from_zoo/utils/ utils.out)

# using opencv-mobile
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv-mobile-4.8.1-luckfox-pico/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# ===== THE MAIN TESTING PROGRAM =====
set(MAIN_BIN_NAME "opencv-mobile-demo")
# everything in src/
file(GLOB SOURCES "src/*.cpp")
add_executable(${MAIN_BIN_NAME} ${SOURCES})

# include & link opencv
target_include_directories(${MAIN_BIN_NAME} PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${MAIN_BIN_NAME} ${OpenCV_LIBS} ${LIBRKNNRT})

# include & link librknn
target_include_directories(${MAIN_BIN_NAME} PRIVATE
    ${LIBRKNNRT_INCLUDES}
)
target_link_libraries(${MAIN_BIN_NAME} ${LIBRKNNRT})

install(TARGETS ${MAIN_BIN_NAME} DESTINATION .)





# file(GLOB SRCS from_zoo/*.cc)
set(FROM_ZOO_BIN "from_zoo")

add_executable(${FROM_ZOO_BIN}
    from_zoo/main.cc
    from_zoo/postprocess.cc
    from_zoo/yolov5_rv1106_1103.cc
)

target_link_libraries(${FROM_ZOO_BIN}
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNRT}
)

# if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${FROM_ZOO_BIN} Threads::Threads)
# endif()

# main.cc wants to include dma_alloc.cpp
target_include_directories(${FROM_ZOO_BIN} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/from_zoo/3rdparty/allocator/dma
)

target_include_directories(${FROM_ZOO_BIN} PRIVATE
    from_zoo
    ${LIBRKNNRT_INCLUDES}
)

install(TARGETS ${FROM_ZOO_BIN} DESTINATION .)


install(FILES yolov5.rknn DESTINATION . )


# file(GLOB YOLO_DEMO_SRC "from_zoo/*.cc")
# add_executable(yolov5-test ${YOLO_DEMO_SRC})
# # target_link_libraries(yolov5-test imageutils)
# include_directories(${LIBRKNNRT_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/from_zoo/utils/ ${CMAKE_CURRENT_SOURCE_DIR}/from_zoo/3rdparty/allocator/dma)


# set(THREADS_PREFER_PTHREAD_FLAG ON)
# find_package(Threads REQUIRED)
# target_link_libraries(opencv-mobile-test Threads::Threads)