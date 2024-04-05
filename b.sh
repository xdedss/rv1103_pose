mkdir build
cd build
cmake .. \
    -DTARGET_SOC=rv1103 \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=armv7l \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/home/lzr/projects/rv/opencv-mobile-test/install
make
make install