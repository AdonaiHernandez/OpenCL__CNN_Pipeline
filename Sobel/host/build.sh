export AOCL_LIB=/root/aocl-rte-18.1.0-625.arm32/host/arm32/lib
export AOCL_INC=/root/aocl-rte-18.1.0-625.arm32/host/include/

g++ main.cpp  -o sobel -Wl,--no-as-needed \
    -I$AOCL_INC \
    -L$AOCL_LIB \
    -lOpenCL \
    -L/root/aocl-rte-18.1.0-625.arm32/host/arm32/lib/ -lalteracl \
    -L/root/aocl-rte-18.1.0-625.arm32/board/c5soc/arm32/lib/ -lintel_soc32_mmd \
	-lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs\
    -lelf \
    -lstdc++

