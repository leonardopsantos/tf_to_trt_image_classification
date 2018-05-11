cuda_add_executable(test_trt test_trt.cu jetsonTX2Power/jetson_tx2_power.cpp)
target_link_libraries(test_trt ${OpenCV_LIBS} nvinfer nvparsers)
