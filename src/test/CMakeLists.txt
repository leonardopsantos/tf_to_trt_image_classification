cuda_add_executable(tensorrt tensorrt.cu jetsonTX2Power/jetson_tx2_power.cpp)
target_link_libraries(tensorrt ${OpenCV_LIBS} nvinfer nvparsers)

