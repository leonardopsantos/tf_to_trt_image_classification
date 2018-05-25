/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * Full license terms provided in LICENSE.md file.
 */

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <fstream>

#include <ctime>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>

#include <sys/time.h>
#include <unistd.h>

#include "jetsonTX2Power/jetson_tx2_power.hh"

#define MS_PER_SEC 1000.0

using namespace std;
using namespace nvinfer1;

class TestConfig;

typedef void (*preprocess_fn_t)(float *input, size_t channels, size_t height,
        size_t width);
float * imageToTensor(const cv::Mat & image);
void preprocessVgg(float *input, size_t channels, size_t height, size_t width);
void preprocessInception(float *input, size_t channels, size_t height,
        size_t width);
size_t argmax(float *input, size_t numel);
void test(const TestConfig &testConfig, cv::Mat & image);
void test(const TestConfig &testConfig, cv::Mat & image, ICudaEngine *engine,
        IExecutionContext *context);
void test(const TestConfig &testConfig, const std::string image_filename);
void test(const TestConfig &testConfig, cv::Mat & image, ICudaEngine *engine,
        IExecutionContext *context, std::vector<PowerReadingDevice> &devices,
        std::string csv_filename);

class TestConfig {
public:
    string binPath;
    string imagePath;
    string planPath;
    string inputNodeName;
    string outputNodeName;
    string preprocessFnName;
    string inputHeight;
    string inputWidth;
    string numOutputCategories;
    string dataType;
    string maxBatchSize;
    string workspaceSize;
    string numRuns;
    string useMappedMemory;
    string statsPath;
    string netName;

    TestConfig(int argc, char * argv[])
    {
        binPath = argv[0];
        imagePath = argv[1];
        planPath = argv[2];
        inputNodeName = argv[3];
        inputHeight = argv[4];
        inputWidth = argv[5];
        outputNodeName = argv[6];
        numOutputCategories = argv[7];
        preprocessFnName = argv[8];
        numRuns = argv[9];
        dataType = argv[10];
        maxBatchSize = argv[11];
        workspaceSize = argv[12];
        useMappedMemory = argv[13];
        statsPath = argv[14];
        netName = argv[15];
    }

    static string UsageString()
    {
        string s = "";
        s += "binPath: \n";
        s += "imagePath: \n";
        s += "planPath: \n";
        s += "inputNodeName: \n";
        s += "inputHeight: \n";
        s += "inputWidth: \n";
        s += "outputNodeName: \n";
        s += "numOutputCategories: \n";
        s += "preprocessFnName: \n";
        s += "numRuns: \n";
        s += "dataType: \n";
        s += "maxBatchSize: \n";
        s += "workspaceSize: \n";
        s += "useMappedMemory: \n";
        s += "statsPath: \n";
        s += "netName: \n";
        return s;

    }

    string ToString()
    {
        string s = "";
        s += "binPath: " + binPath + "\n";
        s += "imagePath: " + imagePath + "\n";
        s += "planPath: " + planPath + "\n";
        s += "inputNodeName: " + inputNodeName + "\n";
        s += "inputHeight: " + inputHeight + "\n";
        s += "inputWidth: " + inputWidth + "\n";
        s += "outputNodeName: " + outputNodeName + "\n";
        s += "numOutputCategories: " + numOutputCategories + "\n";
        s += "preprocessFnName: " + preprocessFnName + "\n";
        s += "numRuns: " + numRuns + "\n";
        s += "dataType: " + dataType + "\n";
        s += "maxBatchSize: " + maxBatchSize + "\n";
        s += "workspaceSize: " + workspaceSize + "\n";
        s += "useMappedMemory: " + useMappedMemory + "\n";
        s += "statsPath: " + statsPath + "\n";
        s += "netName: " + netName + "\n";
        return s;
    }

    static int ToInteger(string value)
    {
        int valueInt;
        stringstream ss;
        ss << value;
        ss >> valueInt;
        return valueInt;
    }

    preprocess_fn_t PreprocessFn() const
    {
        if (preprocessFnName == "preprocess_vgg")
            return preprocessVgg;
        else if (preprocessFnName == "preprocess_inception")
            return preprocessInception;
        else
            throw runtime_error("Invalid preprocessing function name.");
    }

    int InputWidth() const
    {
        return ToInteger(inputWidth);
    }
    int InputHeight() const
    {
        return ToInteger(inputHeight);
    }
    int NumOutputCategories() const
    {
        return ToInteger(numOutputCategories);
    }

    nvinfer1::DataType DataType() const
    {
        if (dataType == "float")
            return nvinfer1::DataType::kFLOAT;
        else if (dataType == "half")
            return nvinfer1::DataType::kHALF;
        else
            throw runtime_error("Invalid data type.");
    }

    int MaxBatchSize() const
    {
        return ToInteger(maxBatchSize);
    }
    int WorkspaceSize() const
    {
        return ToInteger(workspaceSize);
    }
    int NumRuns() const
    {
        return ToInteger(numRuns);
    }
    int UseMappedMemory() const
    {
        return ToInteger(useMappedMemory);
    }
    std::string GetNetName() const
    {
        return netName;
    }
};

class Logger: public ILogger {
    void log(Severity severity, const char * msg) override
    {
//      cout << msg << endl;
    }
} gLogger;

// It is heretic to use just 2 spaces for indentation. The Linux kernel coding
// guidelines says we ought to use 8, so we'll use 8!
std::string get_csv_name(std::string net_name)
{
    char buf[1024];
    time_t t;
    struct tm *tmp;

    time(&t);
    tmp = localtime(&t);

    memset(buf, '\0', sizeof(buf));
    // C++ doesn't have a nice function for this!
    strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M-%S", tmp);

    std::string s = buf;
    s += "-" + net_name + ".csv";

    return s;
}

std::string get_path_prefix(std::string path)
{
    size_t found;
    found = path.find_last_of("/\\");
    return path.substr(0, found);
}

int main(int argc, char * argv[])
{
    if (argc != 16) {
        cout << TestConfig::UsageString() << endl;
        return 0;
    }

    printf("TensorRT version %u.%u.%u, build %u\n", NV_TENSORRT_MAJOR,
            NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_GIE_VERSION);

    TestConfig testConfig(argc, argv);
    cout << "\ntestConfig: \n" << testConfig.ToString() << endl;

    string image_file;
    ifstream image_list(testConfig.imagePath);
    std::string path_prefix = get_path_prefix(testConfig.imagePath);
    int num_runs = 0;
    int num_images = 0;

    std::vector<PowerReadingDevice> pwr_devices = create_devices();
    std::string csv_filename = get_csv_name(testConfig.GetNetName());

    ifstream planFile(testConfig.planPath);
    stringstream planBuffer;
    planBuffer << planFile.rdbuf();
    string plan = planBuffer.str();
    IRuntime *runtime = createInferRuntime(gLogger);
    ICudaEngine *engine = runtime->deserializeCudaEngine((void*) plan.data(),
            plan.size(), nullptr);
    IExecutionContext *context = engine->createExecutionContext();

    // a comment
    // run and compute average time over numRuns iterations
    while (getline(image_list, image_file)) {
        std::string image_filename = path_prefix + "/" + image_file;

        if (access(image_filename.c_str(), F_OK) < 0) {
            cout << "Could not open " << image_filename << endl;
            continue;
        }

        // load and preprocess image
        cv::Mat image = cv::imread(image_filename, CV_LOAD_IMAGE_COLOR);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB, 3);
        cv::resize(image, image,
                cv::Size(testConfig.InputWidth(), testConfig.InputHeight()));

//	test(testConfig, image, engine, context);

        test(testConfig, image, engine, context, pwr_devices, csv_filename);

//	test(testConfig, image);
//	test(testConfig, image_filename);
    }

    to_csv(csv_filename, pwr_devices, testConfig.GetNetName() + ":done");

    engine->destroy();
    context->destroy();
    runtime->destroy();

    return 0;
}

float *imageToTensor(const cv::Mat & image)
{
    const size_t height = image.rows;
    const size_t width = image.cols;
    const size_t channels = image.channels();
    const size_t numel = height * width * channels;

    const size_t stridesCv[3] = { width * channels, channels, 1 };
    const size_t strides[3] = { height * width, width, 1 };

    float * tensor;
    cudaHostAlloc((void**) &tensor, numel * sizeof(float), cudaHostAllocMapped);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < channels; k++) {
                const size_t offsetCv = i * stridesCv[0] + j * stridesCv[1]
                        + k * stridesCv[2];
                const size_t offset = k * strides[0] + i * strides[1]
                        + j * strides[2];
                tensor[offset] = (float) image.data[offsetCv];
            }
        }
    }

    return tensor;
}

void preprocessVgg(float * tensor, size_t channels, size_t height, size_t width)
{
    const size_t strides[3] = { height * width, width, 1 };
    const float mean[3] = { 123.68, 116.78, 103.94 };

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < channels; k++) {
                const size_t offset = k * strides[0] + i * strides[1]
                        + j * strides[2];
                tensor[offset] -= mean[k];
            }
        }
    }
}

void preprocessInception(float * tensor, size_t channels, size_t height,
        size_t width)
{
    const size_t numel = channels * height * width;
    for (int i = 0; i < numel; i++)
        tensor[i] = 2.0 * (tensor[i] / 255.0 - 0.5);
}

size_t argmax(float * tensor, size_t numel)
{
    if (numel <= 0)
        return 0;

    size_t maxIndex = 0;
    float max = tensor[0];
    for (int i = 0; i < numel; i++) {
        if (tensor[i] > max) {
            maxIndex = i;
            max = tensor[i];
        }
    }

    return maxIndex;
}

void test(const TestConfig &testConfig, const std::string image_filename)
{
    printf("TensorRT version %u.%u.%u, build %u\n", NV_TENSORRT_MAJOR,
            NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_GIE_VERSION);

    ifstream planFile(testConfig.planPath);
    stringstream planBuffer;
    planBuffer << planFile.rdbuf();
    string plan = planBuffer.str();
    IRuntime *runtime = createInferRuntime(gLogger);
    ICudaEngine *engine = runtime->deserializeCudaEngine((void*) plan.data(),
            plan.size(), nullptr);
    IExecutionContext *context = engine->createExecutionContext();

    int inputBindingIndex, outputBindingIndex;
    inputBindingIndex = engine->getBindingIndex(
            testConfig.inputNodeName.c_str());
    outputBindingIndex = engine->getBindingIndex(
            testConfig.outputNodeName.c_str());

    // load and preprocess image
    cv::Mat image = cv::imread(image_filename, CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB, 3);
    cv::resize(image, image,
            cv::Size(testConfig.InputWidth(), testConfig.InputHeight()));
    float *input = imageToTensor(image);
    testConfig.PreprocessFn()(input, 3, testConfig.InputHeight(),
            testConfig.InputWidth());

    // allocate memory on host / device for input / output
    float *output;
    float *inputDevice;
    float *outputDevice;
    size_t inputSize = testConfig.InputHeight() * testConfig.InputWidth() * 3
            * sizeof(float);

    cudaHostAlloc(&output, testConfig.NumOutputCategories() * sizeof(float),
            cudaHostAllocMapped);

    if (testConfig.UseMappedMemory()) {
        cudaHostGetDevicePointer(&inputDevice, input, 0);
        cudaHostGetDevicePointer(&outputDevice, output, 0);
    } else {
        cudaMalloc(&inputDevice, inputSize);
        cudaMalloc(&outputDevice,
                testConfig.NumOutputCategories() * sizeof(float));
    }

    float *bindings[2];
    bindings[inputBindingIndex] = inputDevice;
    bindings[outputBindingIndex] = outputDevice;

    std::vector<PowerReadingDevice> devices = create_devices();

    std::string csv_filename = get_csv_name(testConfig.GetNetName());

    to_csv(csv_filename, devices, testConfig.GetNetName() + ":running empty");
    to_csv(csv_filename, devices, testConfig.GetNetName() + ":begin");

    // run and compute average time over numRuns iterations
    double avgTime = 0;
    for (int i = 0; i < testConfig.NumRuns() + 1; i++) {
        chrono::duration<double> diff;

        if (testConfig.UseMappedMemory()) {
            auto t0 = chrono::steady_clock::now();
            context->execute(1, (void**) bindings);
            auto t1 = chrono::steady_clock::now();
            diff = t1 - t0;
        } else {
            auto t0 = chrono::steady_clock::now();
            cudaMemcpy(inputDevice, input, inputSize, cudaMemcpyHostToDevice);
            context->execute(1, (void**) bindings);
            cudaMemcpy(output, outputDevice,
                    testConfig.NumOutputCategories() * sizeof(float),
                    cudaMemcpyDeviceToHost);
            auto t1 = chrono::steady_clock::now();
            diff = t1 - t0;
        }

        if (i != 0)
            avgTime += MS_PER_SEC * diff.count();

        // This is ridiculous...
        ostringstream conv;
        conv << i;
        to_csv(csv_filename, devices,
                testConfig.GetNetName() + ":run #" + conv.str());
    }
    avgTime /= testConfig.NumRuns();

    // save results to file
    int maxCategoryIndex = argmax(output, testConfig.NumOutputCategories())
            + 1001 - testConfig.NumOutputCategories();
    cout << "Most likely category id is " << maxCategoryIndex << endl;
    cout << "Average execution time in ms is " << avgTime << endl;
    ofstream outfile;
    outfile.open(testConfig.statsPath, ios_base::app);
    outfile << "\n" << testConfig.planPath << " " << avgTime;
    // << " " << maxCategoryIndex
    // << " " << testConfig.InputWidth() 
    // << " " << testConfig.InputHeight()
    // << " " << testConfig.MaxBatchSize() 
    // << " " << testConfig.WorkspaceSize() 
    // << " " << testConfig.dataType 
    // << " " << testConfig.NumRuns() 
    // << " " << testConfig.UseMappedMemory();
    outfile.close();

    cudaFree(inputDevice);
    cudaFree(outputDevice);

    cudaFreeHost(input);
    cudaFreeHost(output);

    engine->destroy();
    context->destroy();
    runtime->destroy();

    to_csv(csv_filename, devices, testConfig.GetNetName() + ":done");
}

void test(const TestConfig &testConfig, cv::Mat & image, ICudaEngine *engine,
        IExecutionContext *context)
{
    int inputBindingIndex, outputBindingIndex;
    inputBindingIndex = engine->getBindingIndex(
            testConfig.inputNodeName.c_str());
    outputBindingIndex = engine->getBindingIndex(
            testConfig.outputNodeName.c_str());

    float *input = imageToTensor(image);
    testConfig.PreprocessFn()(input, 3, testConfig.InputHeight(),
            testConfig.InputWidth());

    // allocate memory on host / device for input / output
    float *output;
    float *inputDevice;
    float *outputDevice;
    size_t inputSize = testConfig.InputHeight() * testConfig.InputWidth() * 3
            * sizeof(float);

    cudaHostAlloc(&output, testConfig.NumOutputCategories() * sizeof(float),
            cudaHostAllocMapped);

    if (testConfig.UseMappedMemory()) {
        cudaHostGetDevicePointer(&inputDevice, input, 0);
        cudaHostGetDevicePointer(&outputDevice, output, 0);
    } else {
        cudaMalloc(&inputDevice, inputSize);
        cudaMalloc(&outputDevice,
                testConfig.NumOutputCategories() * sizeof(float));
    }

    float *bindings[2];
    bindings[inputBindingIndex] = inputDevice;
    bindings[outputBindingIndex] = outputDevice;

    std::vector<PowerReadingDevice> devices = create_devices();

    std::string csv_filename = get_csv_name(testConfig.GetNetName());

    to_csv(csv_filename, devices, testConfig.GetNetName() + ":running empty");
    to_csv(csv_filename, devices, testConfig.GetNetName() + ":begin");

    // run and compute average time over numRuns iterations
    double avgTime = 0;
    for (int i = 0; i < testConfig.NumRuns() + 1; i++) {
        chrono::duration<double> diff;

        if (testConfig.UseMappedMemory()) {
            auto t0 = chrono::steady_clock::now();
            context->execute(1, (void**) bindings);
            auto t1 = chrono::steady_clock::now();
            diff = t1 - t0;
        } else {
            auto t0 = chrono::steady_clock::now();
            cudaMemcpy(inputDevice, input, inputSize, cudaMemcpyHostToDevice);
            context->execute(1, (void**) bindings);
            cudaMemcpy(output, outputDevice,
                    testConfig.NumOutputCategories() * sizeof(float),
                    cudaMemcpyDeviceToHost);
            auto t1 = chrono::steady_clock::now();
            diff = t1 - t0;
        }

        if (i != 0)
            avgTime += MS_PER_SEC * diff.count();

        // This is ridiculous...
        ostringstream conv;
        conv << i;
        to_csv(csv_filename, devices,
                testConfig.GetNetName() + ":run #" + conv.str());
    }
    avgTime /= testConfig.NumRuns();

    // save results to file
    int maxCategoryIndex = argmax(output, testConfig.NumOutputCategories())
            + 1001 - testConfig.NumOutputCategories();
    cout << "Most likely category id is " << maxCategoryIndex << endl;
    cout << "Average execution time in ms is " << avgTime << endl;
    ofstream outfile;
    outfile.open(testConfig.statsPath, ios_base::app);
    outfile << "\n" << testConfig.planPath << " " << avgTime;
    // << " " << maxCategoryIndex
    // << " " << testConfig.InputWidth() 
    // << " " << testConfig.InputHeight()
    // << " " << testConfig.MaxBatchSize() 
    // << " " << testConfig.WorkspaceSize() 
    // << " " << testConfig.dataType 
    // << " " << testConfig.NumRuns() 
    // << " " << testConfig.UseMappedMemory();
    outfile.close();

    cudaFree(inputDevice);
    cudaFree(outputDevice);

    cudaFreeHost(input);
    cudaFreeHost(output);

    to_csv(csv_filename, devices, testConfig.GetNetName() + ":done");
}

void test(const TestConfig &testConfig, cv::Mat & image, ICudaEngine *engine,
        IExecutionContext *context,
        std::vector<PowerReadingDevice> &pwr_devices, std::string csv_filename)
{
    int inputBindingIndex, outputBindingIndex;
    inputBindingIndex = engine->getBindingIndex(
            testConfig.inputNodeName.c_str());
    outputBindingIndex = engine->getBindingIndex(
            testConfig.outputNodeName.c_str());

    float *input = imageToTensor(image);
    testConfig.PreprocessFn()(input, 3, testConfig.InputHeight(),
            testConfig.InputWidth());

    // allocate memory on host / device for input / output
    float *output;
    float *inputDevice;
    float *outputDevice;
    size_t inputSize = testConfig.InputHeight() * testConfig.InputWidth() * 3
            * sizeof(float);

    cudaHostAlloc(&output, testConfig.NumOutputCategories() * sizeof(float),
            cudaHostAllocMapped);

    if (testConfig.UseMappedMemory()) {
        cudaHostGetDevicePointer(&inputDevice, input, 0);
        cudaHostGetDevicePointer(&outputDevice, output, 0);
    } else {
        cudaMalloc(&inputDevice, inputSize);
        cudaMalloc(&outputDevice,
                testConfig.NumOutputCategories() * sizeof(float));
    }

    float *bindings[2];
    bindings[inputBindingIndex] = inputDevice;
    bindings[outputBindingIndex] = outputDevice;

    // run and compute average time over numRuns iterations
    double avgTime = 0;
    for (int i = 0; i < testConfig.NumRuns() + 1; i++) {
        chrono::duration<double> diff;

        if (testConfig.UseMappedMemory()) {
            auto t0 = chrono::steady_clock::now();
            context->execute(1, (void**) bindings);
            auto t1 = chrono::steady_clock::now();
            diff = t1 - t0;
        } else {
            auto t0 = chrono::steady_clock::now();
            cudaMemcpy(inputDevice, input, inputSize, cudaMemcpyHostToDevice);
            context->execute(1, (void**) bindings);
            cudaMemcpy(output, outputDevice,
                    testConfig.NumOutputCategories() * sizeof(float),
                    cudaMemcpyDeviceToHost);
            auto t1 = chrono::steady_clock::now();
            diff = t1 - t0;
        }

        if (i != 0)
            avgTime += MS_PER_SEC * diff.count();

    }
    avgTime /= testConfig.NumRuns();
    // This is ridiculous...
    ostringstream conv;
    conv << avgTime;
    to_csv(csv_filename, pwr_devices,
            testConfig.GetNetName() + "-avgTime: " + conv.str());

    // save results to file
    int maxCategoryIndex = argmax(output, testConfig.NumOutputCategories())
            + 1001 - testConfig.NumOutputCategories();
    cout << "Most likely category id is " << maxCategoryIndex << endl;
    cout << "Average execution time in ms is " << avgTime << endl;
    ofstream outfile;
    outfile.open(testConfig.statsPath, ios_base::app);
    outfile << "\n" << testConfig.planPath << " " << avgTime;
    // << " " << maxCategoryIndex
    // << " " << testConfig.InputWidth() 
    // << " " << testConfig.InputHeight()
    // << " " << testConfig.MaxBatchSize() 
    // << " " << testConfig.WorkspaceSize() 
    // << " " << testConfig.dataType 
    // << " " << testConfig.NumRuns() 
    // << " " << testConfig.UseMappedMemory();
    outfile.close();

    cudaFree(inputDevice);
    cudaFree(outputDevice);

    cudaFreeHost(input);
    cudaFreeHost(output);
}

