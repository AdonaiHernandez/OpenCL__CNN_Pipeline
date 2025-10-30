#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <CL/cl.h>

#include "model_params.h"

struct FrameSlot {
    std::vector<uint8_t> host_input_buffer;
    std::vector<uint8_t> host_output_buffer;
    cl_mem cl_buffer_input;
    cl_mem cl_buffer_output;
    cv::Mat display_frame; 
};

const int CONV1_WIDTH = 320;
const int CONV1_HEIGHT = 240;
const int CONV1_CHANNELS_IN = 3;
const int CONV1_X_ZERO_POINT = 127;
const int CONV1_OUT_WIDTH = 160;
const int CONV1_OUT_HEIGHT = 120;
const int CONV1_CHANNELS_OUT = 16;
const int CONV3_CHANNELS_OUT = 32;
const int PAD_SIZE = 1;
const int ALPHA_ZERO_POINT = 0;

cl_command_queue queue_read;
cl_command_queue queue_conv;
cl_command_queue queue_conv2;
cl_command_queue queue_conv3;
cl_command_queue queue_write;
cl_kernel kernel_mem_read ;
cl_kernel kernel_conv;
cl_kernel kernel_conv2;
cl_kernel kernel_conv3;
cl_kernel kernel_mem_write;
cl_mem buffer_weights;
cl_mem buffer_bias;
cl_mem buffer_weights_2;
cl_mem buffer_bias_2;
cl_mem buffer_weights_3;
cl_mem buffer_bias_3;
cl_int status;

struct Box {
    float xmin, ymin, xmax, ymax;
};

void check_error(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error durante la operacion: " << operation << " (Codigo de error: " << err << ")" << std::endl;
        exit(1);
    }
}

std::vector<unsigned char> read_binary_file(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Error: No se pudo abrir el archivo .aocx: " + filename);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Error al leer el archivo .aocx.");
    }
    return buffer;
}

std::vector<std::array<float,4>> generate_priors(int input_width=320, int input_height=240) {
    std::vector<std::array<float,4>> prior_boxes;
    std::vector<std::pair<int,int>> feature_maps = {{30,40},{15,20},{8,10},{4,5}};
    std::vector<std::vector<int>> min_sizes = {
        {10,16,24}, {32,48}, {64,96}, {128,192,256}
    };
    std::vector<int> steps = {8,16,32,64};

    for (size_t k = 0; k < feature_maps.size(); ++k) {
        int fm_h = feature_maps[k].first;
        int fm_w = feature_maps[k].second;
        int step = steps[k];

        for (int i = 0; i < fm_h; ++i) {
            for (int j = 0; j < fm_w; ++j) {
                for (int min_size : min_sizes[k]) {
                    float cx = (j + 0.5f) * step / input_width;
                    float cy = (i + 0.5f) * step / input_height;
                    float box_w = (float)min_size / input_width;
                    float box_h = (float)min_size / input_height;
                    prior_boxes.push_back({cx, cy, box_w, box_h});
                }
            }
        }
    }
    return prior_boxes;
}

std::vector<Box> decode_boxes(const std::vector<std::array<float,4>>& loc, 
                              const std::vector<std::array<float,4>>& priors, 
                              float variance_center=0.1f, float variance_size=0.2f) {
    std::vector<Box> boxes(loc.size());
    for (size_t i = 0; i < loc.size(); ++i) {
        float cx = priors[i][0] + loc[i][0] * variance_center * priors[i][2];
        float cy = priors[i][1] + loc[i][1] * variance_center * priors[i][3];
        float w = priors[i][2] * std::exp(loc[i][2] * variance_size);
        float h = priors[i][3] * std::exp(loc[i][3] * variance_size);

        boxes[i].xmin = cx - w / 2.0f;
        boxes[i].ymin = cy - h / 2.0f;
        boxes[i].xmax = cx + w / 2.0f;
        boxes[i].ymax = cy + h / 2.0f;
    }
    return boxes;
}


std::vector<int> non_max_suppression(const std::vector<Box>& boxes, const std::vector<float>& scores,
                                     float iou_threshold = 0.3f, float score_threshold = 0.6f) {
    std::vector<int> indices;
    std::vector<int> keep;

    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] >= score_threshold) indices.push_back((int)i);
    }

    std::sort(indices.begin(), indices.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];
    });

    while (!indices.empty()) {
        int current = indices[0];
        keep.push_back(current);
        std::vector<int> rest;

        for (size_t i = 1; i < indices.size(); ++i) {
            int idx = indices[i];
            float xx1 = std::max(boxes[current].xmin, boxes[idx].xmin);
            float yy1 = std::max(boxes[current].ymin, boxes[idx].ymin);
            float xx2 = std::min(boxes[current].xmax, boxes[idx].xmax);
            float yy2 = std::min(boxes[current].ymax, boxes[idx].ymax);

            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float inter = w * h;

            float area1 = (boxes[current].xmax - boxes[current].xmin) * (boxes[current].ymax - boxes[current].ymin);
            float area2 = (boxes[idx].xmax - boxes[idx].xmin) * (boxes[idx].ymax - boxes[idx].ymin);
            float ovr = inter / (area1 + area2 - inter);

            if (ovr <= iou_threshold)
                rest.push_back(idx);
        }
        indices = rest;
    }
    return keep;
}

void preprocess(const cv::Mat& img, uint8_t* input_data, int input_width = 320, int input_height = 240) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_width, input_height));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    
    int channel_size = input_width * input_height;
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_height; ++h) {
            for (int w = 0; w < input_width; ++w) {
                input_data[c * channel_size + h * input_width + w] = resized.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

cl_event opencl_execution(std::vector<uint8_t>* host_output_buffer, 
                      cl_mem slot_buffer_input, 
                      cl_mem slot_buffer_output,
                      cl_event write_event) { 

    status = clSetKernelArg(kernel_mem_read, 0, sizeof(cl_mem), &slot_buffer_input); 
    check_error(status, "clSetKernelArg(mem_read, slot_input)");
    status = clSetKernelArg(kernel_mem_write, 0, sizeof(cl_mem), &slot_buffer_output); 
    check_error(status, "clSetKernelArg(mem_write, slot_output)");

    cl_event kernel_read_event, kernel_write_event, read_buffer_event;

    status = clEnqueueTask(queue_read, kernel_mem_read, 1, &write_event, &kernel_read_event); 
    check_error(status, "clEnqueueTask(mem_read)");

    status = clEnqueueTask(queue_conv, kernel_conv, 0, NULL, NULL); 
    check_error(status, "clEnqueueTask(conv)");
    status = clEnqueueTask(queue_conv2, kernel_conv2, 0, NULL, NULL); 
    check_error(status, "clEnqueueTask(conv2)");
    status = clEnqueueTask(queue_conv3, kernel_conv3, 0, NULL, NULL); 
    check_error(status, "clEnqueueTask(conv3)");

    status = clEnqueueTask(queue_write, kernel_mem_write, 0, NULL, &kernel_write_event); 
    check_error(status, "clEnqueueTask(mem_write)");

    size_t out_size_bytes = host_output_buffer->size();
    status = clEnqueueReadBuffer(queue_write, slot_buffer_output, CL_FALSE, 0, out_size_bytes, host_output_buffer->data(), 1, &kernel_write_event, &read_buffer_event);
    check_error(status, "clEnqueueReadBuffer (Async)");
    
    return read_buffer_event;
}

int main(int argc, char* argv[]) {
	
	if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " modelo.onnx [cpu|opencl]" << std::endl;
        return 1;
    }
	
    std::string model_path = argv[1];
	std::string provider = argv[2];
    std::cout << "Cargando modelo: " << model_path << std::endl;
	
    const int input_width = 320; 
    const int input_height = 240;
	
    Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_ERROR, "UltraLightFace");
    Ort::SessionOptions session_options;
    
	//session_options.EnableProfiling("profile_onnx");
	
	std::cout << "Iniciamos onnx" << std::endl;
	Ort::CustomOpDomain custom_domain("com.mycompany.fpga");
	std::cout << "cargamos modelo" << std::endl;
    Ort::Session session(*env, model_path.c_str(), session_options);
	std::cout << "Creada session" << std::endl;

    const char* input_name = "191_quantized";
    const char* output_names[] = {"scores", "boxes"};

    std::vector<std::array<float,4>> priors = generate_priors(input_width, input_height);
	std::cout << "Generadas prior box" << std::endl;
	
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "No se pudo abrir la cámara\n";
        return -1;
    }
	
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 320.0);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240.0);

	
	//OPENCL
	const std::string AOCX_PATH = "pipeconv.aocx";
	std::cout << "Configurando OpenCL..." << std::endl;
	cl_platform_id platform = NULL;
	cl_uint num_platforms;
	status = clGetPlatformIDs(0, NULL, &num_platforms);
	check_error(status, "clGetPlatformIDs (num_platforms)");

	if (num_platforms > 0) {
		cl_platform_id* platforms = new cl_platform_id[num_platforms];
		status = clGetPlatformIDs(num_platforms, platforms, NULL);
		check_error(status, "clGetPlatformIDs (platforms)");
		for (cl_uint i = 0; i < num_platforms; ++i) {
			char pbuf[100];
			status = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(pbuf), pbuf, NULL);
			check_error(status, "clGetPlatformInfo");
			if (strstr(pbuf, "Intel") != NULL) {
				platform = platforms[i];
				std::cout << "Plataforma encontrada: " << pbuf << std::endl;
				break;
			}
		}
		delete[] platforms;
	}

	if (platform == NULL) {
		std::cerr << "No se encontró la plataforma de Intel FPGA." << std::endl;
		return -1;
	}

	cl_device_id device;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device, NULL);
	check_error(status, "clGetDeviceIDs");

	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
	check_error(status, "clCreateContext");

	queue_read = clCreateCommandQueue(context, device, 0, &status);
	queue_conv = clCreateCommandQueue(context, device, 0, &status);
	queue_conv2 = clCreateCommandQueue(context, device, 0, &status);
	queue_conv3 = clCreateCommandQueue(context, device, 0, &status);
	queue_write = clCreateCommandQueue(context, device, 0, &status);
	check_error(status, "clCreateCommandQueue");
	
	std::cout << "Cargando archivo binario .aocx..." << std::endl;
	std::vector<unsigned char> program_buffer = read_binary_file(AOCX_PATH);
	size_t program_size = program_buffer.size();
	const unsigned char* program_source_ptr = program_buffer.data();
	cl_program program = clCreateProgramWithBinary(context, 1, &device, &program_size, &program_source_ptr, NULL, &status);
	check_error(status, "clCreateProgramWithBinary");

	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	check_error(status, "clBuildProgram");

	std::cout << "Creando kernels..." << std::endl;
	kernel_mem_read = clCreateKernel(program, "mem_read", &status);
	check_error(status, "clCreateKernel(mem_read)");
	kernel_conv = clCreateKernel(program, "conv", &status);
	check_error(status, "clCreateKernel(conv)");
	kernel_conv2 = clCreateKernel(program, "conv2", &status);
	check_error(status, "clCreateKernel(conv2)");
	kernel_conv3 = clCreateKernel(program, "conv3", &status);
	check_error(status, "clCreateKernel(conv3)");
	kernel_mem_write = clCreateKernel(program, "mem_write", &status);
	check_error(status, "clCreateKernel(mem_write)");

	buffer_weights = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(conv1_weights), NULL, &status);
	check_error(status, "clCreateBuffer(buffer_weights)");
	buffer_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(conv1_bias), NULL, &status);
	check_error(status, "clCreateBuffer(buffer_bias)");
	buffer_weights_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(conv2_weights), NULL, &status);
	check_error(status, "clCreateBuffer(buffer_weights_2)");
	buffer_bias_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(conv2_bias), NULL, &status);
	check_error(status, "clCreateBuffer(buffer_bias_2)");
	buffer_weights_3 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(conv3_weights), NULL, &status);
	check_error(status, "clCreateBuffer(buffer_weights_3)");
	buffer_bias_3 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(conv3_bias), NULL, &status);
	check_error(status, "clCreateBuffer(buffer_bias_3)");

	status = clEnqueueWriteBuffer(queue_conv, buffer_weights, CL_TRUE, 0, sizeof(conv1_weights), conv1_weights, 0, NULL, NULL);
	check_error(status, "clEnqueueWriteBuffer(buffer_weights)");
	status = clEnqueueWriteBuffer(queue_conv, buffer_bias, CL_TRUE, 0, sizeof(conv1_bias), conv1_bias, 0, NULL, NULL);
	check_error(status, "clEnqueueWriteBuffer(buffer_bias)");
	status = clEnqueueWriteBuffer(queue_conv2, buffer_weights_2, CL_TRUE, 0, sizeof(conv2_weights), conv2_weights, 0, NULL, NULL);
	check_error(status, "clEnqueueWriteBuffer(buffer_weights_2)");
	status = clEnqueueWriteBuffer(queue_conv2, buffer_bias_2, CL_TRUE, 0, sizeof(conv2_bias), conv2_bias, 0, NULL, NULL);
	check_error(status, "clEnqueueWriteBuffer(buffer_bias_2)");
	status = clEnqueueWriteBuffer(queue_conv3, buffer_weights_3, CL_TRUE, 0, sizeof(conv3_weights), conv3_weights, 0, NULL, NULL);
	check_error(status, "clEnqueueWriteBuffer(buffer_weights_3)");
	status = clEnqueueWriteBuffer(queue_conv3, buffer_bias_3, CL_TRUE, 0, sizeof(conv3_bias), conv3_bias, 0, NULL, NULL);
	check_error(status, "clEnqueueWriteBuffer(buffer_bias_3)");

	status = clSetKernelArg(kernel_conv, 0, sizeof(cl_mem), &buffer_weights); check_error(status, "clSetKernelArg(conv, arg 0)");
	status = clSetKernelArg(kernel_conv, 1, sizeof(cl_mem), &buffer_bias); check_error(status, "clSetKernelArg(conv, arg 1)");
	status = clSetKernelArg(kernel_conv2, 0, sizeof(cl_mem), &buffer_weights_2); check_error(status, "clSetKernelArg(conv2, arg 0)");
	status = clSetKernelArg(kernel_conv2, 1, sizeof(cl_mem), &buffer_bias_2); check_error(status, "clSetKernelArg(conv2, arg 1)");
	status = clSetKernelArg(kernel_conv3, 0, sizeof(cl_mem), &buffer_weights_3); check_error(status, "clSetKernelArg(conv3, arg 0)");
	status = clSetKernelArg(kernel_conv3, 1, sizeof(cl_mem), &buffer_bias_3); check_error(status, "clSetKernelArg(conv3, arg 1)");

	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	
	std::cout << "Creando bufer de frame unico..." << std::endl;
	
	size_t input_bytes = (size_t)input_height * input_width * 3;
    size_t output_bytes = (size_t)CONV1_OUT_WIDTH * CONV1_OUT_HEIGHT * CONV3_CHANNELS_OUT;

    FrameSlot slot;

    slot.host_input_buffer.resize(input_bytes);
    slot.host_output_buffer.resize(output_bytes);
    slot.cl_buffer_input = clCreateBuffer(context, CL_MEM_READ_ONLY, input_bytes, NULL, &status);
    check_error(status, "clCreateBuffer(slot_input)");
    slot.cl_buffer_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_bytes, NULL, &status);
    check_error(status, "clCreateBuffer(slot_output)");
	
    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
	std::cout << "Iniciamos bucle single-thread..." << std::endl;

    //Loop principal
    while (true) {

        cap >> slot.display_frame;
        if (slot.display_frame.empty()) {
            std::cerr << "ADVERTENCIA: Frame vacío. Saltando..." << std::endl;
            continue; 
        }

        preprocess(slot.display_frame, slot.host_input_buffer.data(), input_width, input_height);
        
        frame_count++;


        // FPGA
        auto t1_fpga_start = std::chrono::high_resolution_clock::now();

        cl_event write_event;
        status = clEnqueueWriteBuffer(queue_read, slot.cl_buffer_input, CL_TRUE, 0, slot.host_input_buffer.size(), slot.host_input_buffer.data(), 0, NULL, &write_event);
        check_error(status, "clEnqueueWriteBuffer (Async)");
        
        cl_event final_read_event = opencl_execution(&(slot.host_output_buffer), slot.cl_buffer_input, slot.cl_buffer_output, write_event);

        status = clWaitForEvents(1, &final_read_event);
		check_error(status, "clWaitForEvents (final_read_event)");
        
        auto t1_fpga_end = std::chrono::high_resolution_clock::now();
        // Fin FPGA
        
        clReleaseEvent(write_event);
        clReleaseEvent(final_read_event);
        
        
        // ONNX
        auto t2_onnx_start = std::chrono::high_resolution_clock::now();

        Ort::Value input_tensor = Ort::Value::CreateTensor<uint8_t>(
            memory_info,
            slot.host_output_buffer.data(),
            slot.host_output_buffer.size(),
            std::vector<int64_t>{1, 32, 120, 160}.data(),
            4
        );
        
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                          &input_name, &input_tensor, 1, 
                                          output_names, 2);
										  
		auto t2_onnx_end = std::chrono::high_resolution_clock::now();
        // Fin ONNX


        // POSTPROCESS
        auto t3_post_start = std::chrono::high_resolution_clock::now();

        float* conf_ptr = output_tensors[0].GetTensorMutableData<float>();
        float* loc_ptr = output_tensors[1].GetTensorMutableData<float>();

        size_t num_boxes = priors.size();
        std::vector<std::array<float,2>> conf(num_boxes);
        std::vector<std::array<float,4>> loc(num_boxes);
        for (size_t i = 0; i < num_boxes; ++i) {
            conf[i][0] = conf_ptr[i*2 + 0]; conf[i][1] = conf_ptr[i*2 + 1];
            loc[i][0] = loc_ptr[i*4 + 0]; loc[i][1] = loc_ptr[i*4 + 1];
            loc[i][2] = loc_ptr[i*4 + 2]; loc[i][3] = loc_ptr[i*4 + 3];
        }

        std::vector<Box> boxes = decode_boxes(loc, priors);
        std::vector<float> scores(num_boxes);
        for (size_t i = 0; i < num_boxes; ++i) scores[i] = conf[i][1];
        std::vector<int> keep = non_max_suppression(boxes, scores, 0.3f, 0.6f);

        auto t3_post_end = std::chrono::high_resolution_clock::now();
        // Fin POSTPROCESS


        // Profile cada 100 frames
        if (frame_count % 100 == 99) { 
            std::chrono::duration<double, std::milli> T_fpga_ms = t1_fpga_end - t1_fpga_start;
            std::chrono::duration<double, std::milli> T_onnx_ms = t2_onnx_end - t2_onnx_start;
            std::chrono::duration<double, std::milli> T_post_ms = t3_post_end - t3_post_start;
            
            std::cout << "\n--- PROFILING (ms) ---" << std::endl;
            std::cout << "FPGA (Write+Kernel+Read): " << T_fpga_ms.count() << " ms" << std::endl;
            std::cout << "ONNX (session.Run):       " << T_onnx_ms.count() << " ms" << std::endl;
            std::cout << "POST (Decode+NMS):      " << T_post_ms.count() << " ms" << std::endl;
            std::cout << "------------------------\n" << std::endl;
        }


        // Pintamos en ventana
        cv::Mat& frame_to_display = slot.display_frame;
        float scale_w = (float)frame_to_display.cols / input_width;
        float scale_h = (float)frame_to_display.rows / input_height;

        for (int idx : keep) {
            int xmin = (int)(boxes[idx].xmin * input_width * scale_w);
            int ymin = (int)(boxes[idx].ymin * input_height * scale_h);
            int xmax = (int)(boxes[idx].xmax * input_width * scale_w);
            int ymax = (int)(boxes[idx].ymax * input_height * scale_h);
            cv::rectangle(frame_to_display, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0,255,0), 2);
            cv::putText(frame_to_display, cv::format("%.2f", scores[idx]), cv::Point(xmin, ymin - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
        }

        // FPS
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        double fps = 0.0;
        if (diff.count() > 0) {
             fps = 1 / diff.count();
        }
		
		start = std::chrono::high_resolution_clock::now();
        
		cv::putText(frame_to_display, cv::format("FPS: %.2f", fps), cv::Point(10, 20), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 2);

        cv::imshow("Ultra Light Face Detector", frame_to_display);

        if (cv::waitKey(1) == 27) {
             break;
        }
    }
   
    // Cleanup
	std::cout << "Deteniendo..." << std::endl;
    
    cap.release();
    cv::destroyAllWindows();

    clReleaseMemObject(slot.cl_buffer_input);
    clReleaseMemObject(slot.cl_buffer_output);
    clReleaseMemObject(buffer_weights);
    clReleaseMemObject(buffer_bias);
    clReleaseMemObject(buffer_weights_2);
    clReleaseMemObject(buffer_bias_2);
    clReleaseMemObject(buffer_weights_3);
    clReleaseMemObject(buffer_bias_3);
    clReleaseKernel(kernel_mem_read);
    clReleaseKernel(kernel_conv);
    clReleaseKernel(kernel_conv2);
    clReleaseKernel(kernel_conv3);
    clReleaseKernel(kernel_mem_write);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue_read);
    clReleaseCommandQueue(queue_conv);
    clReleaseCommandQueue(queue_conv2);
    clReleaseCommandQueue(queue_conv3);
    clReleaseCommandQueue(queue_write);
    clReleaseContext(context);
    delete env; 

    return 0;
}