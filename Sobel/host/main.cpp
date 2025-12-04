#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <CL/cl.h>
#include <opencv2/opencv.hpp>

// Se eliminan las variables globales de tamaño

const std::string AOCX_FILE_NAME = "pipeline.aocx";

// Función de ayuda para comprobar errores de OpenCL (muy útil para depurar)
void check_error(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error durante la operacion: " << operation << " (Codigo de error: " << err << ")" << std::endl;
        exit(1);
    }
}

// Función de ayuda para cargar el fichero binario .aocx
std::vector<unsigned char> load_binary_file(const std::string& file_name) {
    std::ifstream file(file_name, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el fichero binario " << file_name << std::endl;
        exit(1);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> buffer(size);
    if (!file.read((char*)buffer.data(), size)) {
        std::cerr << "Error: No se pudo leer el fichero binario." << std::endl;
        exit(1);
    }
    return buffer;
}

int main() {
    cl_int status;

    // --- 1. INICIALIZAR OPENCV Y OBTENER DIMENSIONES ---
    std::cout << ">>> 1. Abriendo webcam..." << std::endl;
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: No se pudo abrir la cámara." << std::endl;
        return -1;
    }

    // Pedir una resolución (opcional, pero recomendado)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Obtener las dimensiones REALES que la cámara nos ha dado
    const int WIDTH = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int HEIGHT = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "   - Resolucion real obtenida: " << WIDTH << "x" << HEIGHT << std::endl;

    // --- 2. INICIALIZAR OPENCL ---
    std::cout << ">>> 2. Inicializando OpenCL y la FPGA..." << std::endl;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device, NULL); // Usar ACCELERATOR para hardware
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    cl_command_queue read_queue = clCreateCommandQueue(context, device, 0, &status);
    cl_command_queue sobel_queue = clCreateCommandQueue(context, device, 0, &status);
    cl_command_queue write_queue = clCreateCommandQueue(context, device, 0, &status);
    
    // --- 3. CARGAR KERNEL .aocx ---
    std::cout << ">>> 3. Cargando .aocx y creando kernels..." << std::endl;
    std::vector<unsigned char> program_buffer = load_binary_file(AOCX_FILE_NAME);
    size_t program_size = program_buffer.size();
    cl_program program = clCreateProgramWithBinary(context, 1, &device, &program_size, (const unsigned char**)&program_buffer, NULL, &status);
    check_error(status, "clCreateProgramWithBinary");
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    check_error(status, "clBuildProgram");

    cl_kernel mem_read_kernel = clCreateKernel(program, "mem_read", &status);
    cl_kernel sobel_kernel = clCreateKernel(program, "sobel", &status);
    cl_kernel mem_write_kernel = clCreateKernel(program, "mem_write", &status);

    // --- 4. PREPARAR BUFFERS CON EL TAMAÑO CORRECTO ---
    std::cout << ">>> 4. Creando buffers de memoria del tamaño correcto..." << std::endl;
    //cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, WIDTH * HEIGHT * sizeof(unsigned char), NULL, &status);
    //cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, WIDTH * HEIGHT * sizeof(unsigned char), NULL, &status);
	
	// Le decimos al driver que queremos que esta memoria sea accesible (mapeable) por el host
	/*cl_mem_flags flags = CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY;

	// Creamos el buffer OpenCL en esta memoria especial
	cl_mem input_buffer(context, flags, WIDTH * HEIGHT * sizeof(unsigned char), NULL, &status);
	check_error(status, "create in buffer");
	uchar* puntero_cpu = (uchar*)read_queue.enqueueMapBuffer(
		input_buffer,
		CL_TRUE, // Bloqueante
		CL_MAP_WRITE,
		0,
		WIDTH * HEIGHT * sizeof(unsigned char),
		NULL,
		NULL,
		&status
	);
	check_error(status, "create in buffer pointer");*/
	
	
	int size = WIDTH * HEIGHT * sizeof(unsigned char);
	cl_int err;	
	cl_mem input_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, size, nullptr, &err);	
	check_error(err, "create in buffer");
	cl_mem output_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, size, nullptr, &err);	
	check_error(err, "create out buffer");
	
	
	
	
    // --- 5. CONFIGURAR ARGUMENTOS DE KERNELS (se hace una sola vez) ---
    clSetKernelArg(mem_read_kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(mem_read_kernel, 1, sizeof(int), &WIDTH);
    clSetKernelArg(mem_read_kernel, 2, sizeof(int), &HEIGHT);

    clSetKernelArg(sobel_kernel, 0, sizeof(int), &WIDTH);
    clSetKernelArg(sobel_kernel, 1, sizeof(int), &HEIGHT);

    clSetKernelArg(mem_write_kernel, 0, sizeof(cl_mem), &output_buffer);
    clSetKernelArg(mem_write_kernel, 1, sizeof(int), &WIDTH);
    clSetKernelArg(mem_write_kernel, 2, sizeof(int), &HEIGHT);

    std::cout << ">>> 6. Iniciando bucle de procesamiento en tiempo real..." << std::endl;
    std::cout << ">>> Pulsa 'q' o ESC en la ventana para salir." << std::endl;

    cv::Mat frame;
	uchar* puntero_salida = (uchar*)clEnqueueMapBuffer(write_queue, output_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, nullptr, nullptr, &err);
	check_error(err, "create pointer in buffer");
	cv::Mat img_salida(HEIGHT, WIDTH, CV_8UC1, puntero_salida);
	
	clEnqueueUnmapMemObject(write_queue, output_buffer, puntero_salida, 0, nullptr, nullptr);
	
    // El buffer de salida para el resultado de la FPGA
    std::vector<unsigned char> fpga_output_vector(WIDTH * HEIGHT);
	double ms_cam;
	double ms_pre;
	
	cv::Mat imagen = cv::imread("img.jpg", cv::IMREAD_GRAYSCALE);
	uchar* puntero_cpu = (uchar*)clEnqueueMapBuffer(read_queue, input_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, nullptr, nullptr, &err);
	check_error(err, "create pointer in buffer");
	
	cv::Mat img(HEIGHT, WIDTH, CV_8UC1, puntero_cpu);
	
	imagen.copyTo(img);
	
	clEnqueueUnmapMemObject(read_queue, input_buffer, puntero_cpu, 0, nullptr, nullptr);
	
    while (true) {
		auto start = std::chrono::high_resolution_clock::now();
        //cap >> frame;
		
		/*auto end_cam = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_cam = end_cam - start;
		ms_cam = elapsed_cam.count() * 1000.0;*/
		
		
        //if (frame.empty()) break;

        
        
		//uchar* puntero_cpu = (uchar*)clEnqueueMapBuffer(read_queue, input_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, nullptr, nullptr, &err);
		//check_error(err, "create pointer in buffer");
		//cv::Mat frame_gray(HEIGHT, WIDTH, CV_8UC1, puntero_cpu);
        // Pre-procesado en ARM
        //cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
		
		//cv::extractChannel(frame, frame_gray, 0);
		
        // Transferir frame a la FPGA
        //clEnqueueWriteBuffer(read_queue, input_buffer, CL_FALSE, 0, WIDTH * HEIGHT * sizeof(unsigned char), frame_gray.data, 0, NULL, NULL);
		//clEnqueueUnmapMemObject(read_queue, input_buffer, puntero_cpu, 0, nullptr, nullptr);
		
		/*auto end_preprocess = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_pre = end_preprocess - start;
		ms_pre = elapsed_pre.count() * 1000.0;*/
		
		
        // Lanzar el pipeline
        clEnqueueTask(read_queue, mem_read_kernel, 0, NULL, NULL);
        clEnqueueTask(sobel_queue, sobel_kernel, 0, NULL, NULL);
        clEnqueueTask(write_queue, mem_write_kernel, 0, NULL, NULL);

        // Esperar a que la FPGA termine
        //clFinish(read_queue);
        //clFinish(sobel_queue);
		clFinish(write_queue);

        // Recuperar el resultado
        //clEnqueueReadBuffer(write_queue, output_buffer, CL_TRUE, 0, WIDTH * HEIGHT * sizeof(unsigned char), fpga_output_vector.data(), 0, NULL, NULL);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double fps = 1.0 / elapsed.count();

        // Crear el cv::Mat para mostrarlo, AHORA CON LAS DIMENSIONES CORRECTAS
        //cv::Mat frame_procesado(HEIGHT, WIDTH, CV_8UC1, fpga_output_vector.data());
        
        std::string title = "Sobel en FPGA | FPS: " + std::to_string((int)fps);
        cv::imshow("Sobel FPGA", img_salida);
        cv::setWindowTitle("Sobel FPGA", title);

        char key = (char)cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }
	
	//std::cout << "Cam: " << ms_cam << std::endl;
	//std::cout << "Preprocess: " << ms_pre << std::endl;
    cap.release();
    cv::destroyAllWindows();
	//read_queue.enqueueUnmapMemObject(input_buffer, puntero_cpu, NULL, NULL, &err);
	clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseKernel(mem_read_kernel);
    clReleaseKernel(sobel_kernel);
    clReleaseKernel(mem_write_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(read_queue);
	clReleaseCommandQueue(sobel_queue);
	clReleaseCommandQueue(write_queue);
    clReleaseContext(context);
    
    return 0;
}