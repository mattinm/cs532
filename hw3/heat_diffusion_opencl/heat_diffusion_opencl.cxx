#include "common.hxx"

#ifdef __APPLE__
# include <OpenCL/cl.hpp>
#else
# include <CL/cl.hpp>
#endif

#include <cstring>
#include <utility>

#define CLERROR(err) \
    if ((err) != CL_SUCCESS) { \
        cerr << "[" << __FILE__ << ":" << __LINE__ << "] : " << clErrorString(err) << endl; \
        exit(1); \
    }

#define CLERROR_BOOL(err) CLERROR(err ? CL_SUCCESS : 1)

inline const char *clErrorString(cl_int error)
{
    switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    
    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
    
    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: break;
    }
    
    return "Unknown OpenCL error";
}

// our kernel
std::string kernel_code = 
"__kernel void heat_diffusion_opencl(\n"
"                __global float *v, __global float *next,\n"
"                int xmax, int ymax, int zmax,\n"
"                int maxpos)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    int z = get_global_id(2);\n"
"\n"
"    int position = z * xmax * ymax + y * xmax + x;\n"
"\n"
"    if (position < maxpos && x < xmax && y < ymax && z < zmax) {\n"
"        float sum = v[position];\n"
"\n"
"        if (x > 0)\n"
"            sum += v[position-1];\n"
"        if (x < (xmax-1))\n"
"            sum += v[position+1];\n"
"\n"
"        if (y > 0)\n"
"            sum += v[position-xmax];\n"
"        if (y < (ymax-1))\n"
"            sum += v[position+xmax];\n"
"\n"
"        if (z > 0)\n"
"            sum += v[position-(xmax*ymax)];\n"
"        if (z < (zmax-1))\n"
"            sum += v[position+(xmax*ymax)];\n"
"\n"
"        next[position] = sum / 7.0f;\n"
"        if (next[position] <= 0.01f)\n"
"            next[position] = 0.0f;\n"
"    }\n"
"}";

//OpenCL specific variables
cl::Buffer gpu_v;
cl::Buffer gpu_next;
cl::Kernel kernel;
cl::CommandQueue queue;
cl_int err;

void cleanup()
{
    // everything *should* gracefully exit
}

/* Updates at each timestep */
void update()
{
    // run our kernel
    cl::Event event;
    err = queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange(x_cells, y_cells, z_cells),
            cl::NullRange,
            NULL,
            &event
    );
    CLERROR(err);

    // wait for the kernel to stop
    event.wait();

    // read back the data depending on our framenum
    err = queue.enqueueReadBuffer(
            (framenum & 1) ? gpu_v : gpu_next,
            CL_TRUE,
            0,
            arr_size * sizeof(float),
            heat_matrix
    );
    CLERROR(err);

    // swap the gpu memory buffers depending on our framenum
    err = kernel.setArg(0, (framenum & 1) ? gpu_v : gpu_next);
    CLERROR(err);
    err = kernel.setArg(1, (framenum & 1) ? gpu_next : gpu_v);
    CLERROR(err);
}

int main(int argc, char** argv)
{
    // initialize our data
    if (!initialize(argc, argv, &update, &cleanup, false)) return 1;

    // run on GPU by default, but allow CPU
    int deviceType = CL_DEVICE_TYPE_GPU;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--cpu") == 0)
            deviceType = CL_DEVICE_TYPE_CPU;
    }

    // get our platform list
    std::vector<cl::Platform> platformList;
    cl::Platform::get(&platformList);
    CLERROR_BOOL(platformList.size());

    // print out our vendor
    std::string platformVendor;
    platformList[0].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
    cout << endl << "Platform vendor: " << platformVendor << endl;

    cl_context_properties cprops[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platformList[0])(),
        0
    };

    // create our context
    cl::Context context(deviceType, cprops, NULL, NULL, &err);
    CLERROR(err);

    // get our device
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    CLERROR_BOOL(devices.size());

    cout << "Using device: " << devices[0].getInfo<CL_DEVICE_NAME>() << endl;

    // create our program
    cl::Program program(context, kernel_code, false, &err);
    CLERROR(err);

    // build our program
    err = program.build(devices);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        cerr << "BUILD FAILURE!" << endl << endl;
        cerr << "Build status:   " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << endl;
        cerr << "Build options:  " << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << endl;
        cerr << "Build log:      " << endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << endl;

        exit(1);
    }

    // create our kernel
    kernel = cl::Kernel(program, "heat_diffusion_opencl", &err);
    CLERROR(err);

    // setup our memory
    gpu_v = cl::Buffer(context, CL_MEM_READ_WRITE, arr_size * sizeof(float));
    gpu_next = cl::Buffer(context, CL_MEM_READ_WRITE, arr_size * sizeof(float));

    // setup our array args 
    err = kernel.setArg(0, gpu_v);
    CLERROR(err);
    err = kernel.setArg(1, gpu_next);
    CLERROR(err);

    // setup our scalar args
    err = kernel.setArg(2, x_cells);
    CLERROR(err);
    err = kernel.setArg(3, y_cells);
    CLERROR(err);
    err = kernel.setArg(4, z_cells);
    CLERROR(err);
    err = kernel.setArg(5, arr_size);
    CLERROR(err);

    // create our command queue
    queue = cl::CommandQueue(context, devices[0], 0, &err);
    CLERROR(err);

    // initialize our data
    err = queue.enqueueWriteBuffer(gpu_v, CL_TRUE, 0, arr_size * sizeof(float), heat_matrix);
    CLERROR(err);

    cout << endl;

    // start up opengl
    startOpengl(argc, argv);
}
