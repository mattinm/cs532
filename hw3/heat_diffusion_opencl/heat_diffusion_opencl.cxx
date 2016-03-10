#include "common.hxx"

#ifdef __APPLE__
# include <OpenCL/cl.hpp>
#else
# include <CL/cl.hpp>
#endif

#include <fstream>
#include <utility>

#define CLERROR(err) \
    if ((err) != CL_SUCCESS) { \
        cerr << "[" << __FILE__ << ":" << __LINE__ << "] " << endl; \
        exit(1); \
    }

#define CLERROR_BOOL(err) CLERROR(err ? CL_SUCCESS : -1)

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
            cl::NDRange(1, 1, 1),
            NULL,
            &event
    );
    CLERROR(err);

    // wait for the kernel to stop
    event.wait();

    // read back the data
    err = queue.enqueueReadBuffer(
            gpu_next,
            CL_TRUE,
            0,
            arr_size * sizeof(float),
            heat_matrix
    );
    CLERROR(err);

    // swap the gpu memory buffers
    err = kernel.setArg(0, gpu_next);
    CLERROR(err);
    err = kernel.setArg(1, gpu_v);
    CLERROR(err);
}

int main(int argc, char** argv)
{
    // initialize our data
    if (!initialize(argc, argv, &update, &cleanup, false)) return 1;

    int deviceType = CL_DEVICE_TYPE_GPU;

    // get our platform list
    std::vector<cl::Platform> platformList;
    cl::Platform::get(&platformList);
    CLERROR_BOOL(platformList.size());

    // print out our vendor
    std::string platformVendor;
    platformList[0].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
    cout << "Platform Vendor: " << platformVendor << endl;

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

    // open our file
    std::ifstream file("heat_diffusion_opencl.cl");
    CLERROR_BOOL(file.is_open());

    // read it in character by character
    std::string prog(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));

    // close our file
    file.close();

    // create our program
    cl::Program::Sources source(1, std::make_pair(prog.c_str(), prog.length()+1));
    cl::Program program(context, source);

    // print out our program
    cout << prog.c_str() << endl;

    // build our program
    err = program.build(devices, "");
    CLERROR(err);

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

    // create our command queue
    queue = cl::CommandQueue(context, devices[0], 0, &err);
    CLERROR(err);

    // initialize our data
    err = queue.enqueueWriteBuffer(gpu_v, CL_TRUE, 0, arr_size * sizeof(float), heat_matrix);
    CLERROR(err);

    // start up opengl
    startOpengl(argc, argv);
}
