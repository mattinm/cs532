#include "common.hxx"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

//OpenCL specific variables
cl::Buffer *gpu_v;
cl::Buffer *gpu_next;

void cleanup()
{
}

/*__global__ void gpu_diffuse(float *v, float *next, int xmax, int ymax, int zmax, int maxpos)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    int position = XYZINDEX(x, y, z, xmax, ymax);

    float sum = 0.0f;

    // make sure we're in bounds
    if (position >= maxpos || x >= xmax || y >= ymax || z >= zmax)
        return;

    // just the sides 
    sum += v[position];

    if (x > 0)
        sum += v[XYZINDEX(x-1, y, z, xmax, ymax)];
    if (x < (xmax-1))
        sum += v[XYZINDEX(x+1, y, z, xmax, ymax)];

    if (y > 0)
        sum += v[XYZINDEX(x, y-1, z, xmax, ymax)];
    if (y < (ymax-1))
        sum += v[XYZINDEX(x, y+1, z, xmax, ymax)];

    if (z > 0)
        sum += v[XYZINDEX(x, y, z-1, xmax, ymax)];
    if (z < (zmax-1))
        sum += v[XYZINDEX(x, y, z+1, xmax, ymax)];

    // update the value and minimize small values
    next[position] = sum / 7.0f;
    if (next[position] <= 0.01f)
        next[position] = 0.0f;
}*/

/* Updates at each timestep */
void update()
{
    // swap our gpu pointers around
    //swap_matrices(&gpu_v, &gpu_sum);
}

int main(int argc, char** argv)
{
    // initialize our data
    if (!initialize(argc, argv, &update, &cleanup, false)) return 1;
    int deviceType = CL_DEVICE_TYPE_GPU;

    try {
        // get our platform list
        std::vector<cl::Platform> platformList;
        cl::Platform::get(&platformList);

        cl_context_properties cprops[3] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(platformList[0])(),
            0
        };

        // create our context
        cl::Context context(deviceType, cprops);
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // create our command queue
        cl::CommandQueue queue(context, devices[0], 0);

        // read in our program as a string
        ifstream file("heat_diffusion_opencl.cl");
        string prog(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));

        // create our program
        cl::Program::Sources source(1, make_pair<prog.c_str(), prog.length()+1);
        cl::Program program(context, source);

        // close our file
        file.close();

        // build our program
        program.build(devices);

        // create our kernel
        cl::Kernel kernel(program, "heat_diffusion_opencl");

        // setup our memory
        gpu_v = cl::Buffer(context, CL_MEM_READ_WRITE, arr_size * sizeof(float));
        gpu_next = cl::Buffer(context, CL_MEM_READ_WRITE, arr_size * sizeof(float));

        // setup our args
        kernel.setArg(0, gpu_v);
        kernel.setArg(1, gpu_next);

        // initialize our data
        queue.enqueueWriteBuffer(gpu_v, CL_TRUE, 0, arr_size * sizeof(float), heat_matrix);
    } catch (cl::Error& err) {
        cout << "Caught exception: " << err.what() << "(" << err.err() << ")" << endl;
    }

    // start up opengl
    startOpengl(argc, argv);
}
