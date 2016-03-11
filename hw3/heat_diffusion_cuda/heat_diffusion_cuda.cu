/* W B Langdon at MUN 10 May 2007
 * Program to demonstarte use of OpenGL's glDrawPixels
 */

#include "common.hxx"
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDAASSERT(x) \
    if ((x) != cudaSuccess) { \
        cout << cudaGetErrorString(x) << " in file " << __FILE__ << " at line " << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    }

//CUDA specific variables
float *gpu_v;
float *gpu_sum;
cudaError_t err;
dim3 dimBlock(8,8,8);
dim3 dimGrid(1,1,1);

void cleanup()
{
    cudaFree(gpu_v);
    cudaFree(gpu_sum);
}

__global__ void gpu_diffuse(float *v, float *next, int xmax, int ymax, int zmax, int maxpos)
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
        sum += v[position-1];
    if (x < (xmax-1))
        sum += v[position+1];

    if (y > 0)
        sum += v[position-xmax];
    if (y < (ymax-1))
        sum += v[position+xmax];

    if (z > 0)
        sum += v[position-(xmax*ymax)];
    if (z < (zmax-1))
        sum += v[position+(xmax*ymax)];

    // update the value and minimize small values
    next[position] = sum / 7.0f;
    if (next[position] <= 0.01f)
        next[position] = 0.0f;
}

/* Updates at each timestep */
void update()
{
    gpu_diffuse<<<dimGrid, dimBlock>>>(gpu_v, gpu_sum, x_cells, y_cells, z_cells, arr_size);

    err = cudaGetLastError();
    CUDAASSERT(err);

    err = cudaMemcpy(heat_matrix, gpu_sum, arr_size * sizeof(float), cudaMemcpyDeviceToHost);
    CUDAASSERT(err);

    // swap our gpu pointers around
    swap_matrices(&gpu_v, &gpu_sum);
}

int main(int argc, char** argv)
{
    // initialize our data
    if (!initialize(argc, argv, &update, &cleanup, false)) return 1;

    // create our CUDA data
    err = cudaMalloc((void**) &gpu_v, arr_size * sizeof(float));
    CUDAASSERT(err);
    err = cudaMalloc((void**) &gpu_sum, arr_size * sizeof(float));
    CUDAASSERT(err);

    // copy our CUDA data to the GPU
    err = cudaMemcpy(gpu_v, heat_matrix, arr_size * sizeof(float), cudaMemcpyHostToDevice);
    CUDAASSERT(err);
    /*err = cudaMemcpy(gpu_sum, next_heat_matrix, arr_size * sizeof(float), cudaMemcpyHostToDevice);
    CUDAASSERT(err);*/

    // setup our dims
    
    if (z_cells <= 3) dimBlock.z = 2;
    else if (z_cells <= 7) dimBlock.z = 4;

    if (x_cells <= 3) dimBlock.x = 2;
    else if (x_cells <= 7) dimBlock.x = 4;

    if (y_cells <= 3) dimBlock.y = 2;
    else if (y_cells <= 7) dimBlock.y = 4;

    dimGrid.x = (int)ceil((float)x_cells / dimBlock.x);
    dimGrid.y = (int)ceil((float)y_cells / dimBlock.y);
    dimGrid.z = (int)ceil((float)z_cells / dimBlock.z);

    cout << "Cuda Information" << endl;
    cout << "================" << endl;
    cout << "GRID:  " << dimGrid.x << ", " << dimGrid.y << ", " << dimGrid.z << endl;
    cout << "BLOCK: " << dimBlock.x << ", " << dimBlock.y << ", " << dimBlock.z << endl << endl;

    // start up opengl
    startOpengl(argc, argv);
}
