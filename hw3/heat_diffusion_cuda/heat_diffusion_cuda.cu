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
dim3 dimBlock;
dim3 dimGrid;

void cleanup()
{
    cudaFree(gpu_v);
    cudaFree(gpu_sum);
}

__global__ void gpu_diffuse(float *v, float *sum, int xmax, int ymax, int zmax, int maxpos)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    int position = z * xmax * ymax + y * xmax + x;

    if (x < 1 || x >= (xmax-1) || y < 1 || y >= (ymax-1) || z < 1 || z >= (zmax-1) || position >= maxpos) {
        sum[position] = 0.0;
    } else {
        int temp = z * xmax * ymax + xmax * (y - 1) + x;
        float p1 = v[temp-1];
        float p2 = v[temp];
        float p3 = v[temp+1];

        temp += xmax;
        float p4 = v[temp-1];
        float p5 = v[temp];
        float p6 = v[temp+1];

        temp += xmax;
        float p7 = v[temp-1];
        float p8 = v[temp];
        float p9 = v[temp+1];

        temp = (z-1) * xmax * ymax + xmax * (y - 1) + x;
        float p10 = v[temp-1];
        float p11 = v[temp];
        float p12 = v[temp+1];

        temp += xmax;
        float p13 = v[temp-1];
        float p14 = v[temp];
        float p15 = v[temp+1];

        temp += xmax;
        float p16 = v[temp-1];
        float p17 = v[temp];
        float p18 = v[temp+1];

        temp = (z+1) * xmax * ymax + xmax * (y - 1) + x;
        float p19 = v[temp-1];
        float p20 = v[temp];
        float p21 = v[temp+1];

        temp += xmax;
        float p22 = v[temp-1];
        float p23 = v[temp];
        float p24 = v[temp+1];

        temp += xmax;
        float p25 = v[temp-1];
        float p26 = v[temp];
        float p27 = v[temp+1];

        sum[position] = (
                p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 +
                p10 + p11 + p12 + p13 + p14 + p15 + p16 + p17 + p18 +
                p19 + p20 + p21 + p22 + p23 + p24 + p25 + p26 + p27
        ) / 27.0;
    }
}

/* Updates at each timestep */
void update()
{
    gpu_diffuse<<<dimGrid, dimBlock>>>(gpu_v, gpu_sum, x_cells, y_cells, z_cells, arr_size);

    err = cudaGetLastError();
    CUDAASSERT(err);

    err = cudaMemcpy(next_heat_matrix, gpu_sum, arr_size * sizeof(float), cudaMemcpyDeviceToHost);
    CUDAASSERT(err);

    // swap our gpu pointers around
    swap_matrices(&gpu_v, &gpu_sum);
}

int main(int argc, char** argv)
{
    // initialize our data
    if (!initialize(argc, argv, &update, &cleanup)) return 1;

    // create our CUDA data
    err = cudaMalloc((void**) &gpu_v, arr_size * sizeof(float));
    CUDAASSERT(err);
    err = cudaMalloc((void**) &gpu_sum, arr_size * sizeof(float));
    CUDAASSERT(err);

    // copy our CUDA data to the GPU
    err = cudaMemcpy(gpu_v, heat_matrix, arr_size * sizeof(float), cudaMemcpyHostToDevice);
    CUDAASSERT(err);
    err = cudaMemcpy(gpu_sum, next_heat_matrix, arr_size * sizeof(float), cudaMemcpyHostToDevice);
    CUDAASSERT(err);

    // setup our dims
    if (z_cells <= 3) dimBlock.z = 2;
    else if (z_cells <= 7) dimBlock.z = 4;

    if (x_cells <= 3) dimBlock.x = 2;
    else if (x_cells <= 7) dimBlock.x = 4;

    if (y_cells <= 3) dimBlock.y = 2;
    else if (y_cells <= 7) dimBlock.y = 4;

    dim3 dimBlock(8, 8, 8);
    dim3 dimGrid(ceil(x_cells / dimBlock.x), ceil(y_cells / dimBlock.y), ceil(z_cells / dimBlock.z));

    // start up opengl
    startOpengl(argc, argv);
}
