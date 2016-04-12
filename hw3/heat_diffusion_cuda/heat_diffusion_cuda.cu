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

#define TILE_SIZE  16 

//CUDA specific variables
float *gpu_v;
float *gpu_sum;
cudaError_t err;
dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
dim3 dimGrid(1,1,1);


void cleanup()
{
    cudaFree(gpu_v);
    cudaFree(gpu_sum);
}

__global__ void gpu_diffuse(float *v, float *next, int xmax, int ymax, int zmax)
{
    __shared__ float z_share[TILE_SIZE][TILE_SIZE][3];

    int x = threadIdx.x + TILE_SIZE * blockIdx.x;
    int y = threadIdx.y + TILE_SIZE * blockIdx.y;

    int position = x + y * xmax;
    int zjump = xmax * ymax;
    bool inrect = x < xmax && y < ymax;

    float sum = 0.0f;
    float xprev = 0.0f;
    float xnext = 0.0f;
    float yprev = 0.0f;
    float ynext = 0.0f;

    // initialize the memory
    if (inrect) {
        z_share[threadIdx.x][threadIdx.y][1] = 0.0f;
        z_share[threadIdx.x][threadIdx.y][2] = v[position];
    } else {
        z_share[threadIdx.x][threadIdx.y][0] = 0.0f;
        z_share[threadIdx.x][threadIdx.y][1] = 0.0f;
        z_share[threadIdx.x][threadIdx.y][2] = 0.0f;
    }

    position -= zjump;
    for (int z = 0; z < zmax; ++z) {
        // sync up the before we go changing it
        __syncthreads();

        if (inrect) {
            position += zjump;

            // move the z layers forward one
            z_share[threadIdx.x][threadIdx.y][0] = z_share[threadIdx.x][threadIdx.y][1];
            z_share[threadIdx.x][threadIdx.y][1] = z_share[threadIdx.x][threadIdx.y][2];

            // update the next z
            if (z < (zmax-1)) {
                z_share[threadIdx.x][threadIdx.y][2] = v[position + zjump];
            } else {
                z_share[threadIdx.x][threadIdx.y][2] = 0.0f;
            }

            if (threadIdx.x == 0) {
                if (x > 0)
                    xprev = v[position - 1];
                else
                    xprev = 0.0f;
            } else if (threadIdx.x == (TILE_SIZE-1)) {
                if (x < (xmax-1))
                    xnext = v[position + 1];
                else
                    xnext = 0.0f;
            }

            if (threadIdx.y == 0) {
                if (y > 0)
                    yprev = v[position - xmax];
                else
                    yprev = 0.0f;
            } else if (threadIdx.y == (TILE_SIZE-1)) {
                if (y < (ymax-1))
                    ynext = v[position + xmax];
                else
                    ynext = 0.0f;
            }
        }

        // sync up the memory before using it
        __syncthreads();

        // only sum up if in rect 
        if (inrect) {
            // start with ourself
            sum = z_share[threadIdx.x][threadIdx.y][0]; 
            sum += z_share[threadIdx.x][threadIdx.y][1]; 
            sum += z_share[threadIdx.x][threadIdx.y][2];

            // do our x layers
            if (threadIdx.x == 0)
                sum += xprev;
            else
                sum += z_share[threadIdx.x-1][threadIdx.y][1]; 

            if (threadIdx.x == (TILE_SIZE-1))
                sum += xnext;
            else
                sum += z_share[threadIdx.x+1][threadIdx.y][1]; 

            // do our y layes
            if (threadIdx.y == 0)
                sum += yprev;
            else
                sum += z_share[threadIdx.x][threadIdx.y-1][1]; 

            if (threadIdx.y == (TILE_SIZE-1))
                sum += ynext;
            else
                sum += z_share[threadIdx.x][threadIdx.y+1][1]; 

            // update the value and minimize small values
            sum /= 7.0f;
            if (sum <= 0.01f)
                sum = 0.0f;

            next[position] = sum;
        }
    }
}

/* Updates at each timestep */
void update()
{
    gpu_diffuse<<<dimGrid, dimBlock>>>(gpu_v, gpu_sum, x_cells, y_cells, z_cells);

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
    
    dimGrid.x = (int)ceil((float)x_cells / dimBlock.x);
    dimGrid.y = (int)ceil((float)y_cells / dimBlock.y);

    cout << "Cuda Information" << endl;
    cout << "================" << endl;
    cout << "GRID:  " << dimGrid.x << ", " << dimGrid.y << ", " << dimGrid.z << endl;
    cout << "BLOCK: " << dimBlock.x << ", " << dimBlock.y << ", " << dimBlock.z << endl << endl;

    // start up opengl
    startOpengl(argc, argv);
}
