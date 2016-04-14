/* W B Langdon at MUN 10 May 2007
 * Program to demonstarte use of OpenGL's glDrawPixels
 */

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef __APPLE__
#  include <OpenGL/gl.h>
#  include <OpenGL/glu.h>
#  include <GLUT/glut.h>
#else
#  include <GL/gl.h>
#  include <GL/glu.h>
#  include <GL/glut.h>
#endif

#include <string>
#include <queue>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include "math.h"

#include "lodepng.h"

#define INDEX4(x, y, z, width) ((((y) * (width) + (x)) * 4) + (z))
#define INDEX(x, y, width) ((y) * (width) + (x))

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::ostream;
using std::setw;
using std::right;
using std::left;
using std::fixed;
using std::vector;
using std::priority_queue;
using std::setprecision;

unsigned int window_width, window_height;
unsigned int window_size;

/**
 * the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...
 */

std::vector<unsigned char> image; //the raw pixels
unsigned char *inverted_image;
int *dirs;
float *greyscale;
float *vals;
float *costs;
int *seam;

// cuda variables
#define TILE_SIZE   1024
int *gpu_dirs;
float *gpu_costs;
float *gpu_vals;
int *gpu_seam;
unsigned char *gpu_inverted_image;

cudaError_t err;
dim3 dimBlock(TILE_SIZE, 1, 1);
dim3 dimGrid(1,1,1);

int seams_to_remove;
long start_time;
long start_time_costs;
long start_time_remove;

#define CUDAASSERT(x) \
    if ((x) != cudaSuccess) { \
        cout << cudaGetErrorString(x) << " in file " << __FILE__ << " at line " << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    }

//find the pixel, and the RGBA (z is 0, 1, 2, 3)  part of that pixel
static int POSITION4(int x, int y, int z) {
    return (((y * window_width) + x) * 4) + z;
}

static int POSITION(int x, int y) {
    return ((y * window_width) + x);
}

static long time_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000L + (long)(tv.tv_usec / 1000);
}

__global__ void gpu_calc_costs(float *costs, float *vals, int *dirs, int window_width, int window_height, int count)
{
    // tiled storage for the local
    extern __shared__ float shared_costs[];

    // determine our location
    int x;
    int index;
    int width = window_width - count;
    float current_costs[3];

    // fill in the top row first
    for (x = threadIdx.x; x < width; x += blockDim.x) {
        index = INDEX(x, window_height-1, window_width);

        // initialize our top row 
        shared_costs[x] = vals[index];
        costs[index] = shared_costs[x];
    }

    // go by column
    int it;
    float cost_left, cost_up, cost_right, cost;
    for (int y = window_height - 2; y >= 0; --y) {
        // sync before starting
        __syncthreads();

        for (x = threadIdx.x, it = 0; x < width; x += blockDim.x, ++it) {
            index = INDEX(x, y, window_width);

            // the left edges must know the last right value
            if (x == 0) {
                cost_left = 100000.0f;
            } else {
                cost_left = shared_costs[x-1];
            }

            cost_up = shared_costs[x];

            // the right edges must know the next left value
            if (x == (width-1)) {
                cost_right = 100000.0f;
            } else {
                cost_right = shared_costs[x+1];
            }

            // update our current shared cost and direction
            cost = vals[index];
            if (cost_left < cost_up && cost_left < cost_right) {
                cost += cost_left;
                dirs[index] = -1;
            } else if (cost_right < cost_left && cost_right < cost_up) {
                cost += cost_right;
                dirs[index] = 1;
            } else {
                cost += cost_up;
                dirs[index] = 0;
            }

            // update our main memory
            costs[index] = cost;
            current_costs[it] = cost;
        }

        // sync before flipping the shared_costs array to next
        __syncthreads();

        // update the shared_costs
        for (x = threadIdx.x, it = 0; x < width; x += blockDim.x, ++it) {
            shared_costs[x] = current_costs[it];
        }
    }

    // find the minimum value of y = 0
    if (threadIdx.x == 0) {
    }
}

__global__ void gpu_remove_seam(int *seam, unsigned char *inverted_image, float *vals, int *dirs, int window_width, int window_height, int count) {
    int x;
    int index;

    for (int y = threadIdx.x; y < window_height; y += blockDim.x) {
        x = seam[y];

        for ( ; x < (window_width - count - 1); ++x) {
            index = INDEX4(x, y, 0, window_width);

            inverted_image[index] = inverted_image[index+4];
            inverted_image[index+1] = inverted_image[index+5];
            inverted_image[index+2] = inverted_image[index+6];
            inverted_image[index+3] = inverted_image[index+7];

            index = INDEX(x, y, window_width);
            vals[index] = vals[index+1];            
        }

        x = window_width - count - 1;
        index = INDEX4(x, y, 0, window_width);
        inverted_image[index] = 0;
        inverted_image[index+1] = 0;
        inverted_image[index+2] = 0;
        inverted_image[index+3] = 0;

        vals[INDEX(x, y, window_width)] = 0;
    }
}

/**
 *  The display function gets called repeatedly, updating the visualization of the simulation
 */
int count = 0;
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (count < seams_to_remove) {
        start_time_costs = time_ms();
        float shared_array_size = window_width * sizeof(float);
        gpu_calc_costs<<<dimGrid, dimBlock, shared_array_size>>>(gpu_costs, gpu_vals, gpu_dirs, window_width, window_height, count);
        err = cudaGetLastError();
        CUDAASSERT(err);

        err = cudaMemcpy(costs, gpu_costs, sizeof(*costs) * window_size, cudaMemcpyDeviceToHost);
        CUDAASSERT(err);

        err = cudaMemcpy(dirs, gpu_dirs, sizeof(*dirs) * window_size, cudaMemcpyDeviceToHost);
        CUDAASSERT(err);

        cerr << count << "," << (time_ms() - start_time_costs);

        //calculate the seam to remove
        //first get the min cost at the bottom row
        float min_val = 50000.0f;
        for (int x = 0; x < (window_width - count); x++) {
            if (costs[POSITION(x,0)] < min_val) {
                min_val = costs[POSITION(x,0)];
                seam[0] = x;
                // cout << "min_val now " << min_val << " for x: " << x << endl;
            }
        }

        // calculate the seam array
        for (int y = 1; y < window_height; y++) {
    //        cout << "calculating seam[" << y << "]: based on seam[" << (y-1) << "]: " << seam[y-1] << " + " << dirs[POSITION(seam[y-1], y-1)] << endl;
            seam[y] = seam[y-1] + dirs[POSITION(seam[y-1], y-1)];
        }

        // copy the seam to the GPU
        err = cudaMemcpy(gpu_seam, seam, sizeof(*seam) * window_height, cudaMemcpyHostToDevice);
        CUDAASSERT(err);

        /*
        cout << "seam:";
        for (int i = 0; i < window_height; i++) {
            cout << " " << seam[i];
        }
        cout << endl << endl;
        */

        //remove one line with the seamcarving algorithm
        start_time_remove = time_ms();

        /////////////////////////////////////////////////
        ///// START OF CUDA REMOVE VARIANT
        /////////////////////////////////////////////////
        gpu_remove_seam<<<dimGrid, dimBlock>>>(gpu_seam, gpu_inverted_image, gpu_vals, gpu_dirs, window_width, window_height, count);
        err = cudaGetLastError();
        CUDAASSERT(err);

        // copy the inverted_image back from the GPU
        err = cudaMemcpy(inverted_image, gpu_inverted_image, sizeof(*inverted_image) * window_size * 4, cudaMemcpyDeviceToHost);
        /////////////////////////////////////////////////
        ///// END OF CUDA REMOVE VARIANT
        /////////////////////////////////////////////////

        /////////////////////////////////////////////////
        ///// START OF NORMAL REMOVE VARIANT
        /////////////////////////////////////////////////
        /*
        for (int y = 0; y < window_height; ++y) {
            int x = seam[y];
            int index = POSITION4(x, y, 0);
            int n = window_width - count - x - 1;

            // move the entire row over for the image
            memmove(&inverted_image[index], &inverted_image[index+4], sizeof(*inverted_image) * n * 4);

            // move the entire row over for val 
            index = POSITION(x, y);
            memmove(&vals[index], &vals[index]+1, sizeof(*vals) * n); 

            // set the last values to 0
            memset(&inverted_image[POSITION4(window_width - count - 1, y, 0)], 0, sizeof(*inverted_image) * 4);
            vals[POSITION(window_width - count - 1, y)] = 0;
        }

        // copy the new vals to the GPU
        err = cudaMemcpy(gpu_vals, vals, sizeof(*vals) * window_size, cudaMemcpyHostToDevice);
        CUDAASSERT(err);
        */
        /////////////////////////////////////////////////
        ///// END OF NORMAL REMOVE VARIANT
        /////////////////////////////////////////////////

        cerr << "," << (time_ms() - start_time_remove) << endl;
    } else if (count == seams_to_remove) {
        cout << "It took " << (time(NULL) - start_time) << " seconds to remove " << seams_to_remove << " seams." << endl;
    }

    count++;

    glDrawPixels(window_width, window_height, GL_RGBA, GL_UNSIGNED_BYTE, inverted_image);

    glFlush();
    glutSwapBuffers();

    glutPostRedisplay();
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Invalid arguments." << endl;
        cerr << "Proper usage:" << endl;
        cerr << "\t" << argv[0] << " <png filename> <seams to remove>" << endl;
        exit(1);
    }

    const char* filename = argv[1];
    seams_to_remove = atoi(argv[2]);
    start_time = time(NULL);

    //decode
    unsigned error = lodepng::decode(image, window_width, window_height, filename);

    //if there's an error, display it
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

    //the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...

    window_size = window_width * window_height;
    inverted_image = new unsigned char[window_size * 4];

    dirs = new int[window_size];
    costs = new float[window_size];
    greyscale = new float[window_size];
    vals = new float[window_size];
    seam = new int[window_height];

    //the PNG is inverted in height as to how the pixels are displayed, so we need to flip it.
    for (int y = 0; y < window_height; y++) {
        for (int x = 0; x < window_width; x++) {
            inverted_image[POSITION4(x, (window_height - 1 - y), 0)] = image[POSITION4(x, y, 0)];
            inverted_image[POSITION4(x, (window_height - 1 - y), 1)] = image[POSITION4(x, y, 1)];
            inverted_image[POSITION4(x, (window_height - 1 - y), 2)] = image[POSITION4(x, y, 2)];
            inverted_image[POSITION4(x, (window_height - 1 - y), 3)] = image[POSITION4(x, y, 3)];
        }
    }

    //get the average of all the color channels and use that as the value for each pixel.
    for (int y = 0; y < window_height; y++) {
        for (int x = 0; x < window_width; x++) {
            greyscale[POSITION(x, y)] = (inverted_image[POSITION4(x, y, 0)] +
                                         inverted_image[POSITION4(x, y, 1)] +
                                         inverted_image[POSITION4(x, y, 2)] +
                                         inverted_image[POSITION4(x, y, 3)]) * 0.25;
        }
    }

    //Calculate the gradient for every pixel
    for (int y = 0; y < window_height; y++) {
        for (int x = 0; x < window_width; x++) {
            float result = 0;

            if (x > 0)                  result += fabs(greyscale[POSITION(x, y)] - greyscale[POSITION(x-1, y)]);
            if (x < window_width - 1)   result += fabs(greyscale[POSITION(x, y)] - greyscale[POSITION(x+1, y)]);
            if (y > 0)                  result += fabs(greyscale[POSITION(x, y)] - greyscale[POSITION(x, y-1)]);
            if (y < window_height - 1)  result += fabs(greyscale[POSITION(x, y)] - greyscale[POSITION(x, y+1)]);

            vals[POSITION(x, y)] = result;
//            cout << "vals[" << x << ", " << y << "]: " << vals[POSITION(x,y)] << endl;
        }
    }

    // done with greyscale
    delete[] greyscale;
    greyscale = NULL;

    // setup our gpu memory
    err = cudaMalloc((void **) &gpu_dirs, sizeof(*dirs) * window_size);
    CUDAASSERT(err);
    err = cudaMalloc((void **) &gpu_costs, sizeof(*costs) * window_size);
    CUDAASSERT(err);
    err = cudaMalloc((void **) &gpu_vals, sizeof(*vals) * window_size);
    CUDAASSERT(err);
    err = cudaMalloc((void **) &gpu_inverted_image, sizeof(*inverted_image) * window_size * 4);
    CUDAASSERT(err);
    err = cudaMalloc((void **) &gpu_seam, sizeof(*seam) * window_height);
    CUDAASSERT(err);

    // copy in our current values
    err = cudaMemcpy(gpu_vals, vals, sizeof(*vals) * window_size, cudaMemcpyHostToDevice);
    CUDAASSERT(err);
    err = cudaMemcpy(gpu_inverted_image, inverted_image, sizeof(*inverted_image) * window_size * 4, cudaMemcpyHostToDevice);
    CUDAASSERT(err);

    cout << "Initialized Seam Carver!" << endl;
    cout << "window width: "    << window_width << endl;
    cout << "window height: "   << window_height << endl;
    cout << "window size : "    << window_size << endl;

    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Seam Carving");

    glutDisplayFunc(display);

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);

    glutMainLoop();
}
