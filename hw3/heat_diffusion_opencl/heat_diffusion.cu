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
#include <sstream>
#include <iomanip>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include "math.h"

#define CUDAASSERT(x) \
    if ((x) != cudaSuccess) { \
        cout << cudaGetErrorString(x) << " in file " << __FILE__ << " at line " << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    }

using std::cin;
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

//this is the size of the heat matrix in the x, y and z dimensions
int x_cells, y_cells, z_cells, arr_size;
//This will hold the heat values for every cell in the matrix
//TODO: this is two dimensional and you want to make it three dimensional
float *heat_matrix;
float *next_heat_matrix;
float *gpu_v;
float *gpu_sum;

int window_size;
int window_width;
int window_height;

cudaError_t err;

/**
 *  This code handles the camera, you can ignore it.
 *  It lets you zoom in (by holding control, click and moving the mouse), move left/right up/down by 
 *  holding shift, click and moving the mouse, and rotate by moving the mouse.
 */
int     ox                  = 0;
int     oy                  = 0;
int     buttonState         = 0; 
float   camera_trans[]      = {0, -0.2, -10};
float   camera_rot[]        = {0, 0, 0};
float   camera_trans_lag[]  = {0, -0.2, -10};
float   camera_rot_lag[]    = {0, 0, 0};
const float inertia         = 0.1f;

void onExit()
{
    delete heat_matrix;
    delete next_heat_matrix;
    cudaFree(gpu_v);
    cudaFree(gpu_sum);
}

__global__ void gpu_diffuse(float *v, float *sum, int xmax, int ymax, int zmax)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    int position = z * xmax * ymax + row * xmax + col;

    if (col < 1 || col >= (xmax-1) || row < 1 || row >= (ymax-1) || z < 1 || z >= (zmax-1)) {
        sum[position] = 0.0;
    } else {
        int temp = z * xmax * ymax + xmax * (row - 1) + col;
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

        temp = (z-1) * xmax * ymax + xmax * (row - 1) + col;
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

        temp = (z+1) * xmax * ymax + xmax * (row - 1) + col;
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

    //sum[position] = v[position] * 0.9f;
}

void reshape(int w, int h) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 1000.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}

void mouse_button(int button, int state, int x, int y) {
    int mods;

    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    mods = glutGetModifiers();
    if (mods & GLUT_ACTIVE_SHIFT) 
    {
        buttonState = 2;
    } 
    else if (mods & GLUT_ACTIVE_CTRL) 
    {
        buttonState = 3;
    }

    ox = x; oy = y;

    glutPostRedisplay();
}

void mouse_motion(int x, int y) {
    float dx = (float)(x - ox);
    float dy = (float)(y - oy);

    if (buttonState == 3) 
    {
        // left+middle = zoom
        camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
    } 
    else if (buttonState & 2) 
    {
        // middle = translate
        camera_trans[0] += dx / 100.0f;
        camera_trans[1] -= dy / 100.0f;
    }
    else if (buttonState & 1) 
    {
        // left = rotate
        camera_rot[0] += dy / 5.0f;
        camera_rot[1] += dx / 5.0f;
    }

    ox = x; oy = y;
    glutPostRedisplay();
}


static void swap_matrices(float **a1, float **a2) {
    float *temp;

    temp = *a1;
    *a1 = *a2;
    *a2 = temp;
}

/**
 *  The display function gets called repeatedly, updating the visualization of the simulation
 */
void display() {
    //TODO: this updates the heat simuilation in two dimensions, you want to do it in three dimensions,
    //and also on a GPU.
    //Note we're going to ignore the edge cells because they will act as a heat sink (their values will
    //always stay 0.
    /*
    for (int i = 1; i < x_cells - 1; i++) {
        for (int j = 1; j < y_cells - 1; j++) {
            int pos = j * x_cells + i;
            if (pos + x_cells + 1 >= arr_size) continue;
            next_heat_matrix[pos]  = (
                    heat_matrix[pos-1] + heat_matrix[pos] + heat_matrix[pos+1] +
                    heat_matrix[pos-1-x_cells] + heat_matrix[pos-x_cells] + heat_matrix[pos+1-x_cells] +
                    heat_matrix[pos-1+x_cells] + heat_matrix[pos+x_cells] + heat_matrix[pos+1+x_cells]
            ) / 9.0;
        }
    }*/

    err = cudaMemcpy(gpu_v, heat_matrix, arr_size * sizeof(float), cudaMemcpyHostToDevice);
    CUDAASSERT(err);

    dim3 dimBlock(8, 8, 8);
    dim3 dimGrid(ceil(x_cells / dimBlock.x), ceil(y_cells / dimBlock.y), ceil(z_cells / dimBlock.z));
    gpu_diffuse<<<dimGrid, dimBlock>>>(gpu_v, gpu_sum, x_cells, y_cells, z_cells);

    err = cudaMemcpy(next_heat_matrix, gpu_sum, arr_size * sizeof(float), cudaMemcpyDeviceToHost);
    CUDAASSERT(err);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

//    cout << "trans: " << camera_trans[0] << ", " << camera_trans[1] << ", " << camera_trans[2] << " -- rot: " << camera_rot[0] << ", " << camera_rot[1] << ", " << camera_rot[2] << endl;
    for (int c = 0; c < 3; ++c)
    {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }

    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

    glBegin(GL_POINTS);
    //cout << endl << "-------------------------------" << endl;
    for (int i = 0; i < x_cells; i++) {
        for (int j = 0; j < y_cells; j++) {
            for (int k = 0; k < z_cells; k++) {
                //set the red color of the point to the heat value (which should be between 0.0 and 1.0)
                glColor3f(next_heat_matrix[k * x_cells * y_cells + j * x_cells + i], 0.0f, 0.0f);

                //display the point on a grid between -5.0 and 5.0
                float x_pos = -5.0 + (i * (10.0 / x_cells));
                float y_pos = -5.0 + (j * (10.0 / y_cells));
                float z_pos = k * (10.0 / z_cells);

                //cout << "displaying cell at: " << i << ", " << j << " with color: " << next_heat_matrix[j * x_cells + i] << endl;

                glVertex3f(x_pos, y_pos, z_pos);
            }
        }
    }
    glEnd();
    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();

    //swap the pointers between heat_matrix and heat_matrix_next
    swap_matrices(&heat_matrix, &next_heat_matrix);
}

void usage(char *executable) {
    cerr << "Usage for heat_diffusion_simulation simulation:" << endl;
    cerr << "    " << executable << " <argument list>" << endl;
    cerr << "Possible arguments:" << endl;
    cerr << "   --window_size <x pixels (int)> <y pixels (int)>                 : window size" << endl;
    cerr << "   --sim_size <x cells (int)> <y cells (int)> <z cells (int)>      : number of cells in the simulation" << endl;
    cerr << "Defaults:" << endl;
    cerr << "   --window_size 500 500" << endl;
    cerr << "   --sim_size 1000 1000 1" << endl;
    exit(1);
}

int main(int argc, char** argv) {
    window_width = 500;
    window_height = 500;
    x_cells = 1000;
    y_cells = 1000;
    z_cells = 10;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--window_size") == 0) {
            window_width = atoi(argv[++i]);
            window_height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sim_size") == 0) {
            x_cells = atoi(argv[++i]);
            y_cells = atoi(argv[++i]);
            z_cells = atoi(argv[++i]);
        } else {
            cerr << "Unknown argument '" << argv[i] << "'." << endl;
            usage(argv[0]);
        }
    }

    /**
     * Iniitialize the initial values in the matrix to 0
     */
    //TODO: This is only two dimensional, update to three dimensions.
    arr_size = x_cells * y_cells * z_cells;
    heat_matrix = new float[arr_size];
    next_heat_matrix = new float[arr_size];
    memset(heat_matrix, 0, arr_size * sizeof(float));
    memset(next_heat_matrix, 0, arr_size * sizeof(float));

    err = cudaMalloc((void**) &gpu_v, arr_size * sizeof(float));
    CUDAASSERT(err);
    err = cudaMalloc((void**) &gpu_sum, arr_size * sizeof(float));
    CUDAASSERT(err);

    /**
     *  This makes areas of heat to diffuse in the simulation
     */
    //TODO: Update this to make some areas of heat in three dimensions
    for (int i = x_cells / 3 ; i < (int)(x_cells * 2 / 3); i++) {
        for (int j = y_cells / 3 ; j < (int)(y_cells * 2 / 3); j++) {
            for (int k = z_cells / 3; k < (int)(z_cells * 2 / 3); k++) {
                heat_matrix[k * x_cells * y_cells + j * x_cells + i] = 1.0;
            }
        }
    }

    err = cudaMemcpy(gpu_v, heat_matrix, arr_size * sizeof(float), cudaMemcpyHostToDevice);
    CUDAASSERT(err);
    err = cudaMemcpy(gpu_sum, next_heat_matrix, arr_size * sizeof(float), cudaMemcpyHostToDevice);
    CUDAASSERT(err);

    /*
    for (int i = x_cells * (3.0 / 6.0) ; i < x_cells * (5.0/6.0); i++) {
        for (int j = y_cells * (0.5 / 3.0) ; j < y_cells * (2.5/3.0); j++) {
            heat_matrix[i][j] = 1.0;
        }
    }*/

    cout << "Arguments succesfully parsed." << endl;
    cout << "    window_width:  " << setw(10) << window_width << endl;
    cout << "    window_height: " << setw(10) << window_height << endl;
    cout << "    x_cells:       " << setw(10) << x_cells << endl;
    cout << "    y_cells:       " << setw(10) << y_cells << endl;
    cout << "    z_cells:       " << setw(10) << z_cells << endl;

    window_size = window_width * window_height;

    cout << endl;
    cout << "x points: " << x_cells << endl;
    cout << "y points: " << y_cells<< endl;
    cout << "z points: " << z_cells << endl;
    cout << endl;

    cout << "Initialized heat diffusion simulation!" << endl;
    cout << "window width: "    << window_width << endl;
    cout << "window height: "   << window_height << endl;
    cout << "window size : "    << window_size << endl;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Heat Diffusion Simulation");

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse_button);
    glutMotionFunc(mouse_motion);
    //glutKeyboardFunc(keyboard);
    //glutIdleFunc(idle);

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);

    //Modifying this will change the size of the points
//    glPointSize(4);

    atexit(onExit);
    glutMainLoop();
}
