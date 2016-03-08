/* W B Langdon at MUN 10 May 2007
 * Program to demonstarte use of OpenGL's glDrawPixels
 */

#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
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
#include "math.h"

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

#define XYZINDEX(x, y, z, xmax, ymax) \
    ((z)*(xmax)*(ymax) + (y)*(xmax) + (x))

// function pointers than MUST be overridden
void (*updatefunc)(void) = NULL;
void (*cleanupfunc)(void) = NULL;

//this is the size of the heat matrix in the x, y and z dimensions
int x_cells, y_cells, z_cells, arr_size;

//This will hold the heat values for every cell in the matrix
float *heat_matrix, *next_heat_matrix;

//Size of our window
int window_size, window_width, window_height;

//Track our frame count
unsigned long framenum = 0L;

/* Gracefully kills the program */
void onExit(void)
{
    delete heat_matrix;
    delete next_heat_matrix;
    (*cleanupfunc)();
}

/**
 *  This code handles the camera, you can ignore it.
 *  It lets you zoom in (by holding control, click and moving the mouse), move left/right up/down by 
 *  holding shift, click and moving the mouse, and rotate by moving the mouse.
 */
int     ox                  = 0;
int     oy                  = 0;
int     buttonState         = 0; 
float   camera_trans[]      = {0, 0, -10};
float   camera_rot[]        = {0, -0.2, 0};
float   camera_trans_lag[]  = {0, 0, -10};
float   camera_rot_lag[]    = {0, -0.2, 0};
const float inertia         = 0.1f;

/* Reshapes the GL projection. */
void reshape(int w, int h) 
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 1000.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}

/* Handles mouse button events */
void mouse_button(int button, int state, int x, int y) 
{
    int mods;

    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    mods = glutGetModifiers();
    if (mods & GLUT_ACTIVE_SHIFT) {
        buttonState = 2;
    } else if (mods & GLUT_ACTIVE_CTRL) {
        buttonState = 3;
    }

    ox = x; oy = y;
    glutPostRedisplay();
}

/* Handles mouse movement events */
void mouse_motion(int x, int y)
{
    float dx = (float)(x - ox);
    float dy = (float)(y - oy);

    if (buttonState == 3) {
        // left+middle = zoom
        camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
    } else if (buttonState & 2) {
        // middle = translate
        camera_trans[0] += dx / 100.0f;
        camera_trans[1] -= dy / 100.0f;
    } else if (buttonState & 1) {
        // left = rotate
        camera_rot[0] += dy / 5.0f;
        camera_rot[1] += dx / 5.0f;
    }

    ox = x; oy = y;
    glutPostRedisplay();
}

/* Swaps two given matrices. */
inline void swap_matrices(float **a1, float **a2)
{
    float *temp;

    temp = *a1;
    *a1 = *a2;
    *a2 = temp;
}

/* Called over and over and over to render each individual frame */
void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // call the update function
    (*updatefunc)();
    cout << "Frame #" << framenum++ << endl;

    // setup the camera
    for (int c = 0; c < 3; ++c) {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }

    //reshape(window_width, window_height);
    glLoadIdentity();
    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

    static float width = 10.0f;
    static float half_width = width / 2.0f;

    static float x_half = half_width / x_cells;
    static float y_half = half_width / y_cells;
    static float z_half = half_width / z_cells;

    // draw the points for each point in the heat matrix
    glBegin(GL_POINTS);
    //glBegin(GL_QUADS);
    for (int i = 1; i < (x_cells-1); i++) {
        for (int j = 1; j < (y_cells-1); j++) {
            for (int k = 1; k < (z_cells-1); k++) {
                //set the red color of the point to the heat value (which should be between 0.0 and 1.0)
                float redness = next_heat_matrix[XYZINDEX(i, j, k, x_cells, y_cells)];
                if (!redness) continue;
                glColor3f(redness, 0.0f, 0.0f);

                //display the point on a grid
                float x_pos = -half_width + (i * width / x_cells);
                float y_pos = -half_width + (j * width / y_cells);
                float z_pos = -half_width + (k * width / z_cells);
                glVertex3f(x_pos, y_pos, z_pos);

                // this is for cubes
                /*
                float x1 = x_pos - x_half;
                float x2 = x_pos + x_half;
                float y1 = y_pos - y_half;
                float y2 = y_pos + y_half;
                float z1 = z_pos - z_half;
                float z2 = z_pos + z_half;

                // top face (y2);
                glVertex3f(x2, y2, z2);
                glVertex3f(x2, y2, z1);
                glVertex3f(x1, y2, z1);
                glVertex3f(x1, y2, z2);

                // bottom face (y1)
                glVertex3f(x2, y1, z1);
                glVertex3f(x2, y1, z2);
                glVertex3f(x1, y1, z2);
                glVertex3f(x1, y1, z1);

                // front face (z1)
                glVertex3f(x2, y1, z1);
                glVertex3f(x2, y2, z1);
                glVertex3f(x1, y2, z1);
                glVertex3f(x1, y1, z1);

                // back face (z2)
                glVertex3f(x2, y1, z2);
                glVertex3f(x2, y2, z2);
                glVertex3f(x1, y2, z2);
                glVertex3f(x1, y1, z2);

                // left face (x1)
                glVertex3f(x1, y1, z2);
                glVertex3f(x1, y2, z2);
                glVertex3f(x1, y2, z1);
                glVertex3f(x1, y1, z1);

                // right face (x2)
                glVertex3f(x2, y1, z1);
                glVertex3f(x2, y2, z1);
                glVertex3f(x2, y2, z2);
                glVertex3f(x2, y1, z2);
                */
            }
        }
    }
    glEnd();

    // push the buffer to the screen
    //glFlush();
    glutSwapBuffers();
    glutPostRedisplay();

    //swap the pointers between heat_matrix and heat_matrix_next
    swap_matrices(&heat_matrix, &next_heat_matrix);
}

/* Helper function to print usage. */
void usage(char *executable)
{
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

// helper function to initialize the data
int initialize(int argc, char **argv, void (*_updatefunc)(void), void (*_cleanupfunc)(void))
{
    // default values
    window_width = 500;
    window_height = 500;
    x_cells = 1000;
    y_cells = 1000;
    z_cells = 10;

    // set our update functions
    updatefunc = _updatefunc;
    cleanupfunc = _cleanupfunc;

    // parse the args
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
            return 0;
        }
    }

    // initilize the matrices to zero
    arr_size = x_cells * y_cells * z_cells;
    heat_matrix = new float[arr_size];
    next_heat_matrix = new float[arr_size];
    memset(heat_matrix, 0, arr_size * sizeof(float));
    memset(next_heat_matrix, 0, arr_size * sizeof(float));

    // make some areas to diffuse
    int xmax = x_cells * 1 / 3;
    int ymax = y_cells * 1 / 3;
    int zmax = z_cells / 3;
    for (int i = 1 ; i <= xmax; i++) {
        for (int j = 1 ; j <= ymax; j++) {
            for (int k = 1; k <= zmax ; k++) {
                heat_matrix[XYZINDEX(i, j, k, x_cells, y_cells)] = 1.0;
            }
        }
    }

    xmax = x_cells - 2;
    ymax = y_cells - 2;
    zmax = z_cells - 2;
    for (int i = x_cells * 2 / 3; i <= xmax; ++i) {
        for (int j = y_cells * 1 / 3; j <= ymax; ++j) {
            for (int k = z_cells * 1 / 3; k <= zmax; ++k) {
                heat_matrix[XYZINDEX(i, j, k, x_cells, y_cells)] = 1.0;
            }
        }
    }

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

    return 1;
}

/* Starts opengl. */
void startOpengl(int argc, char **argv)
{
    // initialize the opengl window
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
    glClearColor(0.0, 0.0, 0.0, 1.0f);

    //Modifying this will change the size of the points
//    glPointSize(2);

    atexit(onExit);
    glutMainLoop();
}
