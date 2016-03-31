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
#include "math.h"

#include "lodepng.h"

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

int seams_to_remove;
long start_time;

//find the pixel, and the RGBA (z is 0, 1, 2, 3)  part of that pixel
static int POSITION4(int x, int y, int z) {
    return (((y * window_width) + x) * 4) + z;
}

static int POSITION(int x, int y) {
    return ((y * window_width) + x);
}

/**
 *  The display function gets called repeatedly, updating the visualization of the simulation
 */
int count = 0;
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (count < seams_to_remove) {
        //Then calculate the costs and directions
        //initialze the top row first
        for (int x = 0; x < (window_width - count); x++) {
            costs[POSITION(x, window_height - 1)] = vals[POSITION(x, window_height - 1)];
            dirs[POSITION(x, window_height - 1)] = 0;   //doesn't really matter but we'll initialize it anyways
        }

        //calculate the rest of the costs and dirs
        for (int y = window_height - 2; y >= 0; y--) {
            //do the left side
            if (costs[POSITION(0, y+1)] < costs[POSITION(1, y+1)]) {
                costs[POSITION(0, y)] = vals[POSITION(0, y)] + costs[POSITION(0, y+1)];
                dirs[POSITION(0, y)] = 0;
            } else {
                costs[POSITION(0, y)] = vals[POSITION(0, y)] + costs[POSITION(1, y+1)];
                dirs[POSITION(0, y)] = 1;
            }

            int x;
            for (x = 1; x < (window_width - count) - 1; x++) {
                float cost_left  = costs[POSITION(x-1, y+1)];
                float cost_up    = costs[POSITION(x  , y+1)];
                float cost_right = costs[POSITION(x+1, y+1)];

    //            cout << "x: " << x << ", y: " << y << ", cost_left: " << cost_left << ", cost_up: " << cost_up << ", cost_right: " << cost_right << endl;

                if (cost_left < cost_up && cost_left < cost_right) {
                    costs[POSITION(x, y)] = vals[POSITION(x, y)] + costs[POSITION(x-1, y+1)];
                    dirs[POSITION(x, y)] = -1;
                } else if (cost_right < cost_left && cost_right < cost_up) {
                    costs[POSITION(x, y)] = vals[POSITION(x, y)] + costs[POSITION(x+1, y+1)];
                    dirs[POSITION(x, y)] = 1;
                } else {
                    costs[POSITION(x, y)] = vals[POSITION(x, y)] + costs[POSITION(x, y+1)];
                    dirs[POSITION(x, y)] = 0;
                }
            }

            //do the right size
            if (costs[POSITION(x, y+1)] < costs[POSITION(x-1, y+1)]) {
                costs[POSITION(x, y)] = vals[POSITION(x, y)] + costs[POSITION(x, y+1)];
                dirs[POSITION(x, y)] = 0;
            } else {
                costs[POSITION(x, y)] = vals[POSITION(x, y)] + costs[POSITION(x-1, y+1)];
                dirs[POSITION(x, y)] = -1;
            }
        }

        //calculate the same to remove
        //first get the min cost at the bottom row
        float min_val = 50000;
        for (int x = 0; x < (window_width - count); x++) {
            if (costs[POSITION(x,0)] < min_val) {
                min_val = costs[POSITION(x,0)];
                seam[0] = x;
    //            cout << "min_val now " << min_val << " for x: " << x << endl;
            }
        }

        for (int y = 1; y < window_height; y++) {
    //        cout << "calculating seam[" << y << "]: based on seam[" << (y-1) << "]: " << seam[y-1] << " + " << dirs[POSITION(seam[y-1], y-1)] << endl;
            seam[y] = seam[y-1] + dirs[POSITION(seam[y-1], y-1)];
        }

        /*
        cout << "seam:";
        for (int i = 0; i < window_height; i++) {
            cout << " " << seam[i];
        }
        cout << endl << endl;
        */


        //remove one line with the seamcarving algorithm
        for (int y = 0; y < window_height; y++) {
            int x;
            for (x = seam[y]; x < (window_width - count) - 1; x++) {
                inverted_image[POSITION4(x, y, 0)] = inverted_image[POSITION4(x+1, y, 0)];
                inverted_image[POSITION4(x, y, 1)] = inverted_image[POSITION4(x+1, y, 1)];
                inverted_image[POSITION4(x, y, 2)] = inverted_image[POSITION4(x+1, y, 2)];
                inverted_image[POSITION4(x, y, 3)] = inverted_image[POSITION4(x+1, y, 3)];

                vals[POSITION(x, y)] = vals[POSITION(x+1, y)];            
            }

            inverted_image[POSITION4(x, y, 0)] = 0;
            inverted_image[POSITION4(x, y, 1)] = 0;
            inverted_image[POSITION4(x, y, 2)] = 0;
            inverted_image[POSITION4(x, y, 3)] = 0;

            vals[POSITION(x, y)] = 0;
        }

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
