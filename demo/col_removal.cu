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


//find the pixel, and the RGBA (z is 0, 1, 2, 3)  part of that pixel
static int POSITION(int x, int y, int z) {
    return (((x * window_width) + y) * 4) + z;
}

/**
 *  The display function gets called repeatedly, updating the visualization of the simulation
 */
int count = 0;
void display() {
    float count;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //remove one line with the seamcarving algorithm
    int row_to_remove = drand48() * (window_width - count);
    for (int i = 0; i < window_height; i++) {
        for (int j = row_to_remove; j < (window_width - count); j++) {
            inverted_image[POSITION(i, j, 0)] = inverted_image[POSITION(i, j+1, 0)];
            inverted_image[POSITION(i, j, 1)] = inverted_image[POSITION(i, j+1, 1)];
            inverted_image[POSITION(i, j, 2)] = inverted_image[POSITION(i, j+1, 2)];
            inverted_image[POSITION(i, j, 3)] = inverted_image[POSITION(i, j+1, 3)];
        }

        inverted_image[POSITION(i, (window_width - count), 0)] = 0;
        inverted_image[POSITION(i, (window_width - count), 1)] = 0;
        inverted_image[POSITION(i, (window_width - count), 2)] = 0;
        inverted_image[POSITION(i, (window_width - count), 3)] = 0;
    }

    count++;


    glDrawPixels(window_width, window_height, GL_RGBA, GL_UNSIGNED_BYTE, inverted_image);

    glFlush();
    glutSwapBuffers();

    glutPostRedisplay();
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Invalid arguments." << endl;
        cerr << "Proper usage:" << endl;
        cerr << "\t" << argv[0] << " <png filename>" << endl;
        exit(1);
    }

    const char* filename = argv[1];

    //decode
    unsigned error = lodepng::decode(image, window_width, window_height, filename);

    //if there's an error, display it
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

    //the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...

    window_size = window_width * window_height;
    inverted_image = new unsigned char[window_size * 4];

    //the PNG is inverted in height as to how the pixels are displayed, so we need to flip it.
    for (int i = 0; i < window_height; i++) {
        for (int j = 0; j < window_width; j++) {
            inverted_image[POSITION((window_height - 1 - i), j, 0)] = image[POSITION(i, j, 0)];
            inverted_image[POSITION((window_height - 1 - i), j, 1)] = image[POSITION(i, j, 1)];
            inverted_image[POSITION((window_height - 1 - i), j, 2)] = image[POSITION(i, j, 2)];
            inverted_image[POSITION((window_height - 1 - i), j, 3)] = image[POSITION(i, j, 3)];
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
