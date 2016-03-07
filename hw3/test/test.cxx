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

#include <cmath>

#define LIMIT(X, MIN, MAX) \
    if ((X) < (MIN)) X = MIN; \
    else if ((X) > (MAX)) X = MAX

// camera information
float cam_angle = 0.0f;
float cam_x = 0.0f, cam_y = 1.0f, cam_z = 5.0f;
float cam_lx = 0.0f, cam_ly = 0.0f, cam_lz = 1.0f;

float delta_angle = 0.0f, delta_move = 0.0f;
int x_origin = -1;

// background color
float red = 0.0f, green = 0.0f, blue = 0.0f;

/** Update the camera position */
inline void computePos(float deltaMove)
{
     cam_x += deltaMove * cam_lx * 0.1f;
     cam_z += deltaMove * cam_lz * 0.1f;
}

/** Updates the camera direction */
inline void computeDir(float deltaAngle)
{
    cam_lx = sin(cam_angle + deltaAngle);
    cam_lz = -cos(cam_angle + deltaAngle);
}

/** Process mouse button events. */
void mouse_button(int button, int state, int x, int y)
{
    switch (button) {
        case GLUT_LEFT_BUTTON:
            // reset on mouse release
            if (state == GLUT_UP) {
                cam_angle += delta_angle;
                delta_angle = 0.0f;
                x_origin = -1;
            } else {
                 x_origin = x;
            }

            break;
    }
}

/** Process mouse movement. */
void mouse_move(int x, int y)
{
    if (x_origin >= 0) {
        // update the camera angle
        delta_angle = (x - x_origin) * 0.002f;
        computeDir(delta_angle);
    }
}

/** Process special keys for camera movement.
 */
void special_keys(int key, int x, int y)
{
    int mod = glutGetModifiers();
    float fraction = 0.1f;

    switch (key) {
        case GLUT_KEY_LEFT:
            if (x_origin < 0) delta_angle -= 0.01f;
            break;

        case GLUT_KEY_RIGHT:
            if (x_origin < 0) delta_angle += 0.01f;
            break;

        case GLUT_KEY_UP:
            delta_move += 0.5f;
            break;

        case GLUT_KEY_DOWN:
            delta_move -= 0.5f;
            break;
    }
}

/** Called when we release a special key. */
void release_special_keys(int key, int x, int y)
{
    switch (key) {
        case GLUT_KEY_LEFT:
        case GLUT_KEY_RIGHT:
            if (x_origin < 0) delta_angle = 0.0f;
            break;

        case GLUT_KEY_UP:
            delta_move -= 0.5f;
            break;

        case GLUT_KEY_DOWN:
            delta_move += 0.5f;
            break;
    }
}

/** Called to setup our viewport whenever the window is resized.
 */
void reshape(int width, int height)
{
    // set some minimum constraints
    if (height < 100) height = 100;
    if (width < 100) width = 100;

    float ratio = width * 1.0f / height;

    // setup our projection from the identity projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, width, height);

    // update the perspective
    gluPerspective(45.0f, ratio, 0.1f, 100.0f);

    // reset to model view for drawing
    glMatrixMode(GL_MODELVIEW);
}

/** The main render function. This is called every frame to ensure
 * that the scene is redrawn.
 */
void render(void)
{
    // clear our buffer
    glClearColor(red, green, blue, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // reset the transformations
    glLoadIdentity();

    if (delta_move) computePos(delta_move);
    if (delta_angle && x_origin < 0) {
        cam_angle += delta_angle;
        computeDir(0);
    }

    // setup the camera
    gluLookAt(  cam_x,          cam_y, cam_z,
                cam_x + cam_lx, cam_y, cam_z + cam_lz,
                0.0f,           cam_y, 0.0f);

    // rotate the camera and update the rotation
    glRotatef(cam_angle, 0.0f, 1.0f, 0.0f);

    // draw some points
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    int xmax = 100, ymax = 100, zmax = 10;
    for (int x = 0; x < xmax; ++x) {
        for (int y = 0; y < ymax; ++y) {
            for (int z = 0; z < zmax; ++z) {
                glColor3f(x / (float)xmax, y / (float)ymax, z / (float)zmax);
                glVertex3f(-1.0f + 2.0f * x / (float)xmax, 0.0f + 2.0f * y / (float)ymax, -3.0f + 6.0f * z / (float)zmax);
            }
        }
    }
    glEnd();

    // swap buffers to draw
    //glFlush();
    glutSwapBuffers();

    // redraw every time
    glutPostRedisplay();
}

int main(int argc, char **argv)
{
    int window_width = 500, window_height = 500;
    char *title = "GLUT Test";

    // start our pos and dir
    computePos(delta_move);
    computeDir(0);

    // initialize glut
    glutInit(&argc, argv);

    // perpare our double-buffered window
    glutInitWindowPosition(-1, -1);
    glutInitWindowSize(window_width, window_height);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

    // create our windows
    glutCreateWindow(title);

    // setup our functions
    glutDisplayFunc(render);
    glutReshapeFunc(reshape);

    // keyboard functions
    glutIgnoreKeyRepeat(1);
    glutSpecialFunc(special_keys);
    glutSpecialUpFunc(release_special_keys);

    // mouse functions
    glutMouseFunc(mouse_button);
    glutMotionFunc(mouse_move);

    // enable depth testing
    glEnable(GL_DEPTH_TEST);

    // run glut forever
    glutMainLoop();

    return 1;
}
