/* W B Langdon at MUN 10 May 2007
 * Program to demonstarte use of OpenGL's glDrawPixels
 */

#include "common.hxx"

#ifdef STD_THREADING
# include <thread>
#endif

void cleanup()
{
    // nothing extra to cleanup
}

/* Updates at each timestep */
void thread_update(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax)
{
    int temp = 0, position = 0;
    float divisor = 27.0f;
    for (int i = xmin; i < xmax; ++i) {
        for (int j = ymin; j < ymax; ++j) {
            for (int k = zmin; k < zmax; ++k) {
                position = XYZINDEX(i, j, k, x_cells, y_cells);
                if (heat_matrix[position] <= 0.001f) {
                    next_heat_matrix[position] = 0.0f;
                    continue;
                }

                float sum = 0.0f;

                // just the sides 
                /*
                sum += heat_matrix[XYZINDEX(i, j, k, x_cells, y_cells)];
                sum += heat_matrix[XYZINDEX(i-1, j, k, x_cells, y_cells)];
                sum += heat_matrix[XYZINDEX(i+1, j, k, x_cells, y_cells)];
                sum += heat_matrix[XYZINDEX(i, j-1, k, x_cells, y_cells)];
                sum += heat_matrix[XYZINDEX(i, j+1, k, x_cells, y_cells)];
                sum += heat_matrix[XYZINDEX(i, j, k-1, x_cells, y_cells)];
                sum += heat_matrix[XYZINDEX(i, j, k+1, x_cells, y_cells)];
                */


                // also diagonals and self
                temp = XYZINDEX(i, j-1, k, x_cells, y_cells);
                sum += heat_matrix[temp-1];
                sum += heat_matrix[temp];
                sum += heat_matrix[temp+1];

                temp += x_cells;
                sum += heat_matrix[temp-1];
                sum += heat_matrix[temp];
                sum += heat_matrix[temp+1];

                temp += x_cells;
                sum += heat_matrix[temp-1];
                sum += heat_matrix[temp];
                sum += heat_matrix[temp+1];

                temp = XYZINDEX(i, j-1, k-1, x_cells, y_cells);
                sum += heat_matrix[temp-1];
                sum += heat_matrix[temp];
                sum += heat_matrix[temp+1];

                temp += x_cells;
                sum += heat_matrix[temp-1];
                sum += heat_matrix[temp];
                sum += heat_matrix[temp+1];

                temp += x_cells;
                sum += heat_matrix[temp-1];
                sum += heat_matrix[temp];
                sum += heat_matrix[temp+1];

                temp = XYZINDEX(i, j-1, k+1, x_cells, y_cells);
                sum += heat_matrix[temp-1];
                sum += heat_matrix[temp];
                sum += heat_matrix[temp+1];

                temp += x_cells;
                sum += heat_matrix[temp-1];
                sum += heat_matrix[temp];
                sum += heat_matrix[temp+1];

                temp += x_cells;
                sum += heat_matrix[temp-1];
                sum += heat_matrix[temp];
                sum += heat_matrix[temp+1];

                next_heat_matrix[position] = sum / divisor;

                if (next_heat_matrix[position] <= 0.001f)
                    next_heat_matrix[position] = 0.0f;
            }
        }
    }
}

void update()
{
#ifndef STD_THREADING
    thread_update(1, x_cells - 1, 1, y_cells - 1, 1, z_cells - 1);
#else
    int numthreads = std::thread::hardware_concurrency() - 1;

    int xsize = (x_cells - 1);
    int ysize = (y_cells - 1);
    int zsize = (z_cells - 1);

    int xwidth = (xsize - 1) / (numthreads + 1);
    int ywidth = (ysize - 1) / (numthreads + 1);
    int zwidth = (zsize - 1) / (numthreads + 1);

    int xmin = 1, ymin = 1, zmin = 1;
    int xmax, ymax, zmax;

    cout << "USING THREADS: " << (numthreads+1) << endl;

    // serial way
    for (int i = 0; i < numthreads; ++i) {
        xmax = xmin + xwidth;
        ymax = ymin + ywidth;
        zmax = zmin + zwidth;
        cout << "\tThread #" << (i+1) << ": (" << xmin << "-" << xmax << ", " << ymin << "-" << ymax << ", " << zmin << "-" << zmax << ")" << endl;
        thread_update(xmin, xmax, ymin, ymax, zmin, zmax);
        xmin = xmax;
        ymin = ymax;
        zmin = zmax;
    }

    xmax = xsize;
    ymax = ysize;
    zmax = zsize;
    cout << "\tThread #" << (numthreads+1) << ": (" << xmin << "-" << xmax << ", " << ymin << "-" << ymax << ", " << zmin << "-" << zmax << ")" << endl;
    thread_update(xmin, xmax, ymin, ymax, zmin, zmax);

    // force a true update
    //thread_update(1, xsize, 1, ysize, 1, zsize);

    return;

    // create our threads
    std::thread *threads = new std::thread[numthreads];
    for (int i = 0; i < numthreads; ++i) {
        xmax = xmin + xwidth;
        ymax = ymin + ywidth;
        zmax = zmin + zwidth;

        cout << "\tThread #" << (i+1) << ": (" << xmin << "-" << xmax << ", " << ymin << "-" << ymax << ", " << zmin << "-" << zmax << ")" << endl;

        threads[i] = std::thread(thread_update, xmin, xmax, ymin, ymax, zmin, zmax);
        xmin = xmax;
        ymin = ymax;
        zmin = zmax;
    }

    xmax = xsize;
    ymax = ysize;
    zmax = zsize;
    cout << "\tThread #" << (numthreads+1) << ": (" << xmin << "-" << xmax << ", " << ymin << "-" << ymax << ", " << zmin << "-" << zmax << ")" << endl;
    thread_update(xmin, xmax, ymin, ymax, zmin, zmax);

    // join our threads
    for (int i = 0; i < numthreads; ++i)
        threads[i].join();

    // delete our threads
    delete[] threads;
#endif // STD_THREADING
}

int main(int argc, char** argv)
{
    // initialize our data
    if (!initialize(argc, argv, &update, &cleanup)) return 1;

    // start up opengl
    startOpengl(argc, argv);
}
