/* W B Langdon at MUN 10 May 2007
 * Program to demonstarte use of OpenGL's glDrawPixels
 */

#include "common.hxx"

void cleanup()
{
    // nothing extra to cleanup
}

/* Updates at each timestep */
void update()
{
    int temp = 0;
    for (int i = 1; i < (x_cells-1); ++i) {
        for (int j = 1; j < (y_cells-1); ++j) {
            for (int k = 1; k < (z_cells-1); ++k) {
                temp = XYZINDEX(i, j-1, k, x_cells, y_cells);
                float p1 = heat_matrix[temp-1];
                float p2 = heat_matrix[temp];
                float p3 = heat_matrix[temp+1];

                temp += x_cells;
                float p4 = heat_matrix[temp-1];
                float p5 = heat_matrix[temp];
                float p6 = heat_matrix[temp+1];

                temp += x_cells;
                float p7 = heat_matrix[temp-1];
                float p8 = heat_matrix[temp];
                float p9 = heat_matrix[temp+1];

                temp = XYZINDEX(i, j-1, k-1, x_cells, y_cells);
                float p10 = heat_matrix[temp-1];
                float p11 = heat_matrix[temp];
                float p12 = heat_matrix[temp+1];

                temp += x_cells;
                float p13 = heat_matrix[temp-1];
                float p14 = heat_matrix[temp];
                float p15 = heat_matrix[temp+1];

                temp += x_cells;
                float p16 = heat_matrix[temp-1];
                float p17 = heat_matrix[temp];
                float p18 = heat_matrix[temp+1];

                temp = XYZINDEX(i, j-1, k+1, x_cells, y_cells);
                float p19 = heat_matrix[temp-1];
                float p20 = heat_matrix[temp];
                float p21 = heat_matrix[temp+1];

                temp += x_cells;
                float p22 = heat_matrix[temp-1];
                float p23 = heat_matrix[temp];
                float p24 = heat_matrix[temp+1];

                temp += x_cells;
                float p25 = heat_matrix[temp-1];
                float p26 = heat_matrix[temp];
                float p27 = heat_matrix[temp+1];

                next_heat_matrix[XYZINDEX(i, j, k, x_cells, y_cells)] = (
                        p1  + p2  + p3  + p4  + p5  + p6  + p7  + p8  + p9  +
                        p10 + p11 + p12 + p13 + p14 + p15 + p16 + p17 + p18 +
                        p19 + p20 + p21 + p22 + p23 + p24 + p25 + p26 + p27
                ) / 27.0;
            }
        }
    }
}

int main(int argc, char** argv)
{
    // initialize our data
    if (!initialize(argc, argv, &update, &cleanup)) return 1;

    // start up opengl
    startOpengl(argc, argv);
}
