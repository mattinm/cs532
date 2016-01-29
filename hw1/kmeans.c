#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
# define M_PI 3.14159265358979323846264338327
#endif

void print_usage()
{
    printf("Usage: ./kmeans num_clusters star_file1 <star_file2 ...>\n");
}

/** Cleans up our files by closing and freeing the memory. */
void cleanup_files(FILE **fps, int num_read)
{
    --num_read;
    for ( ; num_read >= 0; --num_read) fclose(fps[num_read]);
    free(fps);
}

int main(int argc, char **argv)
{
    int num_clusters, num_stars, num_files;
    int cur_stars;
    int i;
    double l, b, r;
    double *stars, *star;
    FILE **fps;
    FILE *fp;

    /* ensure good input */
    if (argc < 3) {
        print_usage();
        return 1;
    }
    
    /* get the number of clusters */
    --argc; ++argv;
    num_clusters = atoi(*argv);
    if (num_clusters <= 0) {
        print_usage();
        return 1;
    }

    /* allocate memory for our files */
    num_files = --argc; ++argv;
    if ((fps = malloc(sizeof(*fps) * num_files)) == NULL) {
        printf("Unable to allocate memory for %d files.\n", num_files);
        return 1;
    }

    /* open all of our files and update the num_stars */
    num_stars = 0;
    for (i = 0; i < num_files; ++i, ++argv) {
        if (!(fps[i] = fopen(*argv, "r"))) {
            printf("Failed to open file: %s!\n", *argv);

            /* cleanup the other files and exit */
            cleanup_files(fps, i);
            return 1;
        }

        /* read in the number of stars */
        if (1 != fscanf(fps[i], " %d", &cur_stars) || cur_stars <= 0) {
            printf("Unable to read the number of stars for file: %s!\n", *argv);

            /* cleanup the files and exit */
            cleanup_files(fps, i+1);
            return 1;
        }

        /* update our stars */
        num_stars += cur_stars;
    }

    /* allocate memory for the stars */
    if (!(stars = malloc(sizeof(*stars) * num_stars * 3))) {
        printf("Unable to allocate memory for stars.\n");
        cleanup_files(fps, num_files);
        return 1;
    }

    /* read in the input of each file */
    star = stars;
    for (i = 0; i < num_files; ++i) {
        fp = fps[i];
        while (3 == fscanf(fp, "%lf %lf %lf", &l, &b, &r)) {
            l = l * M_PI / 180;
            b = b * M_PI / 180;

            *star++ = r * sin(b) * cos(l);
            *star++ = r * sin(b) * sin(l);
            *star++ = r * cos(b);
        }
    }

    /* close all of our files */
    cleanup_files(fps, num_files);

    /* print out the first 10 stars */
    for (i = 0; i < 30; i += 3)
        printf("Star #%d: ( %lf, %lf, %lf )\n", (i+1)/3, stars[i], stars[i+1], stars[i+2]);

    /* cleanup and exit */
    free(stars);

    return 0;
}
