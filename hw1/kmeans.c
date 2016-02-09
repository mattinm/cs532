#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
# define M_PI 3.14159265358979323846264338327
#endif

#define RANDOM_DOUBLE(min, max) ((double)(min) + (double)rand() / RAND_MAX * ((double)(max) - (double)(min)))

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
    int i, j;
    double l, b, r;
    double *stars, *star;
    int *clusters, *cluster, *cluster_counts, *cluster_count;
    double xmin, xmax, ymin, ymax, zmin, zmax;
    double *sums, *sum, *means, *mean;
    FILE **fps, *fp;

    int done;

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

    /* allocate memory for the cluster assignment */
    if (!(clusters = malloc(sizeof(*clusters) * num_stars)) ||
            !(cluster_counts = malloc(sizeof(*cluster_counts) * num_clusters)) ) {
        printf("Unable to allocate memory for clusters.\n");
        cleanup_files(fps, num_files);
        return 1;
    }

    /* allocate memory for the sums */
    if (!(sums = malloc(sizeof(*sums) * num_clusters * 3))) {
        printf("Unable to allocate memory for sums.\n");
        cleanup_files(fps, num_files);
        return 1;
    }

    /* allocate memory for the means */
    if (!(means = malloc(sizeof(*means) * num_clusters * 3))) {
        printf("Unable to allocate memory for means.\n");
        cleanup_files(fps, num_files);
        return 1;
    }

    /* read in the input of each file */
    xmin = xmax = ymin = ymax = zmin = zmax = 0.0;
    star = stars;
    for (i = 0; i < num_files; ++i) {
        fp = fps[i];
        while (3 == fscanf(fp, "%lf %lf %lf", &l, &b, &r)) {
            l = l * M_PI / 180;
            b = b * M_PI / 180;
    
            /* x value */
            *star = r * sin(b) * cos(l);
            if (*star < xmin) xmin = *star;
            if (*star > xmax) xmax = *star;

            /* y value */
            ++star; *star = r * sin(b) * sin(l);
            if (*star < ymin) ymin = *star;
            if (*star > ymax) ymax = *star;
            
            /* z value */
            ++star; *star = r * cos(b);
            if (*star < zmin) zmin = *star;
            if (*star > zmax) zmax = *star;

            ++star;
        }
    }

    /* print out some debug info */
    printf("Number of files: %d\n\tNumber of stars: %d\n", num_files, num_stars);
    printf("Number of clusters: %d\n", num_clusters);

    /* get our means by random within the extents */
    mean = means;
    for (i = 0; i < num_clusters; ++i) {
        double x, y, z;
        x = RANDOM_DOUBLE(xmin, xmax);
        y = RANDOM_DOUBLE(ymin, ymax);
        z = RANDOM_DOUBLE(zmin, zmax);

        printf("\tMean #%d: (%.3lf, %.3lf, %.3lf)\n", i+1, x, y, z);

        *mean++ = x;
        *mean++ = y;
        *mean++ = z;
    }

    printf("\n");

    /* update */
    do {
        /* reset the data */
        memset(sums, 0, sizeof(*sums) * num_clusters * 3);
        memset(clusters, 0, sizeof(*clusters) * num_stars);
        memset(cluster_counts, 0, sizeof(*cluster_counts) * num_clusters);

        /* update based on current means */
        star = stars;
        cluster = clusters;
        for (i = 0; i < num_stars; ++i) {
            int min_cluster = 0;
            double x, y, z;
            double xd, yd, zd;
            double cur_distance, min_distance;

            /* save our x/y/z coords for use later */
            x = *star++;
            y = *star++;
            z = *star++;

            /* go through our means by pointer */
            mean = means;
            xd = x + *mean++;
            yd = y + *mean++;
            zd = z + *mean++;

            /* save the current min distance */
            min_distance = xd * xd + yd * yd + zd * zd;

            /* compare against all other means distances */
            for (j = 1; j < num_clusters; ++j) {
                xd = *mean++;
                yd = *mean++;
                zd = *mean++;
                cur_distance = xd * xd + yd * yd + zd * zd;

                /* update the cluster and distance if this is smaller */
                if (cur_distance < min_distance) {
                    min_distance = cur_distance;
                    min_cluster = j;
                }
            }

            /* add our x/y/z to the correct sum */
            sum = &sums[min_cluster * 3];
            *sum++ += x;
            *sum++ += y;
            *sum++ += z;

            /* save our cluster value and increment the sum */
            *cluster++ = min_cluster;
            cluster_counts[min_cluster] += 1;
        }

        /* determine the new means */
        mean = means;
        sum = sums;
        cluster_count = cluster_counts;
        done = 1;
        for (i = 0; i < num_clusters; ++i) {
            double cur_meanx, cur_meany, cur_meanz;
            double cur_dx, cur_dy, cur_dz;

            /* get the mean values for x / y / z */
            cur_meanx = *sum++ / *cluster_count;
            cur_meany = *sum++ / *cluster_count;
            cur_meanz = *sum++ / *cluster_count++;

            /* get the difference from the stored mean x / y / z */
            cur_dx = cur_meanx - mean[0];
            cur_dy = cur_meany - mean[1];
            cur_dz = cur_meanz - mean[2];

            /* if we're more than 1% away, keep going */
            if (cur_dx > 0.01 || cur_dx < -0.01 ||
                    cur_dy > 0.01 || cur_dy < -0.01 ||
                    cur_dz > 0.01 || cur_dz < -0.01) {
                done = 0;
            }

            /* save the value */
            *mean++ = cur_meanx;
            *mean++ = cur_meany;
            *mean++ = cur_meanz;
        }
    } while (!done);

    /* close all of our files */
    cleanup_files(fps, num_files);

    /* print out the means */
    mean = means;
    for (i = 0; i < num_clusters; ++i) {
        printf("Mean %d: (%.3lf, %.3lf, %.3lf)\n", i+1, mean[0], mean[1], mean[2]);

        ++mean; ++mean; ++mean;
    }

    /* cleanup and exit */
    free(stars);

    return 0;
}
