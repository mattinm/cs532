#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
# define M_PI 3.14159265358979323846264338327
#endif

#define RANDOM_DOUBLE(min, max) ((double)(min) + (double)rand() / RAND_MAX * ((double)(max) - (double)(min)))

#define RANDOM_INT(max) (rand() % (max))

#define COMP_PRECISION 0.0001

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
    int i, j, index, jindex;
    double l, b, r;
    double x, y, z;
    double *stars, *stars_lbr;
    int *clusters, *cluster_counts;
    double xmin, xmax, ymin, ymax, zmin, zmax;
    double *sums, *means;
    FILE **fps, *fp;
    char fname[255];

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
    for (i = 0; i < num_files; ++i) {
        if (!(fps[i] = fopen(argv[i], "r"))) {
            printf("Failed to open file: %s!\n", argv[i]);

            /* cleanup the other files and exit */
            cleanup_files(fps, i);
            return 1;
        }

        /* read in the number of stars */
        if (1 != fscanf(fps[i], " %d", &cur_stars) || cur_stars <= 0) {
            printf("Unable to read the number of stars for file: %s!\n", argv[i]);

            /* cleanup the files and exit */
            cleanup_files(fps, i);
            return 1;
        }

        /* update our stars */
        num_stars += cur_stars;
    }

    /* allocate memory for the stars */
    if (!(stars = malloc(sizeof(*stars) * num_stars * 3)) || !(stars_lbr = malloc(sizeof(*stars_lbr) * num_stars * 3))) {
        printf("Unable to allocate memory for stars.\n");
        cleanup_files(fps, num_files);
        return 1;
    }

    /* allocate memory for the cluster assignment */
    if (!(clusters = malloc(sizeof(*clusters) * num_stars)) ||
            !(cluster_counts = malloc(sizeof(*cluster_counts) * num_clusters)) ) {
        printf("Unable to allocate memory for clusters.\n");
        cleanup_files(fps, num_files);
        free(stars);
        free(stars_lbr);
        return 1;
    }

    /* allocate memory for the sums */
    if (!(sums = malloc(sizeof(*sums) * num_clusters * 3))) {
        printf("Unable to allocate memory for sums.\n");
        cleanup_files(fps, num_files);
        free(stars);
        free(stars_lbr);
        free(clusters);
        free(cluster_counts);
        return 1;
    }

    /* allocate memory for the means */
    if (!(means = malloc(sizeof(*means) * num_clusters * 3))) {
        printf("Unable to allocate memory for means.\n");
        cleanup_files(fps, num_files);
        free(stars);
        free(stars_lbr);
        free(clusters);
        free(cluster_counts);
        free(sums);
        return 1;
    }

    /* read in the input of each file */
    xmin = xmax = ymin = ymax = zmin = zmax = 0.0;
    cur_stars = 0;
    index = 0;
    for (i = 0; i < num_files; ++i) {
        fp = fps[i];

        while (3 == fscanf(fp, " %lf %lf %lf", &l, &b, &r)) {

            stars_lbr[index] = l;
            stars_lbr[index+1] = b;
            stars_lbr[index+2] = r;

            l = l * M_PI / 180;
            b = b * M_PI / 180;
    
            /* x value */
            x = r * cos(b) * sin(l);
            //x = r * sin(b) * cos(l);
            if (x < xmin) xmin = x;
            if (x > xmax) xmax = x;

            /* y value */
            y = 4.2 - r * cos(b) * cos(l);
            //y = r * sin(b) * sin(l);
            if (y < ymin) ymin = y;
            if (y > ymax) ymax = y;
            
            /* z value */
            z = r * sin(b);
            //z = r * cos(b);
            if (z < zmin) zmin = z;
            if (z > zmax) zmax = z;

            stars[index] = x;
            stars[index+1] = y;
            stars[index+2] = z;

            ++cur_stars;
            index += 3;
        }
    }

    if (cur_stars != num_stars) {
        printf("WARNING: Incorrect number of stars found: %d vs %d\n", cur_stars, num_stars);
        num_stars = cur_stars;
    }

    printf("X: (%.3lf, %.3lf)\nY: (%.3lf, %.3lf)\nZ: (%.3lf, %.3lf)\n", xmin, xmax, ymin, ymax, zmin, zmax);

    /* close all of our files */
    cleanup_files(fps, num_files);

    /* print out some debug info */
    printf("Number of files: %d\n\tNumber of stars: %d\n", num_files, num_stars);
    printf("Number of clusters: %d\n", num_clusters);

    /* get our means by random within the extents */
    srand(time(NULL));
    for (i = 0; i < num_clusters; ++i) {
        index = i * 3;
        x = RANDOM_DOUBLE(xmin, xmax);
        y = RANDOM_DOUBLE(ymin, ymax);
        z = RANDOM_DOUBLE(zmin, zmax);

        printf("\tMean #%d: (%.3lf, %.3lf, %.3lf)\n", i+1, x, y, z);

        means[index] = x;
        means[index+1] = y;
        means[index+2] = z;
    }

    printf("\n");

    /* update */
    do {
        /* reset the data */
        memset(sums, 0, sizeof(*sums) * num_clusters * 3);
        memset(cluster_counts, 0, sizeof(*cluster_counts) * num_clusters);

        /* update based on current means */
        for (i = 0; i < num_stars; ++i) {
            int min_cluster = 0;
            double xd, yd, zd;
            double cur_distance, min_distance;

            /* save our x/y/z coords for use later */
            index = i * 3;
            x = stars[index];
            y = stars[index+1];
            z = stars[index+2];

            /* go through our means by pointer */
            xd = x - means[0];
            yd = y - means[1];
            zd = z - means[2];

            /* save the current min distance */
            min_distance = xd * xd + yd * yd + zd * zd;

            /* compare against all other means distances */
            for (j = 1; j < num_clusters; ++j) {
                jindex = j * 3;
                xd = x - means[jindex];
                yd = y - means[jindex+1];
                zd = z - means[jindex+2];
                cur_distance = xd * xd + yd * yd + zd * zd;

                /* update the cluster and distance if this is smaller */
                if (cur_distance < min_distance) {
                    min_distance = cur_distance;
                    min_cluster = j;
                }
            }

            /* add our x/y/z to the correct sum */
            index = min_cluster * 3;
            sums[index] += x;
            sums[index+1] += y;
            sums[index+2] += z;

            /* save our cluster value and increment the sum */
            clusters[i] = min_cluster;
            cluster_counts[min_cluster]++;
        }

        /* determine the new means */
        done = 1;
        j = 0;
        printf("\n");
        for (i = 0; i < num_clusters; ++i) {
            double cur_meanx, cur_meany, cur_meanz;
            int cluster_count = cluster_counts[i];

            index = i * 3;

            if (!cluster_count) {
                cur_meanx = RANDOM_DOUBLE(xmin, xmax);
                cur_meany = RANDOM_DOUBLE(ymin, ymax);
                cur_meanz = RANDOM_DOUBLE(zmin, zmax);

                printf("New random mean #%d: (%.3lf, %.3lf, %.3lf)\n", i+1, cur_meanx, cur_meany, cur_meanz);

                means[index] = cur_meanx;
                means[index+1] = cur_meany;
                means[index+2] = cur_meanz;

                done = 0;
                continue;
            }

            /* get the mean values for x / y / z */
            cur_meanx = sums[index] / cluster_count;
            cur_meany = sums[index+1] / cluster_count;
            cur_meanz = sums[index+2] / cluster_count;

            /* if we're more than 1% away, keep going */
            if (fabs(cur_meanx - means[index]) > COMP_PRECISION || 
                    fabs(cur_meany - means[index+1]) > COMP_PRECISION || 
                    fabs(cur_meanz - means[index+2]) > COMP_PRECISION) {
                done = 0;

                /* save the value */
                means[index] = cur_meanx;
                means[index+1] = cur_meany;
                means[index+2] = cur_meanz;
            }

            printf("Mean %d: (%.3lf, %.3lf, %.3lf)\n", i+1, cur_meanx, cur_meany, cur_meanz);
        }
    } while (!done);

    /* output our clusters */
    if (!(fps = malloc(sizeof(*fps) * num_clusters))) {
        printf("Unable to allocate memory for output files.\n");
        goto CLEANUP;
    }

    /* open our save files */
    for (i = 0; i < num_clusters; ++i) {
        snprintf(fname, 255, "cluster%d.txt", i);
        if (!(fp = fopen(fname, "w"))) {
            printf("Unable to open file for writing: %s\n", fname);
            cleanup_files(fps, i);
            goto CLEANUP;
        }

        fprintf(fp, "%d\n", cluster_counts[i]);
        fps[i] = fp;
    }

    /* go through all our stars and write them to the correct file */
    for (i = 0; i < num_stars; ++i) {
        index = i * 3;

        /* write it to the correct cluster */
        fprintf(fps[clusters[i]], "%lf %lf %lf\n", stars_lbr[index], stars_lbr[index+1], stars_lbr[index+2]);
    }

    /* cleanup our files */
    cleanup_files(fps, num_clusters);

CLEANUP: /* cleanup and exit */
    free(clusters);
    free(cluster_counts);
    free(means);
    free(stars);
    free(stars_lbr);
    free(sums);

    return 0;
}
