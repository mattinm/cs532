#include "kmeans.h"
#include <mpi.h>

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
    double *stars_lbr;
    int *clusters, *cluster_counts;
    double xmin, xmax, ymin, ymax, zmin, zmax;
    double xming, xmaxg, yming, ymaxg, zming, zmaxg;
    double *sums, *means;
    FILE **fps, *fp;

    int done, iterations;

    int comm_sz; /* MPI stuff */
    int rank;

    /* slices for the array */
    int *slice_sizes, *displacements, slice_size, zero_size;
    int *cluster_slice, *cluster_counts_local;
    double *slice, *sums_local;

    /* consume MPI commands */
    MPI_Init(&argc, &argv);

    /* ensure good input before we continue */
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

    /* setup MPI */
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* main process should read in the stars */
    stars_lbr = NULL;
    if (rank == 0) {
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

        /* allocate memory for the raw (lbr) stars */
        stars_lbr = malloc(sizeof(*stars_lbr) * num_stars * 3);

        /* read in the input of each file */
        xmin = xmax = ymin = ymax = zmin = zmax = 0.0;
        cur_stars = 0;
        index = 0;
        for (i = 0; i < num_files; ++i) {
            fp = fps[i];

            while (3 == fscanf(fp, " %lf %lf %lf", &l, &b, &r)) {
                stars_lbr[index++] = l;
                stars_lbr[index++] = b;
                stars_lbr[index++] = r;
                ++cur_stars;
            }
        }

        /* see if we got the correct number of stars */
        if (cur_stars != num_stars) {
            printf("WARNING: Incorrect number of stars found: %d vs %d\n", cur_stars, num_stars);
            num_stars = cur_stars;
        }

        /* close all of our files */
        cleanup_files(fps, num_files);

        /* print out some debug info */
        DEBUG_PRINTF(("Number of nodes: %d\n", comm_sz));
        DEBUG_PRINTF(("Number of files: %d\n\tNumber of stars: %d\n", num_files, num_stars));
        DEBUG_PRINTF(("Number of clusters: %d\n", num_clusters));
    }

    /* synchronize some basic data here */
    MPI_Bcast(&num_clusters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_stars, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* prepare the slices (this first process gets the remainder data) */
    slice_size = num_stars / comm_sz;
    zero_size = slice_size + num_stars - slice_size * comm_sz;

    slice_sizes = malloc(sizeof(*slice_sizes) * comm_sz);
    *slice_sizes = zero_size * 3;

    /* displacements are annoying */
    displacements = malloc(sizeof(*displacements) * comm_sz);
    *displacements = 0;

    for (i = 1; i < comm_sz; ++i) {
        slice_sizes[i] = slice_size * 3;
        displacements[i] = displacements[i-1] + slice_sizes[i-1];
    }

    if (rank == 0) {
        slice_size = zero_size;
    }

    slice = malloc(sizeof(*slice) * slice_size * 3);
    DEBUG_PRINTF(("NODE #%d | Size: %d\n", rank, slice_size));

    /* convert from LBR to XYZ */
    MPI_Scatterv(
            stars_lbr,          /* array to scatter */
            slice_sizes,        /* array of sizes */
            displacements,      /* array of offsets */
            MPI_DOUBLE,         /* datatype */
            slice,              /* memory for the slice */
            slice_sizes[rank],         /* size of the slice */
            MPI_DOUBLE,         /* datatype */
            0,                  /* where the main array is from */
            MPI_COMM_WORLD      /* send to all */
    );

    /* convert the slice to XYZ */
    for (i = 0; i < slice_size; ++i) {
        index = i * 3;
        l = slice[index] * M_PI / 180;
        b = slice[index+1] * M_PI / 180;
        r = slice[index+2];

        /* x value */
        x = r * cos(b) * sin(l);
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;

        /* y value */
        y = 4.2 - r * cos(b) * cos(l);
        if (y < ymin) ymin = y;
        if (y > ymax) ymax = y;
        
        /* z value */
        z = r * sin(b);
        if (z < zmin) zmin = z;
        if (z > zmax) zmax = z;

        /* update value inline */
        slice[index] = x;
        slice[index+1] = y;
        slice[index+2] = z;
    }

    DEBUG_PRINTF(("NODE #%d | X: (%lf, %lf) | Y: (%lf, %lf) | Z: (%lf, %lf)\n", rank, xmin, xmax, ymin, ymax, zmin, zmax));

    /* update the min / max for x / y / z */
    MPI_Reduce(&xmin, &xming, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ymin, &yming, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&zmin, &zming, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&xmax, &xmaxg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ymax, &ymaxg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&zmax, &zmaxg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* nullify some values here that no other node but the head neads */
    clusters = NULL;
    cluster_counts = NULL;
    sums = NULL;

    /* get our means by random within the extents */
    means = malloc(sizeof(*means) * num_clusters * 3);
    if (rank == 0) {
        /* only rank 0 cares about the gather values */
        clusters = malloc(sizeof(*clusters) * num_stars);
        cluster_counts = malloc(sizeof(*cluster_counts) * num_clusters);
        sums = malloc(sizeof(*sums) * num_clusters * 3);

        srand(time(NULL));
        DEBUG_PRINTF(("X: (%lf, %lf) | Y: (%lf, %lf) | Z: (%lf, %lf)\n", xming, xmaxg, yming, ymaxg, zming, zmaxg));
        for (i = 0; i < num_clusters; ++i) {
            index = i * 3;
            x = RANDOM_DOUBLE(xming, xmaxg);
            y = RANDOM_DOUBLE(yming, ymaxg);
            z = RANDOM_DOUBLE(zming, zmaxg);

            DEBUG_PRINTF(("\tMean #%d: (%.3lf, %.3lf, %.3lf)\n", i+1, x, y, z));

            means[index] = x;
            means[index+1] = y;
            means[index+2] = z;
        }
        DEBUG_PRINTF(("\n"));
    }

    /* synchronize the starting means */
    MPI_Bcast(means, num_clusters * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    /* allocate memory for the cluster assignment and sums */
    cluster_slice = malloc(sizeof(*clusters) * slice_size);
    cluster_counts_local = malloc(sizeof(*cluster_counts_local) * num_clusters);
    sums_local = malloc(sizeof(*sums_local) * num_clusters * 3);

    /* update */
    iterations = 0;
    do {
        /* reset the data */
        memset(sums_local, 0, sizeof(*sums) * num_clusters * 3);
        memset(cluster_counts_local, 0, sizeof(*cluster_counts_local) * num_clusters);

        /* update based on current means */
        for (i = 0; i < slice_size; ++i) {
            int min_cluster = 0;
            double xd, yd, zd;
            double cur_distance, min_distance;

            /* save our x/y/z coords for use later */
            index = i * 3;
            x = slice[index];
            y = slice[index+1];
            z = slice[index+2];

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
            sums_local[index] += x;
            sums_local[index+1] += y;
            sums_local[index+2] += z;

            /* save our cluster value and increment the sum */
            cluster_slice[i] = min_cluster;
            cluster_counts_local[min_cluster]++;
        }

        /* add up all the cluster counts */
        MPI_Reduce(
                cluster_counts_local,
                cluster_counts,
                num_clusters,
                MPI_INT,
                MPI_SUM,
                0,
                MPI_COMM_WORLD
        );

        /* add up all the sums */
        MPI_Reduce(
                sums_local,
                sums,
                num_clusters * 3,
                MPI_DOUBLE,
                MPI_SUM,
                0,
                MPI_COMM_WORLD
        );

        /* determine the new means */
        /* TODO: DISTRUBTE THIS? */
        if (rank == 0) {
            done = 1;
            for (i = 0; i < num_clusters; ++i) {
                double cur_meanx, cur_meany, cur_meanz;
                int cluster_count = cluster_counts[i];

                index = i * 3;

                if (!cluster_count) {
                    cur_meanx = RANDOM_DOUBLE(xmin, xmax);
                    cur_meany = RANDOM_DOUBLE(ymin, ymax);
                    cur_meanz = RANDOM_DOUBLE(zmin, zmax);

                    DEBUG_PRINTF(("New random mean #%d: (%.3lf, %.3lf, %.3lf)\n", i+1, cur_meanx, cur_meany, cur_meanz));

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

            printf("Iteration: %d\n\n", ++iterations);
        }

        /* send out the new means */
        MPI_Bcast(means, num_clusters * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&done, 1, MPI_INT, 0, MPI_COMM_WORLD);
        /* END TODO: DISTRIBUTE THIS? */
    } while (!done);

    /* sync up our clusters */
    *slice_sizes = zero_size;
    *displacements = 0;
    for (i = 1; i < comm_sz; ++i) {
        slice_sizes[i] /= 3;
        displacements[i] = displacements[i-1] + slice_sizes[i-1];
    }

    MPI_Gatherv(
            cluster_slice,
            slice_sizes[rank],
            MPI_INT,
            clusters,
            slice_sizes,
            displacements,
            MPI_INT,
            0,
            MPI_COMM_WORLD
    );

    if (rank == 0) {
#ifdef OUTPUTFILES
        char fname[255];
#endif

        /* print out the final means */
        printf("\nFinal Means\n");
        printf("===========\n");
        for (i = 0; i < num_clusters; ++i) {
            index = i * 3;
            printf("Mean #%d: %.3lf, %.3lf, %.3lf\n", i, means[index], means[index+1], means[index+2]);
        }

        /* print out the number of iterations */
        printf("\nTOTAL ITERATIONS: %d\n", iterations);

#ifdef OUTPUTFILES
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
#endif /* OUTPUTFILES */
    }

#ifdef OUTPUTFILES
CLEANUP: /* cleanup and exit */
#endif
    free(clusters);
    free(cluster_slice);
    free(cluster_counts);
    free(cluster_counts_local);
    free(displacements);
    free(means);
    free(stars_lbr);
    free(slice);
    free(slice_sizes);
    free(sums);
    free(sums_local);

    return MPI_Finalize() != MPI_SUCCESS;
}
