__kernel void heat_diffusion_opencl(
                __global float *v, __global float *next,
                __global int xmax, __global int ymax, __global int zmax,
                __global int maxpos)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    int position = z * xmax * ymax + y * xmax + x;

    if (position < maxpos && x < xmax && y < ymax && z < zmax) {
        float sum = v[position];

        if (x > 0)
            sum += v[position-1];
        if (x < (xmax-1))
            sum += v[position+1];

        if (y > 0)
            sum += v[position-xmax];
        if (y < (ymax-1))
            sum += v[position+xmax];

        if (z > 0)
            sum += v[position-(xmax*ymax)];
        if (z < (zmax-1))
            sum += v[position+(xmax*ymax)];

        next[position] = sum / 7;
        if (next[position] <= 0.01)
            next[position] = 0;
    }
}
