#include <cuda.h>
#include <omp.h>




/* CUDA is implicitly C++ code, this is needed so out symbols are not mangled */
extern "C" {


#include "dbg.h"
#include "sobel.h"


struct pixel {
        unsigned char R;
        unsigned char G;
        unsigned char B;
        unsigned char A;
};


__constant__ kernel_t kernelX = { {-1, 0, 1}, 
                                  {-2, 0, 2},
                                  {-1, 0, 1} };

__constant__ kernel_t kernelY = { {-1, -2, -1},
                                  { 0,  0,  0},
                                  { 1,  2,  1} };



/* Return the next power of 2 of n, or n if n is already a power of 2 */
static inline uint32_t getNextPowerOf2(uint32_t n) {
    uint32_t cur = n;
    uint32_t rslt = 1;
    /* Find the previous power of 2 */
    while (cur >>= 1) {
        rslt <<= 1;
    }

    if (rslt == n) {
        return n;
    } else {
        return (rslt << 1);
    }
}



__device__ inline unsigned char greyscale_value(struct pixel *pImage, uint32_t pxNum)
{
        /* Get the GreyScale value of our pixel */
        unsigned char R, G, B, greyVal;
        R = pImage[pxNum].R;
        G = pImage[pxNum].G;
        B = pImage[pxNum].B;

        greyVal = (R + G + B) / 3;

        return greyVal;
}




/* pxNum is the index of the central pixel for the convolution in the pImage array */
__device__ inline int32_t convolution_by_3(struct pixel *pImage, kernel_t kernel,
                                           uint32_t pxNum, uint32_t width, uint32_t height)
{
        int32_t grad = 0;
        /* Line below */
        grad += kernel[0][0] * greyscale_value(pImage, pxNum + width + 1);
        grad += kernel[0][1] * greyscale_value(pImage, pxNum + width);
        grad += kernel[0][2] * greyscale_value(pImage, pxNum + width - 1);

        /* current line */
        grad += kernel[1][0] * greyscale_value(pImage, pxNum + 1);
        grad += kernel[1][1] * greyscale_value(pImage, pxNum);
        grad += kernel[1][2] * greyscale_value(pImage, pxNum - 1);

        /* line above */
        grad += kernel[2][0] * greyscale_value(pImage, pxNum - width + 1);
        grad += kernel[2][1] * greyscale_value(pImage, pxNum - width);
        grad += kernel[2][2] * greyscale_value(pImage, pxNum - width - 1);

        return grad;
}




__global__ void sobel_unnorm_kernel(struct pixel *pInImageData, uint16_t *pOutImageData,
                                    uint32_t width, uint32_t height, uint32_t basePx)
{
    uint32_t pxNum = blockIdx.x * blockDim.x + threadIdx.x + basePx;


    /* If we are on a border, do nothing */
    if (pxNum < width /* First line */
     || pxNum % width == 0 /* First column */
     || pxNum % width == width - 1 /* last column */
     || pxNum >= (width * (height - 1)) /* Last line */
    )
    {
        pOutImageData[pxNum] = 0;
    }
    else
    {
        int32_t gradX = convolution_by_3(pInImageData, kernelX, pxNum, width, height);
        int32_t gradY = convolution_by_3(pInImageData, kernelY, pxNum, width, height);
        float gradX_float = (float) gradX;
        float gradY_float = (float) gradY;

        uint16_t normGrad = (uint32_t) sqrt(gradX_float*gradX_float + gradY_float*gradY_float);

        pOutImageData[pxNum] = normGrad;
    }

}



/* This kernel will only handle the normalization.
   The size of the block MUST be a power of two, otherwise the behaviour
   is undefined (and most probably incorrect). */
__global__ void norm_image_kernel(uint16_t *pMaxGrads, uint16_t *pNonNormalized, struct pixel *pOutImage,
                                  uint32_t width, uint32_t height, uint32_t basePx)
{
    uint32_t pxNum = blockIdx.x * blockDim.x + threadIdx.x + basePx;

    uint16_t maxGrad = pMaxGrads[0];

    /* If we are on a border, do nothing */
    if (   pxNum < width /* First line */
            || pxNum % width == 0 /* First column */
            || pxNum % width == width - 1 /* last column */
            || pxNum >= (width * (height - 1)) /* Last line */
       )
    {
        pOutImage[pxNum].R = 0;
        pOutImage[pxNum].G = 0;
        pOutImage[pxNum].B = 0;
        pOutImage[pxNum].A = 255; /* Full opacity */
    }
    else
    {
        unsigned char greyVal = (255 * pNonNormalized[pxNum]) / maxGrad;
        pOutImage[pxNum].R = greyVal;
        pOutImage[pxNum].G = greyVal;
        pOutImage[pxNum].B = greyVal;
        pOutImage[pxNum].A = 255; /* Full transparency */
    }

}



__global__ void max_reduction_kernel(uint16_t *pMaxGrads, uint32_t width,
                                     uint32_t height, uint32_t nPxPerThread)
{
    extern __shared__ uint16_t sData[]; /* Contains the data for the local reduction */

    uint32_t pxNum = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid = threadIdx.x;

    /* Each thread copies its pixel, and maybe more if needed.
       If there is no pixel, then the value 0 is chosen (as it is neutral for a max) */
    if (pxNum < width * height) {
        sData[tid] = pMaxGrads[pxNum];
    } else {
        sData[tid] = 0;
    }

    for (int i = 1; i < nPxPerThread; i++) {
        uint32_t nextPxIdx = pxNum + blockDim.x * gridDim.x * i;
        if (nextPxIdx < width * height) {
            sData[tid] = max(sData[tid], pMaxGrads[nextPxIdx]);
        }
    }

    __syncthreads();

    /* Now, reduce in parallel to find the max */
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride = stride >> 1) {
        if (tid < stride) { /* If we are on the lowest part of the remaining array */
            sData[tid] = max(sData[tid], sData[tid + stride]);
        }

        __syncthreads();
    }


    if (tid == 0) {
        pMaxGrads[blockIdx.x] = sData[0];
    }

    /* At the end of that kernel, pMaxGrads[blockIdx.x] contains the max value of each block.
       We can recursively call the kernel on the smaller resulting array. */
}





void log_time(FILE *logFile, char *testName, uint32_t size, double t, int numThreads)
{
        if (logFile == NULL)
                return;

        fprintf(logFile, "{\"name\": \"%s\", \"size\": %u, \"nProcs\": 1, \"time\": %lf, \"throughput\": %lf},\n",
                testName, size, t, (double)size/t);
}



int sobel(struct image *const pInImage, struct image *pOutImage)
{
        check_warn (pInImage->type == RGBA, "In image must be RGBA");

        uint32_t width = pInImage->width;
        uint32_t height = pInImage->height;
        uint32_t nbPx = width * height;

        cudaError_t ret;
        struct cudaDeviceProp deviceProp;

        ret = cudaGetDeviceProperties(&deviceProp, 0);


        /* Get the threading limits of the device */
        int maxThreadsPerBlock = deviceProp.maxThreadsDim[0];
        int maxConcurrentBlocks = deviceProp.maxGridSize[0];
        int maxConcurrentThreads = maxConcurrentBlocks * maxThreadsPerBlock;
        
        /* Allocate memory on the device for in image, and non-normalized gradient norms,
           and copy the input image. */
        struct pixel *inImageDevice = NULL;
        uint16_t *outNonNormalized = NULL;

        ret = cudaMalloc((void **)&inImageDevice, width * height * sizeof(struct pixel));
        check_warn (ret == cudaSuccess, "Failed to allocate memory for in image on the device");

        ret = cudaMalloc((void **)&outNonNormalized, width * height * sizeof(uint16_t));
        check_warn (ret == cudaSuccess, "Failed to allocate memory for out image on the device");

        ret = cudaMemcpy(inImageDevice, pInImage->data, width * height * sizeof(struct pixel),
                         cudaMemcpyHostToDevice);
        check_warn (ret == cudaSuccess, "Failed to copy input image to device");

        double afterFirstMemcpys = omp_get_wtime();

        /* Now, we need to invoke the sobel kernel, that will make convolutions.
           We must be careful to invoke it as much times as necessary considering
           that there might be more pixels than allocatable threads. */
        for (uint32_t basePx = 0; basePx < nbPx; basePx += maxConcurrentThreads) {

            /* Don't use more blocks than necessary */
            uint32_t runningThreads = min(nbPx - basePx, maxConcurrentThreads);
            uint32_t nBlocks = runningThreads / maxThreadsPerBlock +
                              (runningThreads % maxThreadsPerBlock == 0 ? 0 : 1);

            sobel_unnorm_kernel <<< nBlocks, maxThreadsPerBlock >>>
                (inImageDevice, outNonNormalized, width, height, basePx);
        }


        /* We don't need the input image any longer */
        cudaFree(inImageDevice);

        /* Allocate memory for maximum gradient reduction, and copy non-normalized data */
        uint16_t *maxGradsDevice = NULL;

        ret = cudaMalloc((void **)&maxGradsDevice, width * height * sizeof(uint16_t));
        check_warn (ret == cudaSuccess, "Failed to allocate memory for out image on the device");
        ret = cudaMemcpy(maxGradsDevice, outNonNormalized, width * height * sizeof(uint16_t),
                         cudaMemcpyDeviceToDevice);
        check_warn (ret == cudaSuccess, "Failed to copy maxGradsDevice from device to device");

        /* And invoke iteratively our max-reduction kernel. The number of remaining
           elements is divised by the number of threads per block at each iteration.
           We must be careful: the reduction kernel only works if the number of threads
           on each block is a power of 2. */
        uint32_t remainingElems = nbPx;
        while (remainingElems > 1) {
            uint32_t threadsPerBlock = min(remainingElems, maxThreadsPerBlock);
            threadsPerBlock = getNextPowerOf2(threadsPerBlock);

            /* Don't allocate more blocks than possible. If there are too many pixels,
               some blocks will handle several pixels (handled in the kernel). */
            uint32_t nBlocks = min(remainingElems / threadsPerBlock + (remainingElems % threadsPerBlock == 0 ? 0 : 1),
                                   maxConcurrentBlocks);
            uint32_t nThreads = threadsPerBlock * nBlocks;
            uint32_t nPxPerThread = remainingElems / nThreads + (nbPx % nThreads == 0 ? 0 : 1);
            uint32_t sharedMem = threadsPerBlock * sizeof(uint16_t);

            max_reduction_kernel <<< nBlocks, threadsPerBlock, sharedMem >>>
                            (maxGradsDevice, width, height, nPxPerThread);

            /* One remaining element by running block */
            remainingElems = nBlocks;
        }
            

        /* Allocate memory for the final resulting image, and launch the normalization kernel.
           As for the convolution kernel, we might have to run it several times on different
           parts of the image. */
        struct pixel *outNormalizedDevice = NULL;
        ret = cudaMalloc((void **) &outNormalizedDevice, width * height * sizeof(struct pixel));
        check_warn(ret == cudaSuccess, "Failed to allocate memory for outNormalizedDevice");

        /* Now, it's time to call the kernel that normalises the image gradients and puts it
           into pixels */
        for (uint32_t basePx = 0; basePx < nbPx; basePx += maxConcurrentThreads) {

            /* Don't use more blocks than necessary */
            uint32_t runningThreads = min(nbPx - basePx, maxConcurrentThreads);
            uint32_t nBlocks = runningThreads / maxThreadsPerBlock +
                              (runningThreads % maxThreadsPerBlock == 0 ? 0 : 1);
                                    
            norm_image_kernel <<< nBlocks, maxThreadsPerBlock >>>
                (maxGradsDevice, outNonNormalized, outNormalizedDevice, width, height, basePx);
        }

        double beforeLastMemcpy = omp_get_wtime();

        log_time(stdout, "Without memory movements", width*height, beforeLastMemcpy - afterFirstMemcpys, 1);

        /* Copy the result from device, and we're done */
        pOutImage->width = width;
        pOutImage->height = height;
        pOutImage->type = RGBA;
        pOutImage->data = (unsigned char*)calloc(pOutImage->width * pOutImage->height, sizeof(struct pixel));

        cudaMemcpy(pOutImage->data, outNormalizedDevice, width * height * sizeof(struct pixel),
                   cudaMemcpyDeviceToHost);
        cudaFree(outNormalizedDevice);
        cudaFree(outNonNormalized);
        cudaFree(maxGradsDevice);

        return 0;
}




} /* extern "C" */
