#include <cuda.h>
#include <omp.h>



/* Pour gérer des images grandes, passer un paramètre nPxPerThreads est chiant, car ça décale
   tout tout le temps. Mais faut le faire...
*/




// shared memory does not work
#define USE_SHARED_MEM 0




/* CUDA is implicitly C++ code, this is needed so out symbols are not mangled */
extern "C" {

#include "dbg.h"
#include "sobel.h"


/*int maxNumThreads(int *pMaxGridWidth, int *pMaxGridHeight, int *pMaxBlockDim);*/
/*static inline int maxSquareDim(int numItems);*/


//XXX should change the world to use this...
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




//XXX We do redundant computation between threads, can be improved with shared memory
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
                                    uint32_t width, uint32_t height)
{
    /* Get your pixel number, which is your global threadIdx */
    uint32_t pxNum = blockIdx.x * blockDim.x + threadIdx.x;

    /* Be careful with borders */
    if (pxNum > width * height) {
        return;
    }

    /* If we are on a border, do nothing */
    if (   pxNum < width /* First line */
        || pxNum % width == 0 /* First column */
        || pxNum % width == width - 1 /* last column */
        || pxNum >= (width * (height - 1)) /* Last line */
       )
    {
        pOutImageData[pxNum] = 0; // We could also wait and get the max
    }
    else
    {
        int32_t gradX = convolution_by_3(pInImageData, kernelX, pxNum, width, height);
        int32_t gradY = convolution_by_3(pInImageData, kernelY, pxNum, width, height);
        float gradX_float = (float) gradX;
        float gradY_float = (float) gradY;

        uint16_t gradNorm = (uint32_t) sqrt(gradX_float*gradX_float + gradY_float*gradY_float);

        pOutImageData[pxNum] = gradNorm;
    }
    
    //XXX there should be the normalization on the device here !
}



void log_time(FILE *logFile, char *testName, uint32_t size, double t, int numThreads)
{
        //XXX should do something
}



int sobel(struct image *const pInImage, struct image *pOutImage)
{
        check_warn (pInImage->type == RGBA, "In image must be RGBA");

        uint32_t width = pInImage->width;
        uint32_t height = pInImage->height;

        cudaError_t ret;
        struct cudaDeviceProp deviceProp;

        // We will only use one device, the first one
        ret = cudaGetDeviceProperties(&deviceProp, 0);
        //XXX check ret

        /*int maxBlocks = deviceProp.maxGridSize[0];*/
        //XXX on pourrait juste autoriser les blocs à être carrés, mais la grille serait linéaire
        int maxLinearThreads = deviceProp.maxThreadsDim[0];

        dim3 threadsPerBlock(maxLinearThreads);

        int gridLength = (width * height) / maxLinearThreads +
                        ((width * height) % maxLinearThreads == 0 ? 0 : 1);

        dim3 nBlocks(gridLength);

        //XXX handle the case where there are not enough...
        /*if (gridWidth * gridHeight > maxGridWidth) { [> maxGridWidth is also the max number of blocks (I think) <]*/
            /* We will have to cut... But how do we do this ?? */
        /*}*/
        /* The number of invoked worker threads, used by kernels to know if they have
           to treat several pixels */
        /*int numThreads = gridWidth * gridHeight * maxBlockDim * maxBlockDim;*/


        printf("%d blocks of %d threads each, for %d total worker threads, and %d pixels\n",
                gridLength, maxLinearThreads, gridLength * maxLinearThreads, width*height);


        struct pixel *inImageDevice;
        /*unsigned char *outImageDevice;*/
        uint16_t *outImageDevice;

        /* Allocate memory on the device for both images (in and out) */
        ret = cudaMalloc((void **)&inImageDevice, width * height * sizeof(struct pixel));
        check_warn (ret == cudaSuccess, "Failed to allocate memory for in image on the device");

        ret = cudaMalloc((void **)&outImageDevice, width * height * 4 * sizeof(uint16_t));
        check_warn (ret == cudaSuccess, "Failed to allocate memory for out image on the device");

        /* Copy The input image on the device */
        ret = cudaMemcpy(inImageDevice, pInImage->data, width * height * sizeof(struct pixel),
                         cudaMemcpyHostToDevice);
        check_warn (ret == cudaSuccess, "Failed to copy the image to the device");


        /* Unnormalized version */
        uint16_t *pUnNormalizedOut = NULL;
        pUnNormalizedOut = (uint16_t *)calloc(width*height, sizeof(uint16_t));

        /* Allocate memory for the resulting image */
        pOutImage->width = pInImage->width;
        pOutImage->height = pInImage->height;
        pOutImage->type = RGBA;
        pOutImage->data = (unsigned char*)calloc(pOutImage->width * pOutImage->height, sizeof(struct pixel));
        /*check_mem(pOutImage->data);*/

        /* And launch the kernel */
        sobel_unnorm_kernel <<< nBlocks, threadsPerBlock >>> (inImageDevice, outImageDevice, width, height);

        ret = cudaMemcpy(pUnNormalizedOut, outImageDevice,
                         width * height * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        check_warn (ret == cudaSuccess, "Kernel failed: %s", cudaGetErrorString(ret));


        /* Normalize, on CPU for the moment */
        uint16_t max = 0;
        uint32_t maxPos = 0;
        for (uint32_t i = 0; i < width*height; i++) {
                if (pUnNormalizedOut[i] > max) {
                        max = pUnNormalizedOut[i];
                        maxPos = i;
                }
        }
        printf("Max: %u, at %u\n", max, maxPos);

        for (uint32_t i = 0; i < width*height; i++) {
                unsigned char greyVal = (255 * pUnNormalizedOut[i]) / max;
                pOutImage->data[4*i] = greyVal;
                pOutImage->data[4*i + 1] = greyVal;
                pOutImage->data[4*i + 2] = greyVal;
                pOutImage->data[4*i + 3] = 255; /* full opacity */
        }



        cudaFree(inImageDevice);
        cudaFree(outImageDevice);
        return 0;
        //XXX need a cleanup for image in case of failure.

/*error:*/
        /*cudaFree(inImageDevice);*/
        /*cudaFree(outImageDevice);*/
        /*free_and_null(pOutImage->data);*/
        /*return -1;*/
}




} /* extern "C" */
