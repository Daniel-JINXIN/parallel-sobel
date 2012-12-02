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


int maxNumThreads(int *pMaxGridWidth, int *pMaxGridHeight, int *pMaxBlockDim);
static inline int maxSquareDim(int numItems);


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
<<<<<<< Updated upstream
/* pxNum is the index of the central pixel for the convolution in the pImage array */
__device__ inline int32_t convolution_by_3(struct pixel *pImage, kernel_t kernel,
                                           uint32_t pxNum, uint32_t width, uint32_t height)
=======
__device__ inline int16_t convolution_by_3(unsigned char *pImage, kernel_t kernel,
                                           uint32_t baseIndex, uint32_t width, uint32_t height)
>>>>>>> Stashed changes
{
        int16_t grad = 0;
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



<<<<<<< Updated upstream

#if USE_SHARED_MEM
__global__ void sobel_unnorm_kernel(struct pixel *pInImageData, uint16_t *pOutImageData,
                                    uint32_t width, uint32_t height)
{
        extern __shared__ struct pixel locImgPart[];
=======
__global__ void sobel_unnorm_kernel(unsigned char *pInImageData, uint16_t *pOutImageData,
                                    uint32_t width, uint32_t height)
{
        /*extern __shared__ unsigned char localImagePart[];*/
>>>>>>> Stashed changes

        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        uint32_t basePxNum = blockIdx.y * width + blockIdx.x; /* First pixel of my block */
        uint32_t pxNum = y * width + x; /* My pixel */


        /* Copy the local block PLUS BORDER in local memory */
        uint32_t localBlockWidth = blockDim.x + 2; /* +2 for the borders */
        uint32_t localBlockHeight = blockDim.y + 2; /* +2 for the borders */


        /* be careful with borders */
        if (pxNum > width * height * 4) {
                return;
        }


        /*  
            -----------
            -----------       .......
            -- xxxxx --       .xxxxx.  
            -- xxxxx --       .xxxxx.  x: pixels of the block
            -- xxxxx --  -->  .xxxxx.  .: border necessary for convolution computation
            -- xxxxx --       .xxxxx.  -: surrounding original image
            -- xxxxx --       .xxxxx.
            -----------       .......
            -----------
            To do this, we have one thread by 'x'.
        */

        //XXX That's not optimal, but at least that understandeable
        //XXX Well, not understandable enough to be correct...

        if (threadIdx.x == 0) { /* Copy you line */

            uint32_t lineLocStart = (threadIdx.y + 1) * localBlockWidth; // +1 because of the border on top
            /*uint32_t lineImgStart = basePxNum [> start of block <]*/
                                  /*+ threadIdx.y * width [> down y lines <]*/
                                  /*- 1; [> back one column <]*/
            /* I'm the first of the line, so with border, line starts at me - 1 */
            uint32_t lineImgStart = pxNum - 1;
            for (int i = 0; i < localBlockWidth; i++) {
                locImgPart[lineLocStart + i] = pInImageData[lineImgStart + i];
            }

        } else if (threadIdx.x == 1 && threadIdx.y == 0) {  /* Copy the first line = top border */

            for (int i = 0; i < localBlockWidth; i++) {
                locImgPart[i] = pInImageData[basePxNum - width - 1 + i];
            }

        } else if (threadIdx.x == 1 && threadIdx.y == 1) { /* You will copy the last line = bottom border */

            uint32_t lineLocStart = (localBlockHeight - 1) * localBlockWidth;
            uint32_t lineImgStart = basePxNum /* start of block */
                                  + blockDim.y * width /* down blockDim.y lines */
                                  - 1; /* back one pixel */

            for (int i = 0; i < localBlockWidth; i++) {
                locImgPart[lineLocStart + i] = pInImageData[lineImgStart + i];
            }

        } /* The others will be idle for this part. */

        __syncthreads();


        //XXX: don't count the borders for the moment
        if (x == 0 || x == width || y == 0 || y == height) {
                pOutImageData[pxNum] = 0;
                return;
        }


        /* My position in the local partial copy */
        uint32_t baseIndex = (threadIdx.y + 1) * localBlockWidth + threadIdx.x + 1;
        int32_t gradX = convolution_by_3(locImgPart, kernelX, baseIndex, localBlockWidth, localBlockHeight);
        int32_t gradY = convolution_by_3(locImgPart, kernelY, baseIndex, localBlockWidth, localBlockHeight);

<<<<<<< Updated upstream
        float gradX_asFloat = static_cast<float>(gradX);
        float gradY_asFloat = static_cast<float>(gradY);
        uint16_t gradNorm = static_cast<uint32_t>(
                                sqrt(gradX_asFloat*gradX_asFloat + gradY_asFloat*gradY_asFloat));

        pOutImageData[pxNum] = gradNorm;
}

#else


__global__ void sobel_unnorm_kernel(struct pixel *pInImageData, uint16_t *pOutImageData,
                                    uint32_t width, uint32_t height)
{
        extern __shared__ struct pixel locImgPart[];

        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        /*uint32_t basePxNum = blockIdx.y * width + blockIdx.x; [> First pixel of my block <]*/
        uint32_t pxNum = y * width + x; /* My pixel */


        /*  -----------       .......
            -- xxxxx --       .xxxxx.  
            -- xxxxx --       .xxxxx.  x: pixels of the block
            -- xxxxx --  -->  .xxxxx.  .: border necessary for convolution computation
            -- xxxxx --       .xxxxx.  -: surrounding original image
            -- xxxxx --       .xxxxx.
            -----------       .......
            -----------
            To do this, we have one thread by 'x'.
        */

        int32_t gradX = convolution_by_3(pInImageData, kernelX, pxNum, width, height);
        int32_t gradY = convolution_by_3(pInImageData, kernelY, pxNum, width, height);

        float gradX_asFloat = static_cast<float>(gradX);
        float gradY_asFloat = static_cast<float>(gradY);
        uint16_t gradNorm = static_cast<uint32_t>(
=======
        int16_t gradX = convolution_by_3(pInImageData, kernelX, baseIndex, width, height);
        int16_t gradY = convolution_by_3(pInImageData, kernelY, baseIndex, width, height);

        float gradX_asFloat = static_cast<float>(gradX);
        float gradY_asFloat = static_cast<float>(gradY);
        uint16_t gradNorm = static_cast<uint16_t>(
>>>>>>> Stashed changes
                                sqrt(gradX_asFloat*gradX_asFloat + gradY_asFloat*gradY_asFloat));

        pOutImageData[pxNum] = gradNorm;
}
<<<<<<< Updated upstream


#endif
=======
>>>>>>> Stashed changes


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
        //XXX hard coded for the moment, should be more flexible

#if 1
        int maxBlockDim;
        int maxGridWidth;
        int maxGridHeight;
        //XXX renommer cette fonction
        maxNumThreads(&maxGridWidth, &maxGridHeight, &maxBlockDim);

        /* Try first to fill blocks, and then augment the number of blocks */
        dim3 threadsPerBlock(maxBlockDim, maxBlockDim);

        int gridWidth = width / maxBlockDim + (width % maxBlockDim == 0 ? 0 : 1);
        int gridHeight = height / maxBlockDim + (height % maxBlockDim == 0 ? 0 : 1);

        /* If that is too much threads, each thread will have to do more stuff */
        if (gridWidth * gridHeight > maxGridWidth) { /* maxGridWidth is also the max number of blocks (I think) */
            /* We will have to cut... But how do we do this ?? */
        }


        if (gridWidth > maxGridWidth || maxGridHeight > maxGridHeight || 


        dim3 nBlocks(gridWidth, gridHeight);

        /* The number of invoked worker threads, used by kernels to know if they have
           to treat several pixels */
        int numThreads = gridWidth * gridHeight * maxBlockDim * maxBlockDim;
        printf("%d x %d blocks of %d x %d threads each, for %d worker threads, and %d pixels\n",
                gridWidth, gridHeight, maxBlockDim, maxBlockDim, numThreads, width*height);


#else
        dim3 nBlocks(256, 256);


        /* Divide the image both horizontally and vertically */
        int blockWidth = width / 256 + (width % 256 == 0 ? 0 : 1);
        int blockHeight = height / 256 + (height % 256 == 0 ? 0 : 1);

        dim3 threadsPerBlock(blockWidth, blockHeight);
#endif

        //XXX for the moment, code tha happy case: there are enough threads
        //XXX to do one thread per pixel.

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
<<<<<<< Updated upstream
        // local block size: including a border
#if USE_SHARED_MEM
        uint32_t localMemSize = ((blockWidth + 2) * (blockHeight + 2)) * sizeof(struct pixel);
        sobel_unnorm_kernel <<< nBlocks, threadsPerBlock, localMemSize >>> (inImageDevice, outImageDevice, width, height);
#else
        sobel_unnorm_kernel <<< nBlocks, threadsPerBlock >>> (inImageDevice, outImageDevice, width, height);
#endif

=======
        // image block, PLUS the borders
        /*uint32_t localMemSize = (blockWidth + 1) * (blockHeight + 1);*/
        sobel_unnorm_kernel <<< nBlocks, threadsPerBlock/*, localMemSize*/ >>> (inImageDevice, outImageDevice, width, height);
>>>>>>> Stashed changes

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


<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
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




/* Compute the size of the biggest image that can be treated at once,
   and the corresponding size of the grid and blocks */
int maxNumThreads(int *pMaxGridWidth, int *pMaxGridHeight, int *pMaxBlockDim)
{
    cudaError_t err;
    struct cudaDeviceProp deviceProp;

    // We will only use one device, the first one
    err = cudaGetDeviceProperties(&deviceProp, 0);
    check_warn (err == cudaSuccess, "Failed to query device properties");

    int threadsPerBlock = deviceProp.maxThreadsPerBlock;
    //XXX Vérifier que le résultat est compatible avec ça
    /*int maxThreadsX = deviceProp.maxThreadsDim[0];*/
    /*int maxThreadsY = deviceProp.maxThreadsDim[1];*/

    /* It seems that the first dimension is also the max number of
       blocks that can be handeled */
    /*int maxBlocks = deviceProp.maxGridSize[0];*/

    /* We will try to find square grids and blocks from those dimensions.
       XXX It will be somehow suboptimal, all blocks won't be used.
       But all blocks are not executed at the same time anyway, so it's
       not necessarily a huge deal. Hence, we will find the biggest power
       of 2 such that it's <= sqrt(maxBlocks)*/

    /*int maxGridDim = maxSquareDim(maxBlocks);*/
    int maxBlockDim = maxSquareDim(threadsPerBlock);

    /**pMaxGridDim = maxGridDim;*/
    *pMaxGridWidth = deviceProp.maxGridSize[0];
    *pMaxGridHeight = deviceProp.maxGridSize[1];
    *pMaxBlockDim = maxBlockDim;

    return 0;

}



static inline int maxSquareDim(int numItems)
{
    /* Find the most significant 1 in numItems */
    /* There is probably some bitwise magic that would be faster */
    int lastOne = 0;
    for (int i = sizeof(int) * 8 - 1; i >= 0; i--) {
        if (numItems >> i != 0) {
            lastOne = i;
            break;
        }
    }

    int maxWidth_log2 = lastOne / 2;
    int maxWidth = 1 << maxWidth_log2;

    log_info("For %d, found %d x %d = %d\n", numItems, maxWidth, maxWidth, maxWidth*maxWidth);

    return maxWidth;
}






} /* extern "C" */
