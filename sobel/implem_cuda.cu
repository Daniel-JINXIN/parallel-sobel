#include <cuda.h>
#include <omp.h>

/* CUDA is implicitly C++ code, this is needed so out symbols are not mangled */
extern "C" {

#include "dbg.h"
#include "sobel.h"


__constant__ kernel_t kernelX = { {-1, 0, 1}, 
                                  {-2, 0, 2},
                                  {-1, 0, 1} };

__constant__ kernel_t kernelY = { {-1, -2, -1},
                                  { 0,  0,  0},
                                  { 1,  2,  1} };


__device__ inline unsigned char greyscale_value(unsigned char *pImage, uint32_t baseIndex)
{
        /* Get the GreyScale value of our pixel */
        unsigned char R, G, B, greyVal;
        R = pImage[baseIndex];
        G = pImage[baseIndex + 1];
        B = pImage[baseIndex + 2];

        greyVal = (R + G + B) / 3;

        return greyVal;
}




//XXX We do redundant computation between threads, can be improved with shared memory
__device__ inline int32_t convolution_by_3(unsigned char *pImage, kernel_t kernel,
                                           uint32_t baseIndex, uint32_t width, uint32_t height)
{
        int32_t grad = 0;
        /* Line below */
        grad += kernelX[0][0] * greyscale_value(pImage, baseIndex + 4 * width + 4);
        grad += kernelX[0][1] * greyscale_value(pImage, baseIndex + 4 * width);
        grad += kernelX[0][2] * greyscale_value(pImage, baseIndex + 4 * width - 4);

        /* current line */
        grad += kernelX[1][0] * greyscale_value(pImage, baseIndex + 4);
        grad += kernelX[1][1] * greyscale_value(pImage, baseIndex);
        grad += kernelX[1][2] * greyscale_value(pImage, baseIndex - 4);

        /* line above */
        grad += kernelX[2][0] * greyscale_value(pImage, baseIndex - 4 * width + 4);
        grad += kernelX[2][1] * greyscale_value(pImage, baseIndex - 4 * width);
        grad += kernelX[2][2] * greyscale_value(pImage, baseIndex - 4 * width - 4);

        return grad;
}



__global__ void sobel_unnorm_kernel(unsigned char *pInImageData, uint32_t *pOutImageData,
                                    uint32_t width, uint32_t height)
{

        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        uint32_t pxNum = y * width + x;
        uint32_t baseIndex = 4 * pxNum; /* 4 char by pixel */


        //XXX: don't count the borders for the moment
        if (x == 0 || x == width || y == 0 || y == height) {
                pOutImageData[pxNum] = 0;
                return;
        }

        /* e careful with borders */
        if (baseIndex > width * height * 4) {
                return;
        }

        /* the surrounding pixels, where x == me.
                . . .
                . x .
                . . .
         */

        int32_t gradX = convolution_by_3(pInImageData, kernelX, baseIndex, width, height);
        int32_t gradY = convolution_by_3(pInImageData, kernelY, baseIndex, width, height);

        double gradX_asFloat = static_cast<float>(gradX);
        double gradY_asFloat = static_cast<float>(gradY);
        uint32_t gradNorm = static_cast<uint32_t>(
                                sqrt(gradX_asFloat*gradX_asFloat + gradY_asFloat*gradY_asFloat));

        pOutImageData[pxNum] = gradNorm;
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
        //XXX hard coded for the moment, should be more flexible
        dim3 nBlocks(256, 256);

        /* Divide the image both horizontally and vertically */
        int blockWidth = width / 256 + (width % 256 == 0 ? 0 : 1);
        int blockHeight = height / 256 + (height % 256 == 0 ? 0 : 1);

        printf("%d, %d\n", blockWidth, blockHeight);

        dim3 threadsPerBlock(blockWidth, blockHeight);

        //XXX for the moment, code tha happy case: there are enough threads
        //XXX to do one thread per pixel.

        unsigned char *inImageDevice;
        /*unsigned char *outImageDevice;*/
        uint32_t *outImageDevice;

        /* Allocate memory on the device for both images (in and out) */
        ret = cudaMalloc((void **)&inImageDevice, width * height * 4 * sizeof(unsigned char));
        check_warn (ret == cudaSuccess, "Failed to allocate memory for in image on the device");

        ret = cudaMalloc((void **)&outImageDevice, width * height * 4 * sizeof(uint32_t));
        check_warn (ret == cudaSuccess, "Failed to allocate memory for out image on the device");

        /* Copy The input image on the device */
        ret = cudaMemcpy(inImageDevice, pInImage->data, width * height * 4 * sizeof(unsigned char),
                         cudaMemcpyHostToDevice);
        check_warn (ret == cudaSuccess, "Failed to copy the image to the device");


        /* Unnormalized version */
        uint32_t *pUnNormalizedOut = NULL;
        pUnNormalizedOut = (uint32_t *)calloc(width*height, sizeof(uint32_t));

        /* Allocate memory for the resulting image */
        pOutImage->width = pInImage->width;
        pOutImage->height = pInImage->height;
        pOutImage->type = RGBA;
        pOutImage->data = (unsigned char*)calloc(pOutImage->width * pOutImage->height * 4, sizeof(unsigned char));
        /*check_mem(pOutImage->data);*/

        /* And launch the kernel */
        sobel_unnorm_kernel <<< nBlocks, threadsPerBlock >>> (inImageDevice, outImageDevice, width, height);

        ret = cudaMemcpy(pUnNormalizedOut, outImageDevice,
                         width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        check_warn (ret == cudaSuccess, "Kernel failed: %s", cudaGetErrorString(ret));


        /* Normalize, on CPU for the moment */
        uint32_t max = 0;
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
