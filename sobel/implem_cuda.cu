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



__global__ void sobel_kernel(unsigned char *pInImageData, unsigned char *pOutImageData,
                             uint32_t width, uint32_t height)
{
        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        uint32_t baseIndex = y * width * 4 + x * 4; /* 4 char by pixel */

        /* e careful with borders */
        if (baseIndex > width * height * 4) {
                return;
        }

        unsigned char R, G, B, greyVal;
        R = pInImageData[baseIndex];
        G = pInImageData[baseIndex + 1];
        B = pInImageData[baseIndex + 2];

        greyVal = (R + G + B) / 3;

        pOutImageData[baseIndex] = greyVal;
        pOutImageData[baseIndex + 1] = greyVal;
        pOutImageData[baseIndex + 2] = greyVal;
        pOutImageData[baseIndex + 3] = 255; /* Full opacity */
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
        unsigned char *outImageDevice;

        /* Allocate memory on the device for both images (in and out) */
        ret = cudaMalloc((void **)&inImageDevice, width * height * 4 * sizeof(unsigned char));
        check_warn (ret == cudaSuccess, "Failed to allocate memory for in image on the device");

        ret = cudaMalloc((void **)&outImageDevice, width * height * 4 * sizeof(unsigned char));
        check_warn (ret == cudaSuccess, "Failed to allocate memory for out image on the device");

        /* Copy The input image on the device */
        ret = cudaMemcpy(inImageDevice, pInImage->data, width * height * 4 * sizeof(unsigned char),
                         cudaMemcpyHostToDevice);
        check_warn (ret == cudaSuccess, "Failed to copy the image to the device");


        /* Allocate memory for the resulting image */
        pOutImage->width = pInImage->width;
        pOutImage->height = pInImage->height;
        pOutImage->type = RGBA;
        pOutImage->data = (unsigned char*)calloc(pOutImage->width * pOutImage->height * 4, sizeof(unsigned char));
        check_mem(pOutImage->data);

        /* And launch the kernel */
        sobel_kernel <<< nBlocks, threadsPerBlock >>> (inImageDevice, outImageDevice, width, height);

        ret = cudaMemcpy(pOutImage->data, outImageDevice, width * height * 4 * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost);
        check_warn (ret == cudaSuccess, "Kernel failed: %s", cudaGetErrorString(ret));

        /* display some */
        for (int i = 0; i < 20; i++) {
                printf("%d ", pOutImage->data[i]);
        }
        puts("");

        return 0;
error:
        cudaFree(inImageDevice);
        cudaFree(outImageDevice);
        free_and_null(pOutImage->data);
        return -1;
}



} /* extern "C" */
