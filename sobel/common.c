#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <omp.h>

#include "dbg.h"
#include "sobel.h"



int decode_image(FILE *srcFile, struct matrix *pImage)
{
        (void) srcFile;
        (void) pImage;
        return 0;
}

int encode_image(FILE *destFile, struct matrix *pImage)
{
        (void) destFile;
        (void) pImage;
        return 0;
}



static inline int initKernelX(struct matrix *kernelX)
{
        kernelX->data = calloc(3, sizeof(double *));
        check (kernelX->data != NULL, "No more memory");

        for (int i = 0; i < 3; i++) {
                kernelX->data[i] = calloc(3, sizeof(double));
                check (kernelX->data != NULL, "No more memory");
        }

        kernelX->width  = 3;
        kernelX->height = 3;

        kernelX->data[0][0] = -1;
        kernelX->data[0][1] = 0;
        kernelX->data[0][2] = 1;
        kernelX->data[1][0] = -2;
        kernelX->data[1][1] = 0;
        kernelX->data[1][2] = 2;
        kernelX->data[2][0] = -1;
        kernelX->data[2][1] = 0;
        kernelX->data[2][2] = 1;

        return 0;
error:
        return 1;
}


static inline int initKernelY(struct matrix *kernelY)
{
        kernelY->data = calloc(3, sizeof(double *));
        check (kernelY->data != NULL, "No more memory");

        for (int i = 0; i < 3; i++) {
                kernelY->data[i] = calloc(3, sizeof(double));
                check (kernelY->data != NULL, "No more memory");
        }

        kernelY->width  = 3;
        kernelY->height = 3;

        kernelY->data[0][0] = -1;
        kernelY->data[0][1] = -2;
        kernelY->data[0][2] = -1;
        kernelY->data[1][0] = 0;
        kernelY->data[1][1] = 0;
        kernelY->data[1][2] = 0;
        kernelY->data[2][0] = 1;
        kernelY->data[2][1] = 2;
        kernelY->data[2][2] = 1;

        return 0;
error:
        return 1;
}





int main(int argc, const char *argv[])
{
        int ret;

        if (argc != 3) {
                printf("Usage: %s inImage outImage\n", argv[0]);
                exit(1);
        }

        const char *inFileName = argv[1];
        const char *outFileName = argv[2];
        FILE *inFile = NULL;
        FILE *outFile = NULL;

        inFile = fopen(inFileName, "r");
        check (inFile != NULL, "Error opening %s: %s\n", inFileName, strerror(errno));

        outFile = fopen(outFileName, "w");
        check (outFile != NULL, "Error opening %s: %s\n", outFileName, strerror(errno));

        struct matrix kernelX = MATRIX_INITIALIZER;
        struct matrix kernelY = MATRIX_INITIALIZER;
        ret = initKernelX(&kernelX);
        check (ret == 0, "Failed to create kernelX");

        ret = initKernelY(&kernelY);
        check (ret == 0, "Failed to create kernelY");

        struct matrix inImage = MATRIX_INITIALIZER;
        struct matrix outImage = MATRIX_INITIALIZER;
        struct matrix matX = MATRIX_INITIALIZER;
        struct matrix matY = MATRIX_INITIALIZER;

        ret = decode_image(inFile, &inImage);
        check (ret == 0, "Image decoding failed");

        double startTime = omp_get_wtime();
        ret = convolution(&inImage, &kernelX, &matX);
        check (ret == 0, "Convolution with kernel X failed");

        ret = convolution(&inImage, &kernelY, &matY);
        check (ret == 0, "Convolution with kernel Y failed");

        ret = gradient(&matX, &matY, &outImage);
        check (ret == 0, "Computation of gradient failed");
        double endTime = omp_get_wtime();

        ret = encode_image(outFile, &outImage);
        check (ret == 0, "Error while storing image to disk");

        printf("Interresting stuff finished in %lf\n", endTime - startTime);

        return 0;

error:
        return 1;
}
