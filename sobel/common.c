#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

#include "dbg.h"
#include "sobel.h"
#include "lodepng.h"



/* We make use of the awesome lodepng library, see in third_party/ */
int decode_image(const char *srcFileName, struct matrix *pImage)
{
        check_null(pImage);
        check_null(srcFileName);

        unsigned int ret;
        ret = lodepng_decode32_file(&pImage->data, &pImage->width, &pImage->height, srcFileName);
        check (ret == 0, "Error while loading png file %s: %s\n", srcFileName, lodepng_error_text(ret));

        pImage->type = RGBA;

        return 0;
error:
        free_and_null(pImage->data);
        return -1;
}




static inline int encode_image_rgba(const char *destFileName, struct matrix *pImage)
{
        int ret = lodepng_encode32_file(destFileName, pImage->data,
                                    pImage->width, pImage->height);
        check (ret == 0, "Failed to write image to file %s", destFileName);
        
        return 0;
error:
        return -1;
}



static inline int encode_image_greyscale(const char *destFileName, struct matrix *pImage)
{
        unsigned int ret;
        struct matrix RGBAImage = MATRIX_INITIALIZER;
        RGBAImage.type = RGBA;
        
        ret = greyScale_to_RGBA(pImage, &RGBAImage);
        check (ret == 0, "Failed to convert from Greyscale go RGBA");

        return encode_image_rgba(destFileName, &RGBAImage);
error:
        reset_matrix(&RGBAImage);
        return -1;
}





int encode_image(const char *destFileName, struct matrix *pImage)
{
        check_null(pImage);
        check_null(destFileName);
        check (pImage->type != Irrelevant, "The image must be either RGBA or GreyScale");

        switch (pImage->type) {
                case RGBA:
                        return encode_image_rgba(destFileName, pImage);
                        break;
                case GreyScale:
                        return encode_image_greyscale(destFileName, pImage);
                        break;
                default:
                        log_err("Default case should not be reached");
                        goto error;
        }

error:
        return -1;
}



static inline int initKernelX(struct matrix *kernelX)
{
        kernelX->data = calloc(3 * 3, sizeof(char));
        check_mem (kernelX->data);

        kernelX->width  = 3;
        kernelX->height = 3;

        kernelX->data[0] = -1;
        kernelX->data[1] = 0;
        kernelX->data[2] = 1;

        kernelX->data[3] = -2;
        kernelX->data[4] = 0;
        kernelX->data[5] = 2;

        kernelX->data[6] = -1;
        kernelX->data[7] = 0;
        kernelX->data[8] = 1;

        return 0;
error:
        return 1;
}


static inline int initKernelY(struct matrix *kernelY)
{
        kernelY->data = calloc(3 * 3, sizeof(char));
        check_mem (kernelY->data);

        kernelY->width  = 3;
        kernelY->height = 3;

        kernelY->data[0] = -1;
        kernelY->data[1] = -2;
        kernelY->data[2] = -1;

        kernelY->data[3] = 0;
        kernelY->data[4] = 0;
        kernelY->data[5] = 0;

        kernelY->data[6] = 1;
        kernelY->data[7] = 2;
        kernelY->data[8] = 1;

        return 0;
error:
        return 1;
}




/* Simple extension from Grey values to R = G = B = grey, and A = 0 */
int greyScale_to_RGBA(struct matrix *pGSImage, struct matrix *pRGBAImage)
{
        check (pGSImage->type == GreyScale,
                "The image to convert must be GreyScale, %s found", MATRIX_TYPE_STR(pRGBAImage->type));
        check_warn(pRGBAImage->data == NULL, "Will overwrite non-NULL ptr, potential leak");
{
        uint32_t width  = pGSImage->width;
        uint32_t height = pGSImage->height;
        
        pRGBAImage->width  = width;
        pRGBAImage->height = height;
        pRGBAImage->type   = GreyScale;
        pRGBAImage->data = calloc(pGSImage->width * pGSImage->height * 4, sizeof(unsigned char));
        check_mem(pRGBAImage->data);
        
        for (uint32_t i = 0; i < width * height; i++) {
                uint32_t greyVal = pGSImage->data[i];
                pRGBAImage->data[4*i]     = greyVal;
                pRGBAImage->data[4*i + 1] = greyVal;
                pRGBAImage->data[4*i + 2] = greyVal;
                pRGBAImage->data[4*i + 3] = 255; /* fully opaque */
        }

        return 0;
error:
        reset_matrix(pRGBAImage);
        return -1;
}
}





/* Takes the mean of R, G, B ad grey value. Alpha channel is ignored */
int RGBA_to_greyScale(struct matrix *pRGBAImage, struct matrix *pGSImage)
{
        check (pRGBAImage->type == RGBA, "The image to convert must be RGBA, %s found",
                                         MATRIX_TYPE_STR(pRGBAImage->type));
        check_warn(pGSImage->data == NULL, "Will overwrite non-NULL ptr, potential leak");
{
        uint32_t R, G, B, greyVal;

        uint32_t width = pRGBAImage->width;
        uint32_t height = pRGBAImage->height;
        
        pGSImage->width  = width;
        pGSImage->height = height;
        pGSImage->type   = GreyScale;
        pGSImage->data = calloc(pGSImage->width * pGSImage->height, sizeof(unsigned char));
        check_mem(pGSImage->data);

        for (uint32_t i = 0; i < width * height; i++) {
                /* The RGBA image has 4 bytes by pixel */
                R = pRGBAImage->data[4 * i];
                G = pRGBAImage->data[4 * i + 1];
                B = pRGBAImage->data[4 * i + 2];
                greyVal = (R + G + B) / 3;

                pGSImage->data[i] = (unsigned char)greyVal;
        }

        return 0;
error:
        reset_matrix(pGSImage);
        return -1;
}
}





#if 0
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
#endif






int main(int argc, const char *argv[])
{
        int ret;
        if (argc < 3) {
                printf("Usage: %s inFileName outFileName\n", argv[0]);
                exit(1);
        }

        const char *inFileName  = argv[1];
        const char *outFileName = argv[2];

        struct matrix baseImage      = MATRIX_INITIALIZER;
        struct matrix greyScaleImage = MATRIX_INITIALIZER;

        ret = decode_image(inFileName, &baseImage);
        check (ret == 0, "Failed to decode image");

        ret = RGBA_to_greyScale(&baseImage, &greyScaleImage);
        check (ret == 0, "Failed to convert to grey scale");
        
        ret = encode_image(outFileName, &greyScaleImage);
        check (ret == 0, "Failed to write image");

        return 0;

error:
        free_and_null(baseImage.data);
        free_and_null(greyScaleImage.data);
        return -1;
}
