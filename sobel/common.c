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
int decode_image(const char *srcFileName, struct image *pImage)
{
        check_null(pImage);
        check_null(srcFileName);
{

        unsigned int ret;
        ret = lodepng_decode32_file(&pImage->data, &pImage->width, &pImage->height, srcFileName);
        check (ret == 0, "Error while loading png file %s: %s\n", srcFileName, lodepng_error_text(ret));

        pImage->type = RGBA;

        return 0;
error:
        free_and_null(pImage->data);
        return -1;
}
}




/*
 * Encodes a image that is known to be in RGBA format by calling lodepng.
 * Will fail silently if the image is in the wrong format (should be checked upstream).
 */
static inline int encode_image_rgba(const char *destFileName, struct image *pImage)
{
        int ret = lodepng_encode32_file(destFileName, pImage->data,
                                        pImage->width, pImage->height);
        check (ret == 0, "Failed to write image to file %s", destFileName);
        
        return 0;
error:
        return -1;
}



/*
 * Encodes an image to a file by first transforming it from greyscale to RGBA.
 * Fails silently if the image is not in greyscale
 */
static inline int encode_image_greyscale(const char *destFileName, struct image *pImage)
{
        unsigned int ret;
        struct image RGBAImage = IMAGE_INITIALIZER;
        RGBAImage.type = RGBA;
        
        ret = greyScale_to_RGBA(pImage, &RGBAImage);
        check (ret == 0, "Failed to convert from Greyscale go RGBA");

        return encode_image_rgba(destFileName, &RGBAImage);
error:
        reset_matrix(&RGBAImage);
        return -1;
}





/*
 * Encode any king of image. See public header for full doc
 */
int encode_image(const char *destFileName, struct image *pImage)
{
        check_null(pImage);
        check_null(destFileName);
        check (pImage->type != Unknown, "The image must be either RGBA or GreyScale");

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




/* Simple extension from Grey values to R = G = B = grey, and A = 0
 * See header file for full documentation. */
int greyScale_to_RGBA(struct image *pGSImage, struct image *pRGBAImage)
{
        check (pGSImage->type == GreyScale,
                "The image to convert must be GreyScale, %s found", IMAGE_TYPE_STR(pRGBAImage->type));
        check_warn(pRGBAImage->data == NULL, "Will overwrite non-NULL ptr, potential leak");
{
        uint32_t width  = pGSImage->width;
        uint32_t height = pGSImage->height;
        
        pRGBAImage->width  = width;
        pRGBAImage->height = height;
        pRGBAImage->type   = GreyScale;
        pRGBAImage->data = calloc(pGSImage->width * pGSImage->height * 4, sizeof(int16_t));
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





/*
 * Takes the mean of R, G, B ad grey value. Alpha channel is ignored
 * See header file for full documentation
 */
int RGBA_to_greyScale(struct image *pRGBAImage, struct image *pGSImage)
{
        check (pRGBAImage->type == RGBA, "The image to convert must be RGBA, %s found",
                                         IMAGE_TYPE_STR(pRGBAImage->type));
        check_warn(pGSImage->data == NULL, "Will overwrite non-NULL ptr, potential leak");
{
        uint32_t R, G, B, greyVal;

        uint32_t width = pRGBAImage->width;
        uint32_t height = pRGBAImage->height;
        
        pGSImage->width  = width;
        pGSImage->height = height;
        pGSImage->type   = GreyScale;
        pGSImage->data   = calloc(pGSImage->width * pGSImage->height, sizeof(int16_t));
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





// This is a hack to allow testing, see usein tests/ subdir
#ifndef IN_TEST
int main(int argc, const char *argv[])
{
        int ret;

        if (argc != 3) {
                printf("Usage: %s inImage outImage\n", argv[0]);
                exit(1);
        }

        const char *inFileName = argv[1];
        const char *outFileName = argv[2];

        //XXX we could just check that we have rights on the files right at the beginning

        struct image inImage = IMAGE_INITIALIZER;
        struct image greyScaleImage = IMAGE_INITIALIZER;
        struct image outImage = IMAGE_INITIALIZER;

        ret = decode_image(inFileName, &inImage);
        check (ret == 0, "Image decoding failed");

        ret = RGBA_to_greyScale(&inImage, &greyScaleImage);
        check (ret == 0, "Failed to convert the image in greyscale");

        double startTime = omp_get_wtime();
        ret = sobel(&greyScaleImage, &outImage);
        double endTime = omp_get_wtime();
        check (ret == 0, "Sobel edge detection failed");

        ret = encode_image(outFileName, &outImage);
        check (ret == 0, "Error while storing image to disk");

        printf("Interresting stuff finished in %lf\n", endTime - startTime);

        return 0;

error:
        return 1;
}
#endif
