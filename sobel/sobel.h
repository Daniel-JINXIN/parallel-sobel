#ifndef __SOBEL_H__
#define __SOBEL_H__

#include <stdio.h>
#include <stdint.h>


#define free_and_null(ptr) \
do { \
        if (ptr != NULL) { \
                free(ptr); \
        } \
        ptr = NULL;\
} while (0)


enum imageType { GreyScale, RGBA, Unknown};

#define IMAGE_TYPE_STR(matrixType) \
        (matrixType) == GreyScale ? "Greyscale" : \
        (matrixType) == RGBA ? "RGBA" : \
        (matrixType) == Unknown ? "Unknown" : \
        "Unknown type"


struct image {
        uint32_t width;
        uint32_t height;
        unsigned char *data;
        enum imageType type;
};


struct matrix {
        uint32_t width;
        uint32_t height;
        int16_t *data;
};


#define KERNEL_DIM 3
typedef signed char kernel_t[KERNEL_DIM][KERNEL_DIM] ;


#define MATRIX_INITIALIZER {0, 0, NULL}
#define IMAGE_INITIALIZER  {0, 0, NULL, Unknown}

#define reset_matrix(mat) \
do { \
        (mat)->width = 0; \
        (mat)->height = 0; \
        free_and_null((mat)->data); \
} while (0)


#define reset_image(mat) \
do { \
        (mat)->width = 0; \
        (mat)->height = 0; \
        (mat)->type = Unknown \
        free_and_null((mat)->data); \
} while (0)


/* Those functions will be common to any implementations */
int decode_image(const char *srcFileName, struct image *pImage);
int encode_image(const char *destFileName, struct image *pImage);

/* Those functions will be implementation-dependant */
int convolution3(struct image *pInImage, kernel_t kernel, struct matrix *pOutMatrix);
int gradient(struct matrix *pInMatrixX, struct matrix *pInMatrixY, struct image *pOutImage);

/* Convert back and forth between the PNG RGBA coding and a simple greyscale,
 * so it is a lot more compact in memory */
int greyScale_to_RGBA(struct image *pGSImage, struct image *pRGBAImage);
int RGBA_to_greyScale(struct image *pRGBAImage, struct image *pGSImage);



#define print_mat(pMat) \
do { \
        printf("Image width: %d, height: %d\n", (pMat)->width, (pMat)->height); \
        for (uint32_t i = 0; i < (pMat)->height; i++) {  \
                for (uint32_t j = 0; j < (pMat)->width; j++) { \
                        printf("%d ", (pMat)->data[i*(pMat)->width + j]); \
                } \
                puts(""); \
        } \
} while (0)


#endif /* end of include guard: __SOBEL_H__ */
