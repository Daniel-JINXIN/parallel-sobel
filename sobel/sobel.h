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


enum matrixType { GreyScale, RGBA, Irrelevant };

#define MATRIX_TYPE_STR(matrixType) \
        (matrixType) == GreyScale ? "Greyscale" : \
        (matrixType) == RGBA ? "RGBA" : \
        (matrixType) == Irrelevant ? "Irrelevant" : \
        "Unknown type"


struct matrix {
        uint32_t width;
        uint32_t height;
        unsigned char *data;
        enum matrixType type;
};

#define MATRIX_INITIALIZER {0, 0, NULL, Irrelevant}

#define reset_matrix(mat) \
do { \
        (mat)->width = 0; \
        (mat)->height = 0; \
        (mat)->type = Irrelevant; \
        free_and_null((mat)->data); \
} while (0)



/* Those functions will be common to any implementations */
int decode_image(const char *srcFileName, struct matrix *pImage);
int encode_image(const char *destFileName, struct matrix *pImage);

/* Those functions will be implementation-dependant */
int convolution(struct matrix *pInImage, struct matrix *pKernel, struct matrix *pOutMatrix);
int gradient(struct matrix *pInMatrixX, struct matrix *pInMatrixY, struct matrix *pOutImage);

/* Convert back and forth between the PNG RGBA coding and a simple greyscale,
 * so it is a lot more compact in memory */
int greyScale_to_RGBA(struct matrix *pGSImage, struct matrix *pRGBAImage);
int RGBA_to_greyScale(struct matrix *pRGBAImage, struct matrix *pGSImage);


#endif /* end of include guard: __SOBEL_H__ */
