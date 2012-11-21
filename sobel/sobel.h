#ifndef __SOBEL_H__
#define __SOBEL_H__

#include <stdio.h>


struct matrix {
        int width;
        int height;
        double **data;
};


#define MATRIX_INITIALIZER {0, 0, NULL}



/* Those functions will be common to any implementations */
int decode_image(FILE *srcFile, struct matrix *pImage);
int encode_image(FILE *destFile, struct matrix *pImage);

/* Those functions will be implementation-dependant */
int convolution(struct matrix *pInImage, struct matrix *pKernel, struct matrix *pOutMatrix);
int gradient(struct matrix *pInMatrixX, struct matrix *pInMatrixY, struct matrix *pOutImage);



#endif /* end of include guard: __SOBEL_H__ */
