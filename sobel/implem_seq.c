#include <stdlib.h>
#include <math.h>

#include "dbg.h"

#include "sobel.h"




/* Compute one value by convoluting a 3x3 kernel over a 3x3 portion of the image
 *
 * in: pInImage         Pointer to the image to convolute
 * in: pKernel          Pointer to the kernel to apply
 * in: row, col         Coordinates of the point to get by convolution
 * out: pPixel          Pointer to the pixel to write
 * 
 *
 *
 * Formula for the convolution of a 3x3 kernel f with an image I :
 * forall (i, j)
 *      y(i, j) = sum(k = -1..1) { sum(l = -1..1) { f(k, l) * I(i - k, j - l) } }
 *
 * As in C, arrays start to 0, we reindex those loops:
 *      y(i, j) = sum_(k = 0..2) { sum_(l = 0..2) { f(k-1, l-1) * I(i - k + 1, j - l + 1) } }
 *
 * As we will only convolute by a 3x3 kernel, we unroll this directly, it will simplify
 * border-conditions checking, and could play nicely with the optimizer.
 *
 * Unrolled version:
 *      y(i, j) =   f(0, 0) * I(i+1 j+1)
 *                + f(0, 1) * I(i+1, j)
 *                + f(0, 2) * I(i+1, j-1)
 *
 *                + f(1, 0) * I(i, j+1)
 *                + f(1, 1) * I(i, j)
 *                + f(1, 2) * I(i, j-1)
 *
 *                + f(2, 0) * I(i-1, j+1)
 *                + f(2, 1) * I(i-1, j)
 *                + f(2, 2) * I(i-1, j-1)
 *
 * We see from the above formula that there is a problem for i = 0, j = 0, i = M, j = N
 * where the image is of size MxN. In such a condition, this function will fail. The edge
 * conditions must be handeled in the calling code.
 */
static inline void convolution_3_by_3(struct image *const pInImage, kernel_t kernel,
                                      uint32_t row, uint32_t col,
                                      int16_t *restrict pPixel)
{
        uint32_t w = pInImage->width;
        int16_t acc = 0;
        acc += kernel[0][0] * pInImage->data[(row + 1)*w + col + 1];
        acc += kernel[0][1] * pInImage->data[(row + 1)*w + col];
        acc += kernel[0][2] * pInImage->data[(row + 1)*w + col - 1];

        acc += kernel[1][0] * pInImage->data[row*w + col + 1];
        acc += kernel[1][1] * pInImage->data[row*w + col];
        acc += kernel[1][2] * pInImage->data[row*w + col - 1];

        acc += kernel[2][0] * pInImage->data[(row - 1)*w + col + 1];
        acc += kernel[2][1] * pInImage->data[(row - 1)*w + col];
        acc += kernel[2][2] * pInImage->data[(row - 1)*w + col - 1];

        *pPixel = acc;
}


/*
 * XXX on pourrait aussi étendre virtuellement la matrice de base avec des 0 sur
 * XXX les bords. Ca devrait marcher, et c'est plus propre. Peut-être pas le plus
 * XXX important pour l'instant.
 ***** Edge conditions *****
 * We see from the above formula that there is a problem for i = 0, j = 0, i = M, j = N
 * where the image is of size MxN. We will simply ignore them while computing the
 * convolution, and then copy row 1 n row 0, col 1 in col 0, col M-1 in col M, and
 * row N-1 in row N.
 * For instance, if the following matrix resulted from the convolution (where ? represent
 * values that could not be computed):
 *
 *    ? ? ? ? ?                                           1 1 2 3 3
 *    ? 1 2 3 ?                                           1 1 2 3 3
 *    ? 4 5 6 ?   it would be artificially extented to    4 4 5 6 6
 *    ? 7 8 9 ?                                           7 7 8 9 9
 *    ? ? ? ? ?                                           7 7 8 9 9
 *                                                      
 */
int convolution3(struct image *pInImage, kernel_t kernel, struct matrix *pOutMatrix)
{
        check_null(pInImage);
        check_null(pOutMatrix);
        check_warn(pOutMatrix->data == NULL, "Overwrite non-null pointer possible leak");
{


        /* First, allocate memory for the outMatrix, and set its features */
        pOutMatrix->width = pInImage->width;
        pOutMatrix->height = pInImage->height;
        pOutMatrix->data = calloc(pInImage->width * pInImage->height, sizeof(int16_t));
        check_mem(pOutMatrix->data);


        /* Make the convolution where it's possible */
        for (uint32_t row = 1; row < pInImage->height - 1; row++) {
                for (uint32_t col = 1; col < pInImage->width - 1; col++) {
                        convolution_3_by_3(pInImage, kernel, row, col,
                                           &pOutMatrix->data[row*pOutMatrix->width + col]);
                }
        }

        /* Fill the missing rows with what we arbitrarily decided. Be careful,
         * corners cannot be filled yet, so from 1 to width - 1. */
        for (uint32_t col = 1; col < pOutMatrix->width - 1; col++) {
                pOutMatrix->data[col] = pOutMatrix->data[pOutMatrix->width + 1];

                uint32_t startBeforeLastRow = pOutMatrix->width * (pOutMatrix->height - 2);
                uint32_t startLastRow = pOutMatrix->width * (pOutMatrix->height - 1);
                pOutMatrix->data[startLastRow + col] = pOutMatrix->data[startBeforeLastRow + col];
        }

        /* Now first and last columns, including corners. Iterate row by row. */
        for (uint32_t startRow = 0; startRow < pOutMatrix->height; startRow += pOutMatrix->width) {
                pOutMatrix->data[startRow] = pOutMatrix->data[startRow + 1];

                pOutMatrix->data[startRow + pOutMatrix->width - 1] =
                        pOutMatrix->data[startRow + pOutMatrix->width - 2];
        }


        return 0;
error:
        reset_matrix(pOutMatrix);
        return -1;
}
}


/* Return the norm2 of a vector */
static inline int16_t norm2(int16_t x, int16_t y)
{
        return sqrt(x*x + y*y);
}




static inline void normalize_matrix_to_image(struct matrix *pMat, struct image *pImg)
{
        int16_t max = ~0; // min value
        for (uint32_t px = 0; px < pMat->width * pMat->height; px++) {
                if (pMat->data[px] > max) {
                        max = pMat->data[px];
                }
        }

        for (uint32_t px = 0; px < pMat->width * pMat->height; px++) {
                pImg->data[px] = (unsigned char) ((pMat->data[px] * 255) / max);
        }
}




int gradient(struct matrix *pInMatrixX, struct matrix *pInMatrixY, struct image *pOutImage)
{
        check_null(pInMatrixX);
        check_null(pInMatrixY);
        check_null(pOutImage);
        check(pOutImage->data == NULL, "Overwriting non-null pointer, possible leak");
        check (pInMatrixX->width == pInMatrixY->width && pInMatrixX->height == pInMatrixY->height,
                        "Both matrix must have same dimensions, found (%d, %d) and (%d, %d)",
                        pInMatrixX->width, pInMatrixX->height, pInMatrixY->width, pInMatrixY->height);
{
        pOutImage->width = pInMatrixX->width;
        pOutImage->height = pInMatrixX->height;
        pOutImage->type = GreyScale;
        pOutImage->data = calloc(pOutImage->width * pOutImage->height, sizeof(unsigned char));
        check_mem(pOutImage->data);

        /* The returned norm of the gradient might be far bigger than 255. Hence, we keep more
         * precision on the result, and later normalize from 0 to 255 */
        struct matrix unNormalizedGradient = MATRIX_INITIALIZER;
        unNormalizedGradient.width = pOutImage->width;
        unNormalizedGradient.height = pOutImage->height;
        unNormalizedGradient.data = calloc(pOutImage->width * pOutImage->height, sizeof(int16_t));
        check_mem(unNormalizedGradient.data);

        for (uint32_t px = 0; px < pOutImage->width * pOutImage->height; px++) {
                unNormalizedGradient.data[px] = norm2(pInMatrixX->data[px], pInMatrixY->data[px]);
        }

        normalize_matrix_to_image(&unNormalizedGradient, pOutImage);

        return 0;
error:
        reset_matrix(pOutImage);
        return -1;
}
}
