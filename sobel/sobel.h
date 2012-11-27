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


/*
 * This structure reprents an image, either in RGBA or in GreyScale.
 */
struct image {
        uint32_t width; /* Width of the picture in pixels */
        uint32_t height; /* Height of the picture in pixels */
        unsigned char *data; /* Actual data, of size 4*width*height for a RGBA image
                                and width*height for a GreyScale image.
                                Accessed in row-major, that is data[row*width + col] */
        enum imageType type; /* Type of the image, RGBA or GreyScale. Used to know how
                                to manipulate the image */
};


/*
 * This struct reprensents a matrix that's not necessarily an image
 */
struct matrix {
        uint32_t width; /* Width in elements */
        uint32_t height; /* Height in elements */
        int16_t *data; /* Actual data, of width*height elements */
};


/*
 * Type alias for 3x3 convolution kernels
 */
#define KERNEL_DIM 3
typedef signed char kernel_t[KERNEL_DIM][KERNEL_DIM] ;


/*
 * The following macros are helpers to manipulate matrices and images
 */
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


/* Dumps a matrix or image on the standard output. Used for debug pusposes */
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



/*************************************************************************
 *      Those functions will be common to any implementations
 *              and are hence defined in common.c
 *************************************************************************/


/*
 * Takes the name of a PNG file and builds an RGBA image from it,
 * using the lodepng library.
 *
 * in: srcFileName      Name of the PNG file
 * out: pImage          The ARGB image obtained from the PNG file
 *
 * return: 0 in case of success, -1 if some failure occured
 *
 * NOTE: the pImage structure must be clean when entering and is
 *       given back clean if an error occurs.
 */
int decode_image(const char *srcFileName, struct image *pImage);


/*
 * Takes an image and writes it to a new PNG file using lodepng
 *
 * in: destFileName     Name of the PNG file that will be created
 * in: pImage           Pointer to an image struct, that can be either
 *                      RGBA or GreyScale.
 * return: 0 if successful, -1 otherwise
 */
int encode_image(const char *destFileName, struct image *pImage);


/* Convert back and forth between the PNG RGBA coding and a simple greyscale,
 * so it is a lot more compact in memory */
/*
 * Converts a GreyScale image to a RGBA image. Simply makes R = G = B = grey_value,
 * and A = 255 (full opacity).
 *
 * in: pGSImage        The greyscale image to convert
 * out: pRGBAImage     The resulting RGBA image
 *
 * returns: 0 if successful, -1 otherwise
 *
 * NOTE: pRGBAImage must come clean, and will be given back clean in case of failure
 */
int greyScale_to_RGBA(struct image *pGSImage, struct image *pRGBAImage);


/*
 * Converts an RGBA image to a greyscale image by averaging the values of
 * the three colours, and ignoring the alpha channel
 *
 * in: pRGBAImage       The RGBA image to convert
 * out: pGSImage        The resulting greyscale image
 *
 * returns: 0 if successful, -1 otherwise
 *
 * NOTE: pGSImage must come clean, and will be given back clean in case of failure
 */
int RGBA_to_greyScale(struct image *pRGBAImage, struct image *pGSImage);







/*****************************************************************************
 *      Those functions will be implementation-dependant, and hence
 *      are implemented in implem_*.c. These are the ones to be timed.
 *****************************************************************************/


/*
 * Convolutes a matrix with a 3 by 3 kernel.
 * XXX: va peut-être évoluer, car les convolutions PARTICULIÈRES que l'on
 * utilise sont séparables, et donc encore plus parallèlisables, en particulier
 * intéressant sur GPU.
 * Pourrait donc être séparé en deux fonctions getGradX() et getGradY(), dont
 * l'implémentation pourrait être totalement indépendante.
 */
int convolution3(struct image *pInImage, kernel_t kernel, struct matrix *pOutMatrix);


/*
 * This function combines the gradients in directions X and Y and output the
 * norm of the gradient on every point of the image.
 *
 * in: pInMatrixX       Matrix containing the X component of the gradient
 * in: pInMatrixY       Matrix containing the Y component of the gradient
 * out: pOutImage       GreyScale image where each point contains the norm
 *                      of the gradient, normalized on a 0-255 scale.
 *
 * returns: 0 if successful, -1 otherwise.
 *
 * NOTE: pOutImage must come clean, and will be given back clean in case of failure
 */
int gradient(struct matrix *pInMatrixX, struct matrix *pInMatrixY, struct image *pOutImage);



#endif /* end of include guard: __SOBEL_H__ */
