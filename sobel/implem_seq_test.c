/* Yep, we'll do this, allows to test static functions as well */
#define REMOVE_MAIN
#include "implem_seq.c"
#include "common.c"

#define RUN_TEST(test) \
do { \
        if (test() != 0) { \
                nbFailed++; \
        } \
} while (0)


int test_conv_3_by_3_id()
{
        int i;
        printf("It should behave correctly with identity kernel... ");
        kernel_t kernel = {{0, 0, 0},
                             {0, 1, 0},
                             {0, 0, 0}};

        struct image mat = IMAGE_INITIALIZER;
        int16_t outPx = 0;
        mat.type = GreyScale;
        mat.width = 3;
        mat.height = 3;
        mat.data = calloc(9, sizeof(unsigned char));
        for (i = 0; i < 9; i++) {
                mat.data[i] = i+1;
        }

        convolution_3_by_3(&mat, kernel, 1, 1, &outPx);

        if (outPx == 5) {
                printf("OK\n");
                return 0;
        } else {
                printf("FAILED: got %d instead of %d\n", outPx, mat.data[4]);
                return -1;
        }
}




int test_conv_3_by_3_null()
{
        printf("It should behave correctly with null kernel... ");
        kernel_t kernel = {{0, 0, 0},
                             {0, 0, 0},
                             {0, 0, 0}};

        struct image mat = IMAGE_INITIALIZER;
        int16_t outPx = 0;

        mat.type = GreyScale;
        mat.width = 3;
        mat.height = 3;
        mat.data = calloc(9, sizeof(unsigned char));
        for (int i = 0; i < 9; i++) {
                mat.data[i] = i+1;
        }

        convolution_3_by_3(&mat, kernel, 1, 1, &outPx);

        if (outPx == 0) {
                printf("OK\n");
                return 0;
        } else {
                printf("FAILED: got %d instead of %d\n", outPx, 0);
                return -1;
        }
}



int test_conv_3_by_3_ex1()
{
        printf("It should behave correctly with gradient kernel... ");
        kernel_t kernel;
        (void) initKernelY(kernel);
        

        struct image mat = IMAGE_INITIALIZER;
        int16_t outPx = 0;

        mat.type = GreyScale;
        mat.width = 3;
        mat.height = 3;
        mat.data = calloc(9, sizeof(unsigned char));
        for (int i = 0; i < 9; i++) {
                mat.data[i] = i+1;
        }

        convolution_3_by_3(&mat, kernel, 1, 1, &outPx);

        if (outPx == -24) {
                printf("OK\n");
                return 0;
        } else {
                printf("FAILED: got %d instead of %d\n", outPx, -24);
                return -1;
        }
}



int test_conv_3_by_3_uniformeY()
{
        printf("It should behave correctly with uniform matrix for gradY... ");
        kernel_t kernel;
        (void) initKernelY(kernel);
        

        struct image mat = IMAGE_INITIALIZER;
        int16_t outPx = 0;

        mat.type = GreyScale;
        mat.width = 3;
        mat.height = 3;
        mat.data = calloc(9, sizeof(unsigned char));
        for (int i = 0; i < 9; i++) {
                mat.data[i] = 100;
        }

        convolution_3_by_3(&mat, kernel, 1, 1, &outPx);

        if (outPx == 0) {
                printf("OK\n");
                return 0;
        } else {
                printf("FAILED: got %d instead of %d\n", outPx, 0);
                return -1;
        }
}



int test_conv_3_by_3_uniformeX()
{
        printf("It should behave correctly with uniform matrix for gradX... ");
        kernel_t kernel;
        (void) initKernelX(kernel);
        

        struct image mat = IMAGE_INITIALIZER;
        int16_t outPx = 0;

        mat.type = GreyScale;
        mat.width = 3;
        mat.height = 3;
        mat.data = calloc(9, sizeof(unsigned char));
        for (int i = 0; i < 9; i++) {
                mat.data[i] = 100;
        }

        convolution_3_by_3(&mat, kernel, 1, 1, &outPx);

        if (outPx == 0) {
                printf("OK\n");
                return 0;
        } else {
                printf("FAILED: got %d instead of %d\n", outPx, 0);
                return -1;
        }
}



int main(void)
{
        int nbFailed = 0;

        RUN_TEST(test_conv_3_by_3_id);
        RUN_TEST(test_conv_3_by_3_null);
        RUN_TEST(test_conv_3_by_3_ex1);
        RUN_TEST(test_conv_3_by_3_uniformeX);
        RUN_TEST(test_conv_3_by_3_uniformeY);


        return nbFailed;
}
