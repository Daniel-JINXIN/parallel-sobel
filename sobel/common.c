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
 * Encode an RGBA image ot a file.
 */
int encode_image(const char *destFileName, struct image *pImage)
{
        check_null(pImage);
        check_null(destFileName);
        check (pImage->type == RGBA, "The image must be RGBA");
{

        int ret = lodepng_encode32_file(destFileName, pImage->data,
                                        pImage->width, pImage->height);
        check (ret == 0, "Failed to write image to file %s", destFileName);
 
        return 0;
error:
        return -1;
}
}





/* Remove trailing comma, closes the JSON array, and closes the file */
void finalizeLogFile(FILE *logFile)
{
        /* Rewind from two chars ("\n" and "," */
        fseek(logFile, -2, SEEK_END);

        /* Finalize the array */
        fprintf(logFile, "\n]");

        fclose(logFile);
}






// This is a hack to allow testing, see usein tests/ subdir
#ifndef IN_TEST
int main(int argc, const char *argv[])
{
        int ret;
        FILE *logFile = NULL;
        char logName[256];

        if (argc < 3) {
                printf("Usage: %s inImage outImage [logFileName]\n", argv[0]);
                exit(1);
        }

        /* If there is a logfile, use it */
        if (argc >= 4) {
                logFile = fopen(argv[3], "w");
                check (logFile != NULL, "Error while opening the logfile %s\n", argv[3]);
                fprintf(logFile, "[\n");

                /* If there is a log description, use it. Otherwise, use the log filename */
                if (argc == 5) {
                        strncpy(logName, argv[4], 255);
                } else {
                        strncpy(logName, argv[3], 255);
                }
        }



        const char *inFileName = argv[1];
        const char *outFileName = argv[2];

        //XXX we could just check that we have rights on the files right at the beginning

        struct image inImage = IMAGE_INITIALIZER;
        struct image outImage = IMAGE_INITIALIZER;

        ret = decode_image(inFileName, &inImage);
        check (ret == 0, "Image decoding failed");

        double startTime = omp_get_wtime();
        ret = sobel(&inImage, &outImage);
        double endTime = omp_get_wtime();
        check (ret == 0, "Sobel edge detection failed");

        log_time(logFile, logName, inImage.width * inImage.height, endTime - startTime);

        ret = encode_image(outFileName, &outImage);
        check (ret == 0, "Error while storing image to disk");

        if (logFile) {
                finalizeLogFile(logFile);
        };

        printf("%lf sec (on %u threads)\n", endTime - startTime, atoi(getenv("OMP_NUM_THREADS")));

        return 0;

error:
        if (logFile) {
                fclose(logFile);
        }
        return 1;
}
#endif
