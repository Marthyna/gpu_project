#ifndef READ_PPM
#define READ_PPM

int writeppm(char *filename, int height, int width, unsigned char *data);
unsigned char *readppm(char *filename, int *height, int *width);

#endif
