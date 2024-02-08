#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "readppm.h"


int writeppm(char *filename, int height, int width, unsigned char *data)
{
	FILE *fp;
	int error = 1;
	int i, h, v;

	if (filename != NULL)
	{
		fp = fopen(filename,"w");
		
		if (fp != NULL)
		{
			// Write PPM file
			// Header	
			fprintf(fp, "P3\n");
			fprintf(fp, "# written by Ingemars PPM writer\n");
			fprintf(fp, "%d %d\n", height, width);
			fprintf(fp, "%d\n", 255); // range
			
			// Data
			for (v = height-1; v >=0; v--)
			{
				for (h = 0; h < width; h++)
				{
					i = (width*v + h)*4; // assumes rgba, not rgba
					fprintf(fp, "%d %d %d ", data[i], data[i+1], data[i+2]);
				}
				fprintf(fp, "\n"); // range
			}
			
			if (fwrite("\n",sizeof(char),1,fp) == 1)
				error = 0; // Probable success
			fclose(fp);
		}
	}
	return(error);
}



unsigned char *readppm(char *filename, int *height, int *width)
{
	FILE *fd;
	int  k, nm;
	char c;
	int i,j;
	char b[100];
	float s;
	int red, green, blue;
	long numbytes, howmuch;
	int n;
	int m;
	unsigned char *image;
	
	fd = fopen(filename, "rb");
	if (fd == NULL)
	{
		printf("Could not open %s\n", filename);
		return NULL;
	}
	c = getc(fd);
	if (c=='P' || c=='p')
		c = getc(fd);
	
	if (c == '3')
	{
		printf("%s is a PPM file (plain text version)\n", filename);
		
		// NOTE: This is not very good PPM code! Comments are not allowed
		// except immediately after the magic number.
		c = getc(fd);
		if (c == '\n' || c == '\r') // Skip any line break and comments
		{
			c = getc(fd);
			while(c == '#') 
			{
				fscanf(fd, "%[^\n\r] ", b);
				printf("%s\n",b);
				c = getc(fd);
			}
			ungetc(c,fd); 
		}
		fscanf(fd, "%d %d %d", &n, &m, &k);
		
		printf("%d rows  %d columns  max value= %d\n",n,m,k);
		
		numbytes = n * m * 3;
		image = (unsigned char *) malloc(numbytes);
		if (image == NULL)
		{
			printf("Memory allocation failed!\n"); 
			return NULL;
		}
		for(i=n-1;i>=0;i--) for(j=0;j<m;j++)
		{
			fscanf(fd,"%d %d %d",&red, &green, &blue );
			image[(i*n+j)*3]=red * 255 / k;
			image[(i*n+j)*3+1]=green * 255 / k;
			image[(i*n+j)*3+2]=blue * 255 / k;
		}
	}
	else
	if (c == '6')
	{
		printf("%s is a PPM file (raw version)!\n", filename); 
		
		c = getc(fd);
		if (c == '\n' || c == '\r') // Skip any line break and comments
		{
			c = getc(fd);
			while(c == '#') 
			{
				fscanf(fd, "%[^\n\r] ", b);
				printf("%s\n",b);
				c = getc(fd);
			}
			ungetc(c,fd); 
		}
		fscanf(fd, "%d %d %d", &n, &m, &k);
		printf("%d rows  %d columns  max value= %d\n",n,m,k);
		c = getc(fd); // Skip the last whitespace
		
		numbytes = n * m * 3;
		image = (unsigned char *) malloc(numbytes);
		if (image == NULL)
		{
			printf("Memory allocation failed!\n"); 
			return NULL;
		}
		// Read and re-order as necessary
		for(i=n-1;i>=0;i--) for(j=0;j<m;j++)
		{
			image[(i*n+j)*3+0]=getc(fd);
			image[(i*n+j)*3+1]=getc(fd);
			image[(i*n+j)*3+2]=getc(fd);
		}
	}
	else
	{
		printf("%s is not a PPM file!\n", filename); 
		return NULL;
	}
	
	printf("read image\n");
	
	*height = n;
	*width = m;
	return image;
}
