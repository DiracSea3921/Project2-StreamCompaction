#include "main.h"

void serial_prefix_sum(const int *a, int *b, int length)
{
	if(length<1)
		return;

	b[0] = a[0];
	for(int i=1;i<length;i++){
		b[i] = b[i-1] + a[i];
	}
}

void serial_scatter(const int *a, int *b, int length)
{
	if(length<1)
		return;

	for(int i=0;i<length;i++){
		b[i] = a[i]>0? 1:0;
	}

	serial_prefix_sum(b,b,length);
}