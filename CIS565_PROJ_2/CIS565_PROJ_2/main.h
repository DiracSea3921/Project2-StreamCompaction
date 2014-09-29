#include "kernel.h"
#include <iostream>
#include <fstream>

using namespace std;

void serial_prefix_sum(const int *a, int *b, int length);

void serial_scatter(const int *a, int *b, int length);