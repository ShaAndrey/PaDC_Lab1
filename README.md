# PaDC_Lab1

Script reads two matrices from standard input and outputs their product into standard output. OpenMP can be used to parallelize computations. The following options can be specified:

#### **-s** - sets size of input matrices. Only square matrices are supported. Default: 1

#### **-b** - sets size of block for block matrix multiplication. If not specified, non-block algorithm will be used

#### **-i** - parallelize inner cycle (by columns) of used multiplication algorithm

#### **-o** - parallelize outer cycle (by rows) of used multiplication algorithm

Usage example:

```
$ g++ -fopenmp main.cpp -o script

$ ./script -s 3 -i -b 1 <input >output
```
