
#ifndef ZA_MATRIX_H
#define ZA_MATRIX_H

#include <stddef.h>

/*
    Matrix:
    A dynamically allocated, two-dimensional
    collection of floating-point numbers.
*/
typedef struct
{
    size_t rows;
    size_t cols;

    float *data;

} Matrix;


/* Allocate a matrix and initialize its elements to zero. */
Matrix matrix_create(size_t rows, size_t cols);

/* Release dynamically allocated matrix memory. */
void matrix_free(Matrix *matrix);

/* Multiply two compatible matrices. */
Matrix matrix_multiply(const Matrix *a, const Matrix *b);

/* Apply the ReLU activation function to every element. */
void matrix_relu(Matrix *matrix);

/* Print matrix contents. */
void matrix_print(const Matrix *matrix);

#endif
