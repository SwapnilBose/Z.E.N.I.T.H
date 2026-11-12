
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "za_matrix.h"


/*
    Create a matrix.

    Example:
        Matrix a = matrix_create(2, 3);

    Produces:

        0 0 0
        0 0 0
*/
Matrix matrix_create(size_t rows, size_t cols)
{
    Matrix matrix = {0};

    /* Reject invalid dimensions. */
    if (rows == 0 || cols == 0)
    {
        return matrix;
    }

    /* Prevent integer overflow during allocation. */
    if (rows > SIZE_MAX / cols)
    {
        return matrix;
    }

    size_t count = rows * cols;

    if (count > SIZE_MAX / sizeof(float))
    {
        return matrix;
    }

    /* Allocate memory for all matrix elements. */
    matrix.data = calloc(count, sizeof(float));

    if (matrix.data == NULL)
    {
        return matrix;
    }

    matrix.rows = rows;
    matrix.cols = cols;

    return matrix;
}


/*
    Free matrix memory.

    Always free memory that was allocated
    dynamically when it is no longer needed.
*/
void matrix_free(Matrix *matrix)
{
    if (matrix == NULL)
    {
        return;
    }

    free(matrix->data);

    matrix->data = NULL;
    matrix->rows = 0;
    matrix->cols = 0;
}


/*
    Matrix multiplication.

    If:

        A = (m x n)
        B = (n x p)

    Then:

        C = A * B

        C = (m x p)
*/
Matrix matrix_multiply(const Matrix *a, const Matrix *b)
{
    Matrix result = {0};

    if (a == NULL || b == NULL)
    {
        return result;
    }

    if (a->data == NULL || b->data == NULL)
    {
        return result;
    }

    /* Columns of A must equal rows of B. */
    if (a->cols != b->rows)
    {
        return result;
    }

    result = matrix_create(a->rows, b->cols);

    if (result.data == NULL)
    {
        return result;
    }

    /*
        Standard matrix multiplication.

        Each result element is:

        C[i][j] = sum(A[i][k] * B[k][j])
    */

    for (size_t i = 0; i < a->rows; i++)
    {
        for (size_t j = 0; j < b->cols; j++)
        {
            float sum = 0.0f;

            for (size_t k = 0; k < a->cols; k++)
            {
                float x = a->data[i * a->cols + k];

                float y = b->data[k * b->cols + j];

                sum += x * y;
            }

            result.data[i * result.cols + j] = sum;
        }
    }

    return result;
}


/*
    ReLU activation function.

    ReLU(x) = max(0, x)

    Negative numbers become zero.
    Positive numbers remain unchanged.
*/
void matrix_relu(Matrix *matrix)
{
    if (matrix == NULL || matrix->data == NULL)
    {
        return;
    }

    size_t count = matrix->rows * matrix->cols;

    for (size_t i = 0; i < count; i++)
    {
        if (matrix->data[i] < 0.0f)
        {
            matrix->data[i] = 0.0f;
        }
    }
}


/* Print matrix in readable form. */
void matrix_print(const Matrix *matrix)
{
    if (matrix == NULL || matrix->data == NULL)
    {
        printf("Invalid matrix.\n");
        return;
    }

    for (size_t i = 0; i < matrix->rows; i++)
    {
        for (size_t j = 0; j < matrix->cols; j++)
        {
            printf("%8.3f ",
                matrix->data[i * matrix->cols + j]);
        }

        printf("\n");
    }
}
