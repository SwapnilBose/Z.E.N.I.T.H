
#include <stdio.h>

#include "za_matrix.h"

int main(void)
{
    /*
        Create two 2x2 matrices.

        A =  1  2
             3  4

        B =  5  6
             7  8
    */

    Matrix a = matrix_create(2, 2);
    Matrix b = matrix_create(2, 2);

    if (a.data == NULL || b.data == NULL)
    {
        printf("Memory allocation failed.\n");

        matrix_free(&a);
        matrix_free(&b);

        return 1;
    }

    /* Fill matrix A. */
    a.data[0] = 2.0f;
    a.data[1] = 0.0f;
    a.data[2] = -1.0f;
    a.data[3] = 3.0f;

    /* Fill matrix B. */
    b.data[0] = 5.0f;
    b.data[1] = 6.0f;
    b.data[2] = 7.0f;
    b.data[3] = 8.0f;

    printf("Matrix A:\n");
    matrix_print(&a);

    printf("\nMatrix B:\n");
    matrix_print(&b);

    /* Multiply A and B. */
    Matrix c = matrix_multiply(&a, &b);

    if (c.data == NULL)
    {
        printf("Matrix multiplication failed.\n");

        matrix_free(&a);
        matrix_free(&b);

        return 1;
    }

    printf("\nA multiplied by B:\n");
    matrix_print(&c);

    /* Test ReLU. */
    printf("\nApplying ReLU to a copy of the result:\n");

    Matrix d = matrix_create(2, 2);

    for (size_t i = 0; i < 4; i++)
    {
        d.data[i] = c.data[i] - 30.0f;
    }

    matrix_relu(&d);
    matrix_print(&d);

    /* Release all allocated memory. */
    matrix_free(&a);
    matrix_free(&b);
    matrix_free(&c);
    matrix_free(&d);

    return 0;
}
