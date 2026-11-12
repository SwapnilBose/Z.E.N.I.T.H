
#ifndef ZA_TRAIN_H
#define ZA_TRAIN_H

#include "za_dense.h"

/*
    Train one dense layer using one batch.

    input:
        N x 1 matrix

    target:
        N x 1 matrix

    learning_rate:
        Controls the size of each weight update.

    Returns:
        MSE loss before the parameter update.
        NAN if the inputs are invalid.
*/
float dense_train_step(
    DenseLayer *layer,
    const Matrix *input,
    const Matrix *target,
    float learning_rate
);

#endif
