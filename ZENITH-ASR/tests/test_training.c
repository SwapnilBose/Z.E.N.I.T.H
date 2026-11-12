
#include <stdio.h>
#include <stdlib.h>

#include "za_train.h"

int main(void)
{
    srand(42);

    /*
        Our training dataset:

        x:       0  1  2  3  4
        target:  1  3  5  7  9

        Hidden relationship:

        y = 2x + 1

        The neural network does not know this.
    */

    Matrix input = matrix_create(5, 1);
    Matrix target = matrix_create(5, 1);

    if (input.data == NULL || target.data == NULL)
    {
        printf("Dataset allocation failed.\n");

        matrix_free(&input);
        matrix_free(&target);

        return 1;
    }

    for (size_t i = 0; i < 5; i++)
    {
        input.data[i] = (float)i;

        target.data[i] =
            2.0f * (float)i + 1.0f;
    }

    /* Create one input neuron and one output neuron. */
    DenseLayer model = dense_create(1, 1);

    if (model.weights.data == NULL ||
        model.biases.data == NULL)
    {
        printf("Model allocation failed.\n");

        matrix_free(&input);
        matrix_free(&target);
        dense_free(&model);

        return 1;
    }

    /*
        Start with known incorrect parameters.

        Weight = 0
        Bias   = 0
    */
    model.weights.data[0] = 0.0f;
    model.biases.data[0] = 0.0f;

    const float learning_rate = 0.01f;
    const int epochs = 1000;

    printf("ZENITH-ASR: Training experiment\n");
    printf("--------------------------------\n");

    printf("Initial weight: %.4f\n",
           model.weights.data[0]);

    printf("Initial bias:   %.4f\n\n",
           model.biases.data[0]);

    /* Train the model repeatedly. */
    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        float loss = dense_train_step(
            &model,
            &input,
            &target,
            learning_rate
        );

        if (epoch == 1 || epoch % 100 == 0)
        {
            printf(
                "Epoch %4d | Loss: %.8f | "
                "Weight: %.4f | Bias: %.4f\n",
                epoch,
                loss,
                model.weights.data[0],
                model.biases.data[0]
            );
        }
    }

    printf("\nTraining completed.\n");

    printf("Learned weight: %.4f\n",
           model.weights.data[0]);

    printf("Learned bias:   %.4f\n",
           model.biases.data[0]);

    /* Test a new input. */
    Matrix test_input = matrix_create(1, 1);

    if (test_input.data == NULL)
    {
        printf("Test input allocation failed.\n");

        matrix_free(&input);
        matrix_free(&target);
        dense_free(&model);

        return 1;
    }

    test_input.data[0] = 10.0f;

    Matrix result = dense_forward(
        &model,
        &test_input,
        ACTIVATION_NONE
    );

    if (result.data != NULL)
    {
        printf("\nPrediction for x = 10: %.4f\n",
               result.data[0]);

        printf("Expected answer:       21.0000\n");
    }
    else
    {
        printf("\nPrediction failed.\n");
    }

    /* Clean up. */
    matrix_free(&input);
    matrix_free(&target);
    matrix_free(&test_input);
    matrix_free(&result);

    dense_free(&model);

    return 0;
}
