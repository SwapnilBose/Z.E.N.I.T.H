
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "za_network.h"

int main(void)
{
    /*
     * Architecture:
     *
     * 1 input neuron
     * 4 hidden neurons
     * 1 output neuron
     */
    size_t architecture[] = {1, 4, 1};

    const size_t layer_count = 2;

    const size_t sample_count = 5;

    const float learning_rate = 0.01f;
    const int epochs = 5000;

    Matrix inputs = matrix_create(sample_count, 1);
    Matrix targets = matrix_create(sample_count, 1);

    if (inputs.data == NULL || targets.data == NULL)
    {
        printf("Failed to allocate training data.\n");

        matrix_free(&inputs);
        matrix_free(&targets);

        return 1;
    }

    /*
     * Training dataset:
     *
     * y = 2x + 1
     *
     * x:       0  1  2  3  4
     * target:  1  3  5  7  9
     */

    for (size_t i = 0; i < sample_count; i++)
    {
        inputs.data[i] = (float)i;
        targets.data[i] = 2.0f * (float)i + 1.0f;
    }

    /* Make random initialization reproducible. */
    srand(42);

    printf("ZENITH-ASR: Dynamic Neural Network\n");
    printf("-----------------------------------\n");

    Network network = network_create(
        architecture,
        layer_count
    );

    if (network.layers == NULL)
    {
        printf("Failed to create network.\n");

        matrix_free(&inputs);
        matrix_free(&targets);

        return 1;
    }

    printf("Network created successfully.\n");
    printf("Trainable layers: %zu\n\n", network.layer_count);

    /*
     * TRAINING
     */
    printf("Training started...\n\n");

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        float loss = network_train_step(
            &network,
            &inputs,
            &targets,
            learning_rate
        );

        if (!isfinite(loss))
        {
            printf("Training failed at epoch %d.\n", epoch);

            network_free(&network);
            matrix_free(&inputs);
            matrix_free(&targets);

            return 1;
        }

        if (epoch == 1 || epoch % 500 == 0)
        {
            printf(
                "Epoch %4d | Loss: %.8f\n",
                epoch,
                loss
            );
        }
    }

    /*
     * TESTING
     */
    printf("\nTraining completed.\n");

    Matrix predictions = network_forward(
        &network,
        &inputs
    );

    if (predictions.data == NULL)
    {
        printf("Prediction failed.\n");

        network_free(&network);
        matrix_free(&inputs);
        matrix_free(&targets);

        return 1;
    }

    printf("\nPredictions:\n");

    for (size_t i = 0; i < sample_count; i++)
    {
        printf(
            "Input: %.1f | Predicted: %.4f | Expected: %.1f\n",
            inputs.data[i],
            predictions.data[i],
            targets.data[i]
        );
    }

    /*
     * Test an unseen input.
     *
     * Expected:
     * y = 2(10) + 1 = 21
     */
    Matrix new_input = matrix_create(1, 1);

    if (new_input.data == NULL)
    {
        printf("Failed to allocate test input.\n");

        matrix_free(&predictions);
        network_free(&network);
        matrix_free(&inputs);
        matrix_free(&targets);

        return 1;
    }

    new_input.data[0] = 10.0f;

    Matrix new_prediction = network_forward(
        &network,
        &new_input
    );

    if (new_prediction.data == NULL)
    {
        printf("Prediction for unseen input failed.\n");

        matrix_free(&new_input);
        matrix_free(&predictions);
        network_free(&network);
        matrix_free(&inputs);
        matrix_free(&targets);

        return 1;
    }

    printf("\nUnseen input test:\n");
    printf("Input: 10\n");
    printf("Predicted: %.4f\n", new_prediction.data[0]);
    printf("Expected: 21.0000\n");

    /*
     * MEMORY CLEANUP
     */
    matrix_free(&new_prediction);
    matrix_free(&new_input);
    matrix_free(&predictions);

    network_free(&network);

    matrix_free(&inputs);
    matrix_free(&targets);

    printf("\nAll resources released.\n");

    return 0;
}
