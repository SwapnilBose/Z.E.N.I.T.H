
#include <stdio.h>
#include <math.h>

#include "za_network.h"

int main(void)
{
    printf("ZENITH-ASR: Mini-Batch Training Test\n");
    printf("------------------------------------\n");

    /*
     * Architecture:
     *
     * 1 input neuron
     * 8 hidden neurons
     * 1 output neuron
     */
    const size_t sizes[] = {1, 8, 1};

    Network network = network_create(sizes, 2);

    if (network.layers == NULL)
    {
        printf("ERROR: Network creation failed.\n");
        return 1;
    }

    /*
     * Five training examples.
     *
     * Target relationship: y = 2x + 1
     */
    Matrix input = matrix_create(5, 1);
    Matrix target = matrix_create(5, 1);

    if (input.data == NULL || target.data == NULL)
    {
        printf("ERROR: Matrix allocation failed.\n");

        matrix_free(&input);
        matrix_free(&target);
        network_free(&network);

        return 1;
    }

    for (size_t i = 0; i < 5; i++)
    {
        input.data[i] = (float)i;
        target.data[i] = 2.0f * (float)i + 1.0f;
    }

    printf("\nBatch size: %zu\n", input.rows);
    printf("Training examples: 5\n");

    /*
     * Train using the entire batch.
     */
    printf("\nTraining started...\n");

    const int epochs = 5000;
    const float learning_rate = 0.01f;

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        float loss = network_train_step(
            &network,
            &input,
            &target,
            learning_rate
        );

        if (!isfinite(loss))
        {
            printf("ERROR: Training failed at epoch %d.\n", epoch);

            matrix_free(&input);
            matrix_free(&target);
            network_free(&network);

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

    printf("\nTraining completed.\n");

    /*
     * Test all five examples together.
     */
    Matrix predictions = network_forward(&network, &input);

    if (predictions.data == NULL)
    {
        printf("ERROR: Prediction failed.\n");

        matrix_free(&input);
        matrix_free(&target);
        network_free(&network);

        return 1;
    }

    printf("\nBatch predictions:\n");

    int passed = 1;

    for (size_t i = 0; i < input.rows; i++)
    {
        float predicted = predictions.data[i];
        float expected = target.data[i];

        printf(
            "Input: %.1f | Predicted: %.4f | Expected: %.1f\n",
            input.data[i],
            predicted,
            expected
        );

        if (fabsf(predicted - expected) > 0.05f)
            passed = 0;
    }

    printf("\nBatch test: %s\n",
           passed ? "PASSED" : "FAILED");

    /*
     * Release all allocated resources.
     */
    matrix_free(&predictions);
    matrix_free(&input);
    matrix_free(&target);
    network_free(&network);

    printf("\nAll resources released.\n");

    return passed ? 0 : 1;
}
