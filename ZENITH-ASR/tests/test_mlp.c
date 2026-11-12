
#include <stdio.h>
#include <stdlib.h>

#include "za_mlp.h"

int main(void)
{
    Matrix input = matrix_create(5, 1);
    Matrix target = matrix_create(5, 1);

    if (input.data == NULL || target.data == NULL)
    {
        printf("Dataset allocation failed.\n");
        matrix_free(&input);
        matrix_free(&target);
        return 1;
    }

    /* Dataset: y = 2x + 1 */
    for (size_t i = 0; i < 5; i++)
    {
        input.data[i] = (float)i;
        target.data[i] = 2.0f * (float)i + 1.0f;
    }

    MLPNetwork model = mlp_create();

    if (model.hidden.weights.data == NULL ||
        model.hidden.biases.data == NULL ||
        model.output.weights.data == NULL ||
        model.output.biases.data == NULL)
    {
        printf("Network allocation failed.\n");
        matrix_free(&input);
        matrix_free(&target);
        mlp_free(&model);
        return 1;
    }

    /*
     * Deterministic starting parameters.
     * All hidden pre-activations are positive for x = 0..4.
     */
    model.hidden.weights.data[0] = 0.5f;
    model.hidden.weights.data[1] = 1.0f;

    model.hidden.biases.data[0] = 1.0f;
    model.hidden.biases.data[1] = 1.0f;

    model.output.weights.data[0] = 0.5f;
    model.output.weights.data[1] = 0.5f;

    model.output.biases.data[0] = 0.5f;

    const float learning_rate = 0.01f;
    const int epochs = 3000;

    printf("ZENITH-ASR: Multilayer training\n");
    printf("--------------------------------\n");

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        float loss = mlp_train_step(
            &model,
            &input,
            &target,
            learning_rate
        );

        if (epoch == 1 || epoch % 300 == 0)
        {
            printf(
                "Epoch %4d | Loss: %.8f\n",
                epoch,
                loss
            );
        }
    }

    printf("\nTraining completed.\n");

    Matrix test = matrix_create(1, 1);

    if (test.data == NULL)
    {
        printf("Test allocation failed.\n");
        matrix_free(&input);
        matrix_free(&target);
        mlp_free(&model);
        return 1;
    }

    test.data[0] = 10.0f;

    Matrix prediction = mlp_forward(&model, &test);

    if (prediction.data != NULL)
    {
        printf(
            "\nPrediction for x = 10: %.4f\n",
            prediction.data[0]
        );

        printf("Expected answer:       21.0000\n");
    }
    else
    {
        printf("Prediction failed.\n");
    }

    matrix_free(&input);
    matrix_free(&target);
    matrix_free(&test);
    matrix_free(&prediction);

    mlp_free(&model);

    return 0;
}
