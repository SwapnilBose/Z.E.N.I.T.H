
#include <stdio.h>
#include <math.h>

#include "za_mlp.h"

/* Calculate average MSE without changing network parameters. */
static float calculate_loss(
    const MLPNetwork *model,
    const Matrix *input,
    const Matrix *target)
{
    Matrix prediction = mlp_forward(model, input);

    if (prediction.data == NULL)
        return NAN;

    float loss = 0.0f;

    for (size_t i = 0; i < input->rows; i++)
    {
        float error = prediction.data[i] - target->data[i];
        loss += error * error;
    }

    loss /= (float)input->rows;

    matrix_free(&prediction);

    return loss;
}

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

    /* Use fixed parameters so the test is reproducible. */
    model.hidden.weights.data[0] = 0.5f;
    model.hidden.weights.data[1] = 1.0f;

    model.hidden.biases.data[0] = 1.0f;
    model.hidden.biases.data[1] = 1.0f;

    model.output.weights.data[0] = 0.5f;
    model.output.weights.data[1] = 0.5f;

    model.output.biases.data[0] = 0.5f;

    /*
     * Pointers to all seven trainable parameters.
     *
     * 0, 1: hidden weights
     * 2, 3: hidden biases
     * 4, 5: output weights
     * 6:    output bias
     */
    float *parameters[7] =
    {
        &model.hidden.weights.data[0],
        &model.hidden.weights.data[1],

        &model.hidden.biases.data[0],
        &model.hidden.biases.data[1],

        &model.output.weights.data[0],
        &model.output.weights.data[1],

        &model.output.biases.data[0]
    };

    float original[7];
    float analytical[7];

    for (int i = 0; i < 7; i++)
        original[i] = *parameters[i];

    const float learning_rate = 0.001f;
    const float epsilon = 0.001f;

    /*
     * Run one real training step.
     *
     * The parameter change reveals the analytical gradient:
     *
     * gradient = (old_parameter - new_parameter) / learning_rate
     */
    float loss = mlp_train_step(
        &model,
        &input,
        &target,
        learning_rate
    );

    if (!isfinite(loss))
    {
        printf("Training step failed.\n");
        matrix_free(&input);
        matrix_free(&target);
        mlp_free(&model);
        return 1;
    }

    for (int i = 0; i < 7; i++)
    {
        analytical[i] =
            (original[i] - *parameters[i]) / learning_rate;

        /* Restore original parameter. */
        *parameters[i] = original[i];
    }

    printf("ZENITH-ASR: Gradient checking\n");
    printf("--------------------------------\n");
    printf("Initial loss: %.6f\n\n", loss);

    int passed = 0;

    for (int i = 0; i < 7; i++)
    {
        /* Calculate loss at parameter + epsilon. */
        *parameters[i] = original[i] + epsilon;

        float loss_plus = calculate_loss(
            &model, &input, &target
        );

        /* Calculate loss at parameter - epsilon. */
        *parameters[i] = original[i] - epsilon;

        float loss_minus = calculate_loss(
            &model, &input, &target
        );

        /* Restore parameter. */
        *parameters[i] = original[i];

        if (!isfinite(loss_plus) || !isfinite(loss_minus))
        {
            printf("Parameter %d: calculation failed.\n", i);
            continue;
        }

        /* Central finite difference. */
        float numerical =
            (loss_plus - loss_minus) / (2.0f * epsilon);

        float difference = fabsf(analytical[i] - numerical);

        /* Allow a small floating-point tolerance. */
        float tolerance = 0.005f + 0.01f * fabsf(numerical);

        printf(
            "Parameter %d | Analytical: % .6f | "
            "Numerical: % .6f | Difference: %.6f | ",
            i,
            analytical[i],
            numerical,
            difference
        );

        if (difference <= tolerance)
        {
            printf("PASS\n");
            passed++;
        }
        else
        {
            printf("FAIL\n");
        }
    }

    printf("\n--------------------------------\n");
    printf("Passed: %d / 7\n", passed);

    if (passed == 7)
        printf("All gradient checks passed!\n");
    else
        printf("Some gradients need investigation.\n");

    matrix_free(&input);
    matrix_free(&target);
    mlp_free(&model);

    return passed == 7 ? 0 : 1;
}
