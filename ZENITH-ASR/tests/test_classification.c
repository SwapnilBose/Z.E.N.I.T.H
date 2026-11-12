
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "za_network.h"
#include "za_softmax.h"
#include "za_loss.h"

/*
 * ZENITH-ASR
 * Implementation 012.2
 *
 * Three-class classification test.
 *
 * Input: 2 features
 * Hidden: 8 neurons
 * Output: 3 classes
 *
 * Class 0: around (-2, -2)
 * Class 1: around ( 2, -2)
 * Class 2: around ( 0,  2)
 */

#define SAMPLES_PER_CLASS 20
#define CLASS_COUNT 3
#define INPUT_SIZE 2

#define TOTAL_SAMPLES (SAMPLES_PER_CLASS * CLASS_COUNT)

#define EPOCHS 1000
#define LEARNING_RATE 0.03f
#define MAX_GRADIENT_NORM 5.0f

static void set_sample(
    Matrix *inputs,
    Matrix *targets,
    size_t row,
    float x,
    float y,
    size_t class_id
)
{
    inputs->data[row * INPUT_SIZE] = x;
    inputs->data[row * INPUT_SIZE + 1] = y;

    /*
     * One-hot encoding:
     *
     * Class 0: [1, 0, 0]
     * Class 1: [0, 1, 0]
     * Class 2: [0, 0, 1]
     */

    for (size_t c = 0; c < CLASS_COUNT; c++)
    {
        targets->data[row * CLASS_COUNT + c] =
            (c == class_id) ? 1.0f : 0.0f;
    }
}

static void generate_dataset(
    Matrix *inputs,
    Matrix *targets
)
{
    size_t row = 0;

    /*
     * Class 0: lower-left cluster.
     */
    for (size_t i = 0; i < SAMPLES_PER_CLASS; i++)
    {
        float offset = (float)(i % 5) * 0.1f;

        set_sample(
            inputs,
            targets,
            row++,
            -2.0f + offset,
            -2.0f - offset,
            0
        );
    }

    /*
     * Class 1: lower-right cluster.
     */
    for (size_t i = 0; i < SAMPLES_PER_CLASS; i++)
    {
        float offset = (float)(i % 5) * 0.1f;

        set_sample(
            inputs,
            targets,
            row++,
            2.0f - offset,
            -2.0f + offset,
            1
        );
    }

    /*
     * Class 2: upper cluster.
     */
    for (size_t i = 0; i < SAMPLES_PER_CLASS; i++)
    {
        float offset = (float)(i % 5) * 0.1f;

        set_sample(
            inputs,
            targets,
            row++,
            offset,
            2.0f - offset,
            2
        );
    }
}

static size_t argmax_row(
    const Matrix *matrix,
    size_t row
)
{
    size_t best = 0;

    for (size_t j = 1; j < matrix->cols; j++)
    {
        if (matrix->data[row * matrix->cols + j] >
            matrix->data[row * matrix->cols + best])
        {
            best = j;
        }
    }

    return best;
}

static float calculate_accuracy(
    const Matrix *probabilities,
    const Matrix *targets
)
{
    size_t correct = 0;

    for (size_t r = 0; r < probabilities->rows; r++)
    {
        size_t predicted = argmax_row(probabilities, r);
        size_t expected = argmax_row(targets, r);

        if (predicted == expected)
            correct++;
    }

    return (float)correct / (float)probabilities->rows;
}

int main(void)
{
    printf("====================================\n");
    printf(" ZENITH CLASSIFICATION TEST\n");
    printf(" Implementation 012.2\n");
    printf("====================================\n\n");

    srand(42);

    /*
     * STEP 1: CREATE DATASET
     */
    Matrix inputs = matrix_create(
        TOTAL_SAMPLES,
        INPUT_SIZE
    );

    Matrix targets = matrix_create(
        TOTAL_SAMPLES,
        CLASS_COUNT
    );

    if (inputs.data == NULL || targets.data == NULL)
    {
        printf("ERROR: Dataset allocation failed.\n");

        matrix_free(&inputs);
        matrix_free(&targets);

        return 1;
    }

    generate_dataset(&inputs, &targets);

    printf("Dataset generated.\n");
    printf("Samples: %d\n", TOTAL_SAMPLES);
    printf("Features: %d\n", INPUT_SIZE);
    printf("Classes: %d\n\n", CLASS_COUNT);

    /*
     * STEP 2: CREATE NETWORK
     *
     * 2 inputs -> 8 hidden neurons -> 3 outputs
     */
    size_t architecture[] = {2, 8, 3};

    Network network = network_create(
        architecture,
        2
    );

    if (network.layers == NULL)
    {
        printf("ERROR: Network creation failed.\n");

        matrix_free(&inputs);
        matrix_free(&targets);

        return 1;
    }

    printf("Network created: 2 -> 8 -> 3\n\n");

    /*
     * STEP 3: INITIAL LOSS
     */
    Matrix initial_logits = network_forward(
        &network,
        &inputs
    );

    Matrix initial_probabilities = softmax_forward(
        &initial_logits
    );

    float initial_loss = cross_entropy_loss(
        &initial_probabilities,
        &targets
    );

    float initial_accuracy = calculate_accuracy(
        &initial_probabilities,
        &targets
    );

    matrix_free(&initial_logits);
    matrix_free(&initial_probabilities);

    if (!isfinite(initial_loss))
    {
        printf("ERROR: Initial loss calculation failed.\n");

        network_free(&network);
        matrix_free(&inputs);
        matrix_free(&targets);

        return 1;
    }

    printf("Initial loss: %.6f\n", initial_loss);
    printf("Initial accuracy: %.2f%%\n\n",
           initial_accuracy * 100.0f);

    /*
     * STEP 4: TRAINING
     */
    printf("Training started...\n\n");

    for (int epoch = 1; epoch <= EPOCHS; epoch++)
    {
        float loss = network_train_classification_step(
            &network,
            &inputs,
            &targets,
            LEARNING_RATE,
            MAX_GRADIENT_NORM
        );

        if (!isfinite(loss))
        {
            printf("ERROR: Training failed at epoch %d.\n", epoch);

            network_free(&network);
            matrix_free(&inputs);
            matrix_free(&targets);

            return 1;
        }

        if (epoch % 100 == 0 || epoch == 1)
        {
            printf(
                "Epoch %4d | Loss: %.6f\n",
                epoch,
                loss
            );
        }
    }

    /*
     * STEP 5: EVALUATION
     */
    Matrix final_logits = network_forward(
        &network,
        &inputs
    );

    Matrix final_probabilities = softmax_forward(
        &final_logits
    );

    float final_loss = cross_entropy_loss(
        &final_probabilities,
        &targets
    );

    float final_accuracy = calculate_accuracy(
        &final_probabilities,
        &targets
    );

    printf("\n====================================\n");
    printf(" FINAL RESULTS\n");
    printf("====================================\n");

    printf("Initial loss: %.6f\n", initial_loss);
    printf("Final loss:   %.6f\n", final_loss);

    printf("Initial accuracy: %.2f%%\n",
           initial_accuracy * 100.0f);

    printf("Final accuracy:   %.2f%%\n",
           final_accuracy * 100.0f);

    /*
     * STEP 6: SHOW SAMPLE PREDICTIONS
     */
    printf("\nSample predictions:\n");

    for (size_t r = 0; r < TOTAL_SAMPLES; r += 10)
    {
        size_t predicted = argmax_row(
            &final_probabilities,
            r
        );

        size_t expected = argmax_row(
            &targets,
            r
        );

        printf(
            "Sample %2zu | Predicted: %zu | Expected: %zu\n",
            r,
            predicted,
            expected
        );
    }

    /*
     * STEP 7: TEST RESULT
     */
    int passed =
        isfinite(final_loss) &&
        final_loss < initial_loss &&
        final_accuracy >= 0.90f;

    printf("\n====================================\n");

    if (passed)
        printf(" CLASSIFICATION TEST PASSED\n");
    else
        printf(" CLASSIFICATION TEST NEEDS INVESTIGATION\n");

    printf("====================================\n");

    /*
     * CLEANUP
     */
    matrix_free(&final_logits);
    matrix_free(&final_probabilities);

    network_free(&network);
    matrix_free(&inputs);
    matrix_free(&targets);

    return passed ? 0 : 1;
}
