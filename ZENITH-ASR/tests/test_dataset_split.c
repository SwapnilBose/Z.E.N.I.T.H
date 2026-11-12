#include "za_dataset.h"

#include <stdio.h>

/*
 * Check whether two rows contain exactly the same input features
 * and target values. The datasets must have matching dimensions.
 */
static int rows_are_equal(
    const Dataset *a, size_t row_a,
    const Dataset *b, size_t row_b)
{
    for (size_t j = 0; j < a->input_size; j++)
    {
        if (a->inputs.data[row_a * a->input_size + j] !=
            b->inputs.data[row_b * b->input_size + j])
            return 0;
    }

    for (size_t j = 0; j < a->output_size; j++)
    {
        if (a->targets.data[row_a * a->output_size + j] !=
            b->targets.data[row_b * b->output_size + j])
            return 0;
    }

    return 1;
}

/* Count occurrences of one complete sample in another dataset. */
static size_t count_row_occurrences(
    const Dataset *reference,
    size_t reference_row,
    const Dataset *candidate)
{
    size_t count = 0;

    for (size_t i = 0; i < candidate->sample_count; i++)
    {
        if (rows_are_equal(reference, reference_row, candidate, i))
            count++;
    }

    return count;
}

/*
 * Verify that two datasets contain the same complete samples,
 * including input-target pairs. Occurrence counts handle duplicates.
 */
static int dataset_has_same_samples(
    const Dataset *original,
    const Dataset *candidate)
{
    if (original == NULL || candidate == NULL ||
        original->sample_count != candidate->sample_count ||
        original->input_size != candidate->input_size ||
        original->output_size != candidate->output_size ||
        original->inputs.data == NULL ||
        original->targets.data == NULL ||
        candidate->inputs.data == NULL ||
        candidate->targets.data == NULL)
        return 0;

    for (size_t i = 0; i < original->sample_count; i++)
    {
        size_t expected =
            count_row_occurrences(original, i, original);
        size_t actual =
            count_row_occurrences(original, i, candidate);

        if (actual != expected)
            return 0;
    }

    return 1;
}

/* Check whether two datasets contain identical rows in identical order. */
static int datasets_are_equal(const Dataset *a, const Dataset *b)
{
    if (a == NULL || b == NULL ||
        a->sample_count != b->sample_count ||
        a->input_size != b->input_size ||
        a->output_size != b->output_size ||
        a->inputs.data == NULL || a->targets.data == NULL ||
        b->inputs.data == NULL || b->targets.data == NULL)
        return 0;

    for (size_t i = 0; i < a->sample_count; i++)
    {
        if (!rows_are_equal(a, i, b, i))
            return 0;
    }

    return 1;
}

/*
 * Verify that every complete original sample appears exactly the
 * same number of times across training, validation, and testing.
 */
static int check_split_coverage(
    const Dataset *original,
    const Dataset *training,
    const Dataset *validation,
    const Dataset *testing)
{
    if (original == NULL || training == NULL ||
        validation == NULL || testing == NULL ||
        original->inputs.data == NULL ||
        original->targets.data == NULL ||
        training->inputs.data == NULL ||
        training->targets.data == NULL ||
        validation->inputs.data == NULL ||
        validation->targets.data == NULL ||
        testing->inputs.data == NULL ||
        testing->targets.data == NULL)
        return 0;

    const Dataset *sets[] = {training, validation, testing};

    size_t total = training->sample_count +
                   validation->sample_count +
                   testing->sample_count;

    if (total != original->sample_count)
        return 0;

    /* All subsets must have the same feature dimensions. */
    for (size_t s = 0; s < 3; s++)
    {
        if (sets[s]->input_size != original->input_size ||
            sets[s]->output_size != original->output_size)
            return 0;
    }

    /*
     * Compare complete-row occurrence counts, not just one input
     * feature. This also works if the dataset contains duplicate rows.
     */
    for (size_t i = 0; i < original->sample_count; i++)
    {
        size_t expected =
            count_row_occurrences(original, i, original);
        size_t actual = 0;

        for (size_t s = 0; s < 3; s++)
            actual += count_row_occurrences(original, i, sets[s]);

        if (actual != expected)
            return 0;
    }

    return 1;
}

static int test_dataset_shuffle(void)
{
    printf("Test 1: Dataset shuffling\n");
    printf("-------------------------\n");

    /*
     * Keep an untouched reference copy and create two independent
     * copies to verify deterministic shuffling with the same seed.
     */
    Dataset original = dataset_load_csv("data/linear.csv", 2, 1, 1);
    Dataset first = dataset_load_csv("data/linear.csv", 2, 1, 1);
    Dataset second = dataset_load_csv("data/linear.csv", 2, 1, 1);

    if (original.inputs.data == NULL || original.targets.data == NULL ||
        first.inputs.data == NULL || first.targets.data == NULL ||
        second.inputs.data == NULL || second.targets.data == NULL)
    {
        printf("FAILED: Could not load datasets.\n");

        dataset_free(&original);
        dataset_free(&first);
        dataset_free(&second);
        return 0;
    }

    if (!dataset_shuffle(&first, 12345) ||
        !dataset_shuffle(&second, 12345))
    {
        printf("FAILED: Shuffle operation failed.\n");

        dataset_free(&original);
        dataset_free(&first);
        dataset_free(&second);
        return 0;
    }

    if (!datasets_are_equal(&first, &second))
    {
        printf("FAILED: Same seed produced different shuffles.\n");

        dataset_free(&original);
        dataset_free(&first);
        dataset_free(&second);
        return 0;
    }

    /*
     * Ensure all original complete samples remain present, with
     * each input row still associated with its original target.
     */
    if (!dataset_has_same_samples(&original, &first))
    {
        printf("FAILED: Samples or input-target pairs were damaged.\n");

        dataset_free(&original);
        dataset_free(&first);
        dataset_free(&second);
        return 0;
    }

    /* Shuffling must not modify the untouched reference dataset. */
    if (original.inputs.data[0] != 1.0f ||
        original.inputs.data[1] != 2.0f ||
        original.targets.data[0] != 5.0f)
    {
        printf("FAILED: Original dataset was modified.\n");

        dataset_free(&original);
        dataset_free(&first);
        dataset_free(&second);
        return 0;
    }

    printf("Shuffled order:\n");

    for (size_t i = 0; i < first.sample_count; i++)
    {
        printf(
            "x1 = %.0f, x2 = %.0f, target = %.0f\n",
            first.inputs.data[i * first.input_size],
            first.inputs.data[i * first.input_size + 1],
            first.targets.data[i * first.output_size]);
    }

    printf("Same seed produced identical results.\n");
    printf("All complete input-target samples were preserved.\n");
    printf("Original dataset unchanged.\n");

    dataset_free(&original);
    dataset_free(&first);
    dataset_free(&second);

    printf("Dataset shuffle PASSED.\n\n");
    return 1;
}

static int test_dataset_split(void)
{
    printf("Test 2: Dataset splitting\n");
    printf("-------------------------\n");

    Dataset original = dataset_load_csv("data/linear.csv", 2, 1, 1);

    if (original.inputs.data == NULL || original.targets.data == NULL)
    {
        printf("FAILED: Could not load dataset.\n");
        dataset_free(&original);
        return 0;
    }

    Dataset training = {0};
    Dataset validation = {0};
    Dataset testing = {0};

    /*
     * 50% training, 25% validation, 25% testing.
     *
     * Four samples produce:
     * Training   = 2
     * Validation = 1
     * Testing    = 1
     */
    if (!dataset_split(
            &original,
            0.50f,
            0.25f,
            54321,
            &training,
            &validation,
            &testing))
    {
        printf("FAILED: Dataset split operation failed.\n");

        dataset_free(&original);
        dataset_free(&training);
        dataset_free(&validation);
        dataset_free(&testing);
        return 0;
    }

    printf("Split sizes:\n");
    printf("Training:   %zu\n", training.sample_count);
    printf("Validation: %zu\n", validation.sample_count);
    printf("Testing:    %zu\n", testing.sample_count);

    if (training.sample_count != 2 ||
        validation.sample_count != 1 ||
        testing.sample_count != 1)
    {
        printf("FAILED: Incorrect split sizes.\n");
        goto failure;
    }

    /*
     * Check complete input-target rows across all three subsets.
     * This detects lost, duplicated, or damaged samples.
     */
    if (!check_split_coverage(
            &original,
            &training,
            &validation,
            &testing))
    {
        printf("FAILED: Samples were lost, duplicated, or damaged.\n");
        goto failure;
    }

    /* Verify that splitting did not modify the source dataset. */
    if (original.inputs.data[0] != 1.0f ||
        original.inputs.data[1] != 2.0f ||
        original.targets.data[0] != 5.0f)
    {
        printf("FAILED: Original dataset was modified.\n");
        goto failure;
    }

    printf("All complete samples preserved exactly once.\n");
    printf("Input-target pairs preserved.\n");
    printf("Original dataset unchanged.\n");

    dataset_free(&original);
    dataset_free(&training);
    dataset_free(&validation);
    dataset_free(&testing);

    printf("Dataset split PASSED.\n\n");
    return 1;

failure:

    dataset_free(&original);
    dataset_free(&training);
    dataset_free(&validation);
    dataset_free(&testing);

    return 0;
}

int main(void)
{
    printf("ZENITH-ASR Dataset Split Tests\n");
    printf("==============================\n\n");

    if (!test_dataset_shuffle())
        return 1;

    if (!test_dataset_split())
        return 1;

    printf("ALL DATASET SPLIT TESTS PASSED!\n");
    printf("All allocated resources released.\n");

    return 0;
}
