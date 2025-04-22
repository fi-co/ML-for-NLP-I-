function [scores_abstract, scores_concrete] = trainVoxelwiseTargetPredictionModels( ...
    examples_abstract, targets_abstract, examples_concrete, targets_concrete, ...
    weightMatrix_abstract, weightMatrix_concrete, r_abstract, r_concrete, varargin)

    Parse input parameters and check syntax.

    Split examples and targets into abstract and concrete categories.

    Set default parameters and parse optional input arguments.

    Initialize variables for storing results.

    --- Abstract Concepts ---
    Precompute necessary values for abstract concepts:
        - Standardize targets for z-score.
        - Center examples.
        - Initialize predicted values storage.

    Loop through each voxel for abstract concepts:
        - Apply cross-validation using groups defined by labelsGroup.
        - Compute betas using a numerically stable method.
        - Predict values for each test set and store predictions.
        - Calculate scores based on correlation between predicted and actual values.

    --- Concrete Concepts ---
    Update labelsGroup for concrete data.

    Precompute necessary values for concrete concepts:
        - Standardize targets for z-score.
        - Center examples.
        - Initialize predicted values storage.

    Loop through each voxel for concrete concepts:
        - Apply cross-validation using groups defined by labelsGroup.
        - Compute betas using a numerically stable method.
        - Predict values for each test set and store predictions.
        - Calculate scores based on correlation between predicted and actual values.

    Return scores_abstract and scores_concrete.
end
