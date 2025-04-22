## Overview

I had fun playing with this for NLP course! although I had to do it in mat language which is definetly not my area – the pseudocode say it all.

This system uses machine learning techniques to decode brain activity patterns (fMRI data) associated with different types of concepts, specifically differentiating between abstract and concrete semantic representations. 

## Scripts

### 1. `learnDecoder_concrete:abstract.m`

This function learns decoders for semantic vectors from brain imaging data, treating abstract and concrete concepts separately.

**Inputs:**
- `trainingData_abstract`: Matrix of brain activity patterns for abstract concepts (#examples × #voxels)
- `trainingTargets_abstract`: Matrix of semantic vector targets for abstract concepts (#examples × #dimensions)
- `trainingData_concrete`: Matrix of brain activity patterns for concrete concepts (#examples × #voxels)
- `trainingTargets_concrete`: Matrix of semantic vector targets for concrete concepts (#examples × #dimensions)

**Outputs:**
- `weightMatrix_abstract`: Weight matrix for abstract concepts (#voxels+1 × #dimensions)
- `r_abstract`: Regularization parameter values for each dimension (abstract concepts)
- `weightMatrix_concrete`: Weight matrix for concrete concepts (#voxels+1 × #dimensions)
- `r_concrete`: Regularization parameter values for each dimension (concrete concepts)

**Key Features:**
- Uses kernel ridge regression with a linear kernel
- Implements efficient cross-validation within the training set
- Selects optimal regularization parameters for each semantic dimension
- Avoids large matrix inversions through SVD to handle full-brain voxel patterns

### 2. `trainVoxelwiseTargetPredictionAbstractConcrete.m`

This function trains voxelwise prediction models to decode target features from brain activity, separately for abstract and concrete concepts.

**Inputs:**
- Brain data (examples) and semantic features (targets) for both abstract and concrete concepts
- Weight matrices and regularization parameters from `learnDecoder`
- Various optional parameters for customization

**Outputs:**
- `scores_abstract`: Prediction performance scores for abstract concepts
- `scores_concrete`: Prediction performance scores for concrete concepts

**Key Features:**
- Uses 10-fold cross-validation by default
- Z-score normalization for targets and centering for examples
- Voxelwise beta calculation using regularized regression
- Performance evaluation using correlation between predicted and actual values

### 3. `pseudo_voxelwise.m`

A pseudocode outline of the voxelwise target prediction function, providing a high-level overview of the algorithm structure.

## Technical Details

- **Regularization**: The system uses ridge regression with cross-validation to select optimal regularization parameters, balancing model fit and generalization.
- **Cross-validation**: Default 10-fold cross-validation splits the data to evaluate prediction performance.
- **Kernel Trick**: Uses the kernel trick to efficiently handle high-dimensional voxel data.
- **Performance Measure**: Correlation between predicted and actual semantic features serves as the performance metric.

## Usage

Example workflow:

```
matlab
% Load your fMRI data and semantic vectors
load('fMRI_data.mat');
load('semantic_vectors.mat');

% Learn decoders
[weightMatrix_abstract, r_abstract, weightMatrix_concrete, r_concrete] = learnDecoder(trainingData_abstract, trainingTargets_abstract, trainingData_concrete, trainingTargets_concrete);

% Train voxelwise prediction models and evaluate
[scores_abstract, scores_concrete] = trainVoxelwiseTargetPredictionModels(examples_abstract, targets_abstract, examples_concrete, targets_concrete, weightMatrix_abstract, weightMatrix_concrete, r_abstract, r_concrete);

% Analyze results
analyzeResults(scores_abstract, scores_concrete);
```
