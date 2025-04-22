function [scores_abstract, scores_concrete] = trainVoxelwiseTargetPredictionModels( ...
    examples_abstract, targets_abstract, examples_concrete, targets_concrete, ...
    weightMatrix_abstract, weightMatrix_concrete, r_abstract, r_concrete, varargin)

% Process parameters
if nargin < 4
    fprintf('syntax: trainVoxelwiseTargetPredictionModels(<examples_abstract>,<targets_abstract>,<examples_concrete>,<targets_concrete>,<weightMatrix_abstract>,<weightMatrix_concrete>,<r_abstract>,<r_concrete>,\[optional\])\n'); return;
end

examples_a = examples_abstract; [n2a, ma] = size(examples_a);
targets_a  = targets_abstract; [n1a, mta] = size(targets_a);
if n1a ~= n2a
    fprintf('error: targets_abstract must have as many rows as examples_abstract\n'); return;
end
na = n1a;

examples_c = examples_concrete; [n2c, mc] = size(examples_c);
targets_c  = targets_concrete; [n1c, mtc] = size(targets_c);
if n1c ~= n2c
    fprintf('error: targets_concrete must have as many rows as examples_concrete\n'); return;
end
nc = n1c;

% Initialize variables for storing results
scores_abstract = zeros(mta, ma); % Preallocate for abstract concepts
scores_concrete = zeros(mtc, mc); % Preallocate for concrete concepts

% Defaults
meta   = [];
lambda = 1;
useCorrelation = 0;
useOptimization= 0;
voxelMask = ones(1, max(ma, mc));
% 10-fold cross-validation
labelsGroup = 1+rem((1:na)',10);

idx = 9; % Adjust index to start after the fixed inputs
while idx <= nargin
    argval = varargin{idx}; idx = idx + 1;
    switch argval
      case {'meta'}
        meta = varargin{idx}; idx = idx + 1;
      case {'groupby'}
        labelsGroup = varargin{idx}; idx = idx + 1;
      case {'lambda'}
        lambda = varargin{idx}; idx = idx + 1;
      case {'voxelMask'}
        voxelMask = varargin{idx}; idx = idx + 1;
      case {'useOptimization'}
        useOptimization = varargin{idx}; idx = idx + 1;
      otherwise
        fprintf('error: unknown parameter %s\n',argval); return;
    end
end

% Main loop for abstract concepts
useSearchlight = ~isempty(meta);

% Precompute variables for abstract concepts
onecol    = ones(na,1);
targetsZ_a  = zscore(targets_a);
targetsC_a  = targets_a - repmat(mean(targets_a,1),na,1);
examplesZ_a = examples_a - repmat(mean(examples_a,1),na,1);
predicted_a = zeros(na, mta);

% Split data into cross-validation groups
groups_a = unique(labelsGroup); nGroups_a = length(groups_a);
for iga = 1:nGroups_a
    mask_a = (labelsGroup == groups_a(iga));
    indicesTest_a{iga}  = find( mask_a); nTest_a(iga)  = length(indicesTest_a{iga});
    indicesTrain_a{iga} = find(~mask_a); nTrain_a(iga) = length(indicesTrain_a{iga});
    targetsPerGroup_a{iga} = targets_a(indicesTrain_a{iga},:);
end

% Voxelwise loop for abstract concepts
for va = 1:ma
    if rem(va,1000) == 0
        fprintf('iter: %d\n',va)
    end
    if voxelMask(va)
        data_a = examplesZ_a(:,va);
        for iga = 1:nGroups_a
            tmp_a = data_a(indicesTrain_a{iga},:);
            betas_a = (tmp_a'*tmp_a + r_abstract(va))\(tmp_a'*targetsC_a(indicesTrain_a{iga},:));
            predicted_a(indicesTest_a{iga},:) = data_a(indicesTest_a{iga},:)*betas_a;
        end
        scores_abstract(:,va) = sum(targetsZ_a .* zscore(predicted_a),1)/(na-1);
    end
end

% Main loop for concrete concepts
labelsGroup = 1+rem((1:nc)',10);

% Precompute variables for concrete concepts
onecol    = ones(nc,1);
targetsZ_c  = zscore(targets_c);
targetsC_c  = targets_c - repmat(mean(targets_c,1),nc,1);
examplesZ_c = examples_c - repmat(mean(examples_c,1),nc,1);
predicted_c = zeros(nc, mtc);

% Split data into cross-validation groups
groups_c = unique(labelsGroup); nGroups_c = length(groups_c);
for igc = 1:nGroups_c
    mask_c = (labelsGroup == groups_c(igc));
    indicesTest_c{igc}  = find( mask_c); nTest_c(igc)  = length(indicesTest_c{igc});
    indicesTrain_c{igc} = find(~mask_c); nTrain_c(igc) = length(indicesTrain_c{igc});
    targetsPerGroup_c{igc} = targets_c(indicesTrain_c{igc},:);
end

% Voxelwise loop for concrete concepts
for vc = 1:mc
    if rem(vc,1000) == 0
        fprintf('iter: %d\n',vc)
    end
    if voxelMask(vc)
        data_c = examplesZ_c(:,vc);
        for igc = 1:nGroups_c
            tmp_c = data_c(indicesTrain_c{igc},:);
            betas_c = (tmp_c'*tmp_c + r_concrete(vc))\(tmp_c'*targetsC_c(indicesTrain_c{igc},:));
            predicted_c(indicesTest_c{igc},:) = data_c(indicesTest_c{igc},:)*betas_c;
        end
        scores_concrete(:,vc) = sum(targetsZ_c .* zscore(predicted_c),1)/(nc-1);
    end
end

end
