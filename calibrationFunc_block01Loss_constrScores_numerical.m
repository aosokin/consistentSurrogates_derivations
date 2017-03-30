%calibrationFunc_block01Loss_constrScores_numerical numerically computes the calibration function for the block 0-1 loss
% when the scores are constrained to be equal within each block.
% The formal derivation is provided in Proposition 12: H(eps) = eps^2 /(4 * b), where b is the number fo blocks.
%
% If you find this code useful, please, cite the following paper:
% On Structured Prediction Theory with Calibrated Convex Surrogate Losses
% Anton Osokin, Francis Bach, Simon Lacoste-Julien
% arXiv:1703.02403v1, 2017

%% initial definitions
numBlocks = 3;
blockSize = 4;
numLabels = blockSize * numBlocks;
epsValues = [0.01 : 0.05 : 1, 1];
calibrationFunction = nan( numel( epsValues ), 1 );

% variables of the optimization problem
numProbs = numLabels;
numScores = numLabels;

% the loss matrix: block 0-1 loss
addpath('helpers')
L = makeBlock01Loss( numLabels, numBlocks );
F = 1 - unique(L, 'rows')';

%% start computations
for iEpsValue = 1 : numel( epsValues )
    eps = epsValues( iEpsValue );
    fprintf('Computing for eps = %f (%d of %d)\n', eps, iEpsValue, numel(epsValues));
    
    % The block 0-1 loss is symmetric w.r.t. permutations of labels within a block.
    % When the blocks are of the same size, we can consider only one pair of labels.
    i = 1;  % corresponds to i in Eq. (33)
    j = blockSize + 1;  % corresponds to j in Eq. (33)
    
     % Run the computation
    calibrationFunction(iEpsValue) = computeCalibrationFunc_symmetriesBroken( L, i, j, eps, F );
end

%% analitical solution
analitycFunc = @(eps) eps .^ 2 / (4 * numBlocks);

%% plot the analytical and numerical solutions
figure(1);
clf;
hold on;
plot( epsValues, analitycFunc(epsValues) , 'r' );
legendStr = {'Analytical solution'};

plot( epsValues, calibrationFunction, 'b' );
legendStr{end + 1} = 'Numerical solution';

legend( legendStr );

