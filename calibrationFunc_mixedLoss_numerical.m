%calibrationFunc_mixedLoss_numerical numerically computes the calibration function for the mixed loss: 0-1 and block 0-1.
% The formal derivation is provided in Proposition 14.
% The derivation is also illustrated in calibrationFunc_mixedLoss_symbolic.m
%
% If you find this code useful, please, cite the following paper:
% On Structured Prediction Theory with Calibrated Convex Surrogate Losses
% Anton Osokin, Francis Bach, Simon Lacoste-Julien
% arXiv:1703.02403v1, 2017

%% initial definitions
numBlocks = 3;
blockSize = 4;
eta = 0.3;
numLabels = blockSize * numBlocks;
epsValues = [0.01 : 0.05 : 1, 1];
calibrationFunction = nan( numel( epsValues ), 1 );

% variables of the optimization problem
numProbs = numLabels;
numScores = numLabels;

% the loss matrix: mixed (0-1 and block 0-1) loss
addpath('helpers')
L_block01 = makeBlock01Loss( numLabels, numBlocks );
L_01 = ones(numLabels) - eye(numLabels);
L = eta * L_01 + (1 - eta) * L_block01;

%% start computations
for iEpsValue = 1 : numel( epsValues )
    eps = epsValues( iEpsValue );
    fprintf('Computing for eps = %f (%d of %d)\n', eps, iEpsValue, numel(epsValues));

    % The mixed loss is symmetric w.r.t. permutations of labels within a block.
    % When the blocks are of the same size we need to consider two cases: labels inside a block, labels in different blocks.

    % Case 1: same block
    if blockSize >= 2
        i = 1;  % corresponds to i in Eq. (47)
        j = 2;  % corresponds to j in Eq. (47)
    
        % Run the computation
        calibrationFunction(iEpsValue) = computeCalibrationFunc_symmetriesBroken( L, i, j, eps );
    else
        calibrationFunction(iEpsValue) = inf;
    end

    % Case 2: different blocks
    if numBlocks >= 2
        i = 1;  % corresponds to i in Eq. (47)
        j = blockSize + 1;  % corresponds to j in Eq. (47)
    
        % Run the computation
        calibrationFunction_thisCase = computeCalibrationFunc_symmetriesBroken( L, i, j, eps );
        calibrationFunction(iEpsValue) = min( calibrationFunction_thisCase, calibrationFunction(iEpsValue));
    end
end

%% analitical solution
% formula from Proposition 14
analitycFunc = @(eps) ( eps .^ 2 / (4 * numLabels) .* (eps <= eta / (1-eta)) + ...
    (eps .^ 2 * blockSize / (2 * numLabels * (blockSize + 1)) - eta.*(eps+1).*(blockSize-1).*(2*eps-eps*eta-eta) ./ (4*numLabels*(blockSize+1))) .* (eps > eta / (1-eta)) );

%% plot the analytical and numerical solutions
figure(1);
clf;
hold on;
plot( epsValues, analitycFunc(epsValues) , 'r' );
legendStr = {'Analytical solution'};

plot( epsValues, calibrationFunction, 'b' );
legendStr{end + 1} = 'Numerical solution';

legend( legendStr );

