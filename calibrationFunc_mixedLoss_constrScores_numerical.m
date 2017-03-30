%calibrationFunc_mixedLoss_constrScores_numerical numerically computes the calibration function for the mixed loss (0-1 and block 0-1)
% when the scores are constrained to be equals inside the block.
% The formal derivation is provided in Proposition 15.
% The derivation is also illustrated in calibrationFunc_mixedLoss_constrScores_symbolic.m
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
F = 1 - unique(L_block01, 'rows')';

%% start computations
for iEpsValue = 1 : numel( epsValues )
    eps = epsValues( iEpsValue );
    fprintf('Computing for eps = %f (%d of %d)\n', eps, iEpsValue, numel(epsValues));

    % The mixed loss is symmetric w.r.t. permutations of labels within a block.
    % When the blocks are of the same size we need to consider only one case (see the proof of proposition 15): labels in different blocks.
    i = 1;  % corresponds to i in Eq. (49)
    j = blockSize + 1;  % corresponds to j in Eq. (49)

    % Run the computation
    calibrationFunction(iEpsValue) = computeCalibrationFunc_symmetriesBroken( L, i, j, eps, F  );
end

%% analitical solution
% formula from Proposition 15
analitycFunc = @(eps) ( (eps-eta/2).^2 / (4*numBlocks) * (eta*numBlocks/numLabels + 1 - eta).^2 / (1 - eta/2).^2) .* (eps > eta / 2);

%% plot the analytical and numerical solutions
figure(1);
clf;
hold on;
plot( epsValues, analitycFunc(epsValues) , 'r' );
legendStr = {'Analytical solution'};

plot( epsValues, calibrationFunction, 'b' );
legendStr{end + 1} = 'Numerical solution';

legend( legendStr );

