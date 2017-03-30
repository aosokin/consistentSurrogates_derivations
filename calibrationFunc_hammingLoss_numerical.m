%calibrationFunc_hammingLoss_numerical numerically computes the calibration function for the Hamming loss on binary variables (Figure 1a, red).
%
% If you find this code useful, please, cite the following paper:
% On Structured Prediction Theory with Calibrated Convex Surrogate Losses
% Anton Osokin, Francis Bach, Simon Lacoste-Julien
% arXiv:1703.02403v1, 2017

%% initial definitions
numBits = 5;
numLabels = 2 ^ numBits;
epsValues = sort([0.01 : 0.05 : 1, 1, 0 : 1 / numBits : 1]);
calibrationFunction = nan( numel( epsValues ), numBits );

% variables of the optimization problem
numProbs = numLabels;
numScores = numLabels;

% the loss matrix: Hamming loss
addpath('helpers')
L = makeHammingLoss( numBits );

%% start computations
for iEpsValue = 1 : numel( epsValues )
    eps = epsValues( iEpsValue );
    fprintf('Computing for eps = %f (%d of %d)\n', eps, iEpsValue, numel(epsValues));

    % To break symmetries, we need to consider all the layers of the Boolean cube.
    calibrationFunction(iEpsValue) = inf;

    for numDiffBits = 1 : numBits
        % Case of numDiffBits different bits
        i = convertLabelingVectorToIndex( zeros(numBits, 1), numBits );  % corresponds to i in Eq. (20)
        jLabel = zeros(numBits, 1);
        jLabel(numBits - numDiffBits + 1 : end) = 1;
        j = convertLabelingVectorToIndex(jLabel, numBits ); % corresponds to j in Eq. (20)
    
        calibrationFunction(iEpsValue, numDiffBits) = computeCalibrationFunc_symmetriesBroken( L, i, j, eps );
    end
end

%% plot the numerical solution
figure(1);
clf;
hold on;
legendStr = {};
for numDiffBits = 1 : numBits
    plot( epsValues, calibrationFunction(:, numDiffBits) );
    legendStr{end + 1} = ['Numerical solution, diff in ', num2str(numDiffBits), ' bits'];
end
legend( legendStr );

