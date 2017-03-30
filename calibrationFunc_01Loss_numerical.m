%calibrationFunc_01loss_numerical numerically computes the calibration function for the 0-1 loss.
% The formal derivation is provided in Proposition 10: H(eps) = eps^2 /(4k), where k is the number of labels
%
% If you find this code useful, please, cite the following paper:
% On Structured Prediction Theory with Calibrated Convex Surrogate Losses
% Anton Osokin, Francis Bach, Simon Lacoste-Julien
% arXiv:1703.02403v1, 2017

%% initial definitions
numLabels = 8;
epsValues = [0.01 : 0.05 : 1, 1];
calibrationFunction = nan( numel( epsValues ), 1 );

% variables of the optimization problem
numProbs = numLabels;
numScores = numLabels;

% the loss matrix: 0-1 loss
L = ones(numLabels) - eye(numLabels);

%% start computations
for iEpsValue = 1 : numel( epsValues )
    eps = epsValues( iEpsValue );
    fprintf('Computing for eps = %f (%d of %d)\n', eps, iEpsValue, numel(epsValues));
    
    % The 0-1 loss is symmetric w.r.t. all labels, so we can consider only one pair of labels
    i = 1;  % corresponds to i in Eq. (31)
    j = 2;  % corresponds to j in Eq. (31)
    
    % Run the computation
    calibrationFunction(iEpsValue) = computeCalibrationFunc_symmetriesBroken( L, i, j, eps );
end

%% analitical solution
analitycFunc = @(eps) eps .^ 2 / (4 * numLabels);

%% plot the analytical and numerical solutions
figure(1);
clf;
hold on;
plot( epsValues, analitycFunc(epsValues) , 'r' );
legendStr = {'Analytical solution'};

plot( epsValues, calibrationFunction, 'b' );
legendStr{end + 1} = 'Numerical solution';

legend( legendStr );

