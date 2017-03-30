function calibrationFunction = computeCalibrationFunc_symmetriesBroken( L, i, j, eps, F )
%computeCalibrationFunc_symmetriesBroken computes the calibration function with symmetries broken 
% by solving the optimization problem (20) of the following paper:
% On Structured Prediction Theory with Calibrated Convex Surrogate Losses
% Anton Osokin, Francis Bach, Simon Lacoste-Julien
% arXiv:1703.02403v1, 2017

if ~exist('F', 'var') || isempty(F)
    F = eye( size(L, 1) );
end

% get the number of possible ground-truth labels and number of parameters of the score function
numProbs = size(L, 2); % ground-truth labels
numPredictions = size(L, 1); % predictions
numScoreParams = size(F, 2); % score parameters

% check that the number of predictions is consistent
assert( size(L, 1) == size(F, 1) )

% we are going to compute everything using quadprog
% [x, fval] = quadprog( H, f, A, b, Aeq, beq, lb, ub);
% we will need to define H, f, A, b, Aeq, beq, lb, ub

% total numer of variables
numVars = numProbs + numScoreParams;

% init everything.
% order of variables: q_1, ..., q_k, theta_1, ..., theta_k
Aeq = zeros( 0, numVars );
A = zeros( 0, numVars );
beq = zeros(0, 1);
b = zeros(0, 1);

% simplex constraint
Aeq = [Aeq; ones( 1, numProbs ), zeros( 1, numScoreParams )]; beq = [beq; 1];  % \sum_k q_k = 1
lb = [ zeros( numProbs, 1 ); -100 * ones( numScoreParams, 1 ) ];  % put some bounds of scores to avoid numerical blow up
ub = [ ones( numProbs, 1 ); 100 * ones( numScoreParams, 1 ) ];

% inequalities showing that label i is the best prediction
% (Lq)_i <= (Lq)_c
for c = 1 : numScoreParams
    newConstr = zeros( 1, numVars );
    newConstr( 1 : numProbs ) = L(i, :) - L(c, :);
    A = [A; newConstr ]; b = [b; 0];
end

% f_j is the maximum value : f_j >= f_c
for c = 1 : numScoreParams
    newConstr = zeros( 1, numVars );
    newConstr( numProbs + 1 : numVars) = F(c,:) - F(j,:); % (F * theta)_j >= (F * theta)_c

    A = [A; newConstr ]; b = [b; 0];
end

% there is epsilon difference in the losses: (Lq)_j - (Lq)_i >= eps
newConstr = zeros( 1, numVars );
newConstr( 1 : numProbs ) = L(i, :) - L(j, :);
A = [A; newConstr ]; b = [b; -eps];

% objective function
% delta W = || (F'*F)^0.5*theta + (F'*F)^(-0.5)*F'*L*q ||_2^2
H_theta_theta = F'*F;
H_q_theta = L'*F;
H_q_q = (L'*F)*((F'*F)\(F'*L));

H = [ H_q_q,   H_q_theta ; ...
      H_q_theta', H_theta_theta ] / numPredictions;
H = (H + H') / 2;
f = zeros( numVars, 1 );

[x, curFuncValue, exitflag, output, lambda ] = quadprog( H, f, A, b, Aeq, beq, lb, ub, [], ...
    optimoptions( 'quadprog', 'Algorithm', 'interior-point-convex', 'MaxIter', 1000, 'Display', 'off' ) );
if exitflag == -2  % infeasible problem
    calibrationFunction = inf;
else
    calibrationFunction = curFuncValue;
end


end

