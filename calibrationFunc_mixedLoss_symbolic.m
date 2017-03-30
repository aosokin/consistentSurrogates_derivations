%calibrationFunc_mixedLoss_symbolic derives the calibration function for the mixed loss (0-1 and block 0-1)
% when there are no constraints on the scores.
% See Proposition 14 of our paper for the formal proof.
%
% On Structured Prediction Theory with Calibrated Convex Surrogate Losses
% Anton Osokin, Francis Bach, Simon Lacoste-Julien
% arXiv:1703.02403v1, 2017
%
% We have:
%   k - number of labels
%   b - number of blocks
%   s - size of the block (all blocks of the same size, so k == d*s)
%   k phi(f, q) = q^T L^T f + 0.5 f^T R f      - expected objective
%   k delta phi(f, q) = k ( phi(f, q) - \min_f' phi(f', q)  ) = 0.5 || f + L q ||_2^2
%   L \in R^{k*k} - the loss matrix
%   L = eta*L_01 + (1-eta)*L_01block         - block-01-loss (higly structured) + noise of unstructured 01-loss
%   \delta ell(f, q) = {loss induced by scores f} - {best possible loss for probability q} = (Lq)_j - (Lq)_i
%
% We assume that symmetry is already broken, i.e. 
%   index i corresponds to the first label of the first block
%   index j corresponds to the first label of the second block
%
% The resulting problem is convex.
% Because of the convexity we can assume that all the scores within each block (except the too selected) are equal.
% We use the follwoing notation for the scores:
%   f1_1 - first block and first label (corresponding to i)
%   f1_2 - first block and all the other labels 
%   f2_1 - second block and first label (corresponding to j)
%   f2_2 - second block and all the other labels 
% q1_1 q1_2 q2_1 q2_2 denote the corresponding probabilities.
% We can show that probabilities of all the other blocks (except the first two can be safely set to zero).


syms q1_1 q1_2 q2_1 q2_2
syms f1_1 f1_2 f2_1 f2_2
syms s k b eps eta
syms f

% some assumptions to allow MATLAB not to consider weird cases
assume( 0 <= eta & eta <= 1 );
assume( s >= 1 );
assume( k >= 1);

% probability masses of the first two blocks
Q_1 = q1_1 + (s-1) * q1_2;
Q_2 = q2_1 + (s-1) * q2_2;

% overall objective
obj_initial = 0.5*( ...
    (f1_1 + eta*(1-q1_1) + (1-eta)*(1-Q_1))^2 + ...
    (s-1)*(f1_2 + eta*(1-q1_2) + (1-eta)*(1-Q_1))^2 + ...
    (f2_1 + eta*(1-q2_1) + (1-eta)*(1-Q_2))^2 + ...
    (s-1)*(f2_2 + eta*(1-q2_2) + (1-eta)*(1-Q_2))^2 ...
    );
obj_initial = obj_initial / k;

% according to observations all the probability mass is concentrated on the selected labels
obj = obj_initial;
obj = subs( obj, q1_1, 0.5 + eps/2);
obj = subs( obj, q1_2, 0);
obj = subs( obj, q2_1, 0.5 - eps/2);
obj = subs( obj, q2_2, 0);

% scores on the selected objects are always equal: assign them to f
obj = subs( obj, f1_1, f);
obj = subs( obj, f2_1, f);

% remaining variables: f, f1_2, f2_2
% try to do unconstrained minimization w.r.t. f, f1_2, f2_2

f_star = solve(diff(obj, f) == 0, f);
% f_star = -1/2

f1_2_star = solve(diff(obj, f1_2) == 0, f1_2); 
% f1_2_star = -eta-(eps/2 - 1/2)*(eta - 1)
% f1_2_star <= f_star, when 0<=eta<=1 and 0<=eps<=eta/(1-eta)

f2_2_star = solve(diff(obj, f2_2) == 0, f2_2); 
% f2_2_star = (eps/2 + 1/2)*(eta - 1) - eta
% f2_2_star <= f_star, always true when 0<=eta<=1 and 0<=eps<=1

% Two cases: 1) 0<=eps<=eta/(1-eta); 2) eta/(1-eta)<eps<=1
% Case 1: 0<=eps<=eta/(1-eta)
% all constrainst are satisfied, can minimize w.r.t. all f independently

obj_case1 = obj;
obj_case1 = subs( obj_case1, f, f_star );
obj_case1 = subs( obj_case1, f1_2, f1_2_star );
obj_case1 = subs( obj_case1, f2_2, f2_2_star );

% as a result we get H(eps) = eps^2/4/k
fprintf('Case 1: 0<=eps<=eta. We get H(eps) = %s\n', char(simplify(obj_case1)));

% Case 2: eta/(1-eta)<eps<=1
% f2_2_star > f_star, which means we have to assign f2_2_star to f
obj_case2 = obj;
obj_case2 = subs( obj_case2, f1_2, f);

f_star = solve(diff(obj_case2, f) == 0, f); % f_star = -(eps - eta + s - eps*eta - eps*s + eta*s + eps*eta*s + 1)/(2*s + 2)
f2_2_star = solve(diff(obj_case2, f2_2) == 0, f2_2); % f2_2_star = (eps/2 + 1/2)*(eta - 1) - eta

% expression (f_star - f2_2_star) is linear in eps
% derivative diff( f_star - f2_2_star, eps) equals (s-1)(1-eta)/(2*s + 2) + (1-eta)/2 which is >= 0 for 0<=eta<=1
% At eps == 0, we have simplify(subs(f_star - f2_2_star, eps, 0)) == eta/(s + 1), which is greater than 0
% As a conclusion, f_star - f2_2_star >= 0 always holds (in the setting of Case 2)
% Now, we can substitute f_star and f2_2_star into the objective:
obj_case2 = subs( obj_case2, f, f_star );
obj_case2 = subs( obj_case2, f2_2, f2_2_star );
obj_case2 = simplify(obj_case2);

manual_rewriting = eps^2*s/(2*k*(s+1)) - eta*(eps+1)*(s-1)/(4*k*(s+1))*(eps + (eps - eps*eta - eta));

if simplify( manual_rewriting - obj_case2) ~= 0
    error(['Incorrect rewriting of formula: ', char(simplify(obj_case2))]);
end

fprintf('Case 2: eta/(1-eta)<eps<=1. We get H(eps) = %s\n', char(manual_rewriting) );



