%calibrationFunc_mixedLoss_constrScores_symbolic derives the calibration function for the mixed loss (0-1 and block 0-1)
% when the scores are constrained to be equals inside the block.
% See Proposition 15 of our paper for the formal proof.
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
% In this function, we assume that all the scores within one block are constrained to be equal.
%
% The resulting problem is convex.
% Because of the convexity we can assume that all the scores within each block (except the too selected are equal).
% We use the follwoing notation for the scores:
%   f1 - scores of the first block (containing i)
%   f2 - scores of the second block (containing j)
%   Q1 - joint probability mass of the first block
%   Q2 - joint probability mass of the second block
% In the proof of Proposition 15, we show that probabilities of all the other blocks can be safely set to zero.

syms Q1 Q2
syms f1 f2
syms s k eps eta
syms f

% some assumptions to allow MATLAB not to consider weird cases
assume( 0 <= eta & eta <= 1 );
assume( s >= 1 );
assume( k >= 1);

% overall objective
obj_initial = s/2*( ...
    (f1 +(1-eta)*(1-Q1) + eta - eta*Q1/s)^2 + ...
    (f2 +(1-eta)*(1-Q2) + eta - eta*Q2/s)^2 ...
    );
obj_initial = obj_initial / k;

% according to observations all the probability mass is concentrated on the selected labels
obj = obj_initial;
obj = subs( obj, Q2, (1-eps)/(2-eta) );  % good for eps >= eta/2
obj = subs( obj, Q1, (1+eps-eta)/(2-eta) );

% scores on the selected objects are always equal: assign them to f
obj = subs( obj, f1, f);
obj = subs( obj, f2, f);

% do unconstrained minimization w.r.t. f:
f_star = solve(diff(obj, f) == 0, f);
% f_star = -(s - eta + eta*s)/(2*s)

final_obj = simplify(subs(obj, f, f_star));
% final_obj = ((2*eps - eta)^2*(eta + s - eta*s)^2)/(4*k*s*(eta - 2)^2)
fprintf('We get H(eps) = %s\n', char(final_obj) );


