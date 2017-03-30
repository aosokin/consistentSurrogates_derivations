function F = makeF_separableScores( numBits )
%makeF_separableScores is a helper function to compute calibration function for the Hamming loss in the case of separable scores
% This function implements F defined in Section 5 and Appendix E.

numLabels = 2 ^ numBits;
labelCodes = nan(numLabels, numBits);
for iLabel = 1 : numLabels
    labelCodes(iLabel, :) = convertLabelingIndexToVector(iLabel, numBits);
end

F = nan(numLabels, numBits + 1);
F(:, 1) = 0.5;
for iBit = 1 : numBits
    F(:, 1 + iBit) = labelCodes(:, iBit);
end

end

