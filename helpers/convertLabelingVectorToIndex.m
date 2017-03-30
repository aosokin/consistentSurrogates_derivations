function iLabel = convertLabelingVectorToIndex( vectorLabel, numBits, numValues )
%convertLabelingVectorToIndex is a helper function for dealing with the Hamming loss

if ~exist('numValues', 'var') || isempty(numValues)
    numValues = 2;
end

iLabel = 1 + sum( numValues .^ (numBits-1:-1:0)' .* vectorLabel(:) );

end

