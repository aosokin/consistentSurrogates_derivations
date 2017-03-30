function vectorLabel = convertLabelingIndexToVector( iLabel, numBits, numValues )
%convertLabelingIndexToVector is a helper function for dealing with the Hamming loss

if ~exist('numValues', 'var') || isempty(numValues)
    numValues = 2;
end

vectorLabel = nan( numBits, 1 );
remainingPartOfLabel = iLabel - 1;
for iVar = 1 : numBits
    vectorLabel( numBits - iVar + 1) = mod( remainingPartOfLabel, numValues );
    remainingPartOfLabel = ( remainingPartOfLabel - vectorLabel( numBits - iVar + 1) ) / numValues;
end

end

