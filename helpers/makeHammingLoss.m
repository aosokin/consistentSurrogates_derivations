function lossMatrix = makeHammingLoss( numBits, numValues )
%makeHammingLoss constructs the loss matrix of the Hamming loss on (numBits) varaibles taking (numValues, default: 2) values each

if ~exist('numValues', 'var') || isempty(numValues)
    numValues = 2;
end

lossMatrix = ones( numValues ) - eye(numValues);
for iVar = 2 : 1 : numBits
    curLoss = repmat( lossMatrix + 1, numValues, numValues);
    for iValue = 1 : numValues
        
        curLoss( (iValue - 1) * size(lossMatrix, 1) + ( 1 : size(lossMatrix, 1) ), ...
                 (iValue - 1) * size(lossMatrix, 2) + ( 1 : size(lossMatrix, 2) ) ) = lossMatrix;
    end
    
    lossMatrix = curLoss;
end
lossMatrix = lossMatrix / numBits;

end

