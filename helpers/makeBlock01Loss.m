function L = makeBlock01Loss( numLabels, numBlocks )
%makeBlock01Loss constructs the loss matrix of the block 0-1 loss

blockIds = repmat( 1 : numBlocks, [ceil( numLabels / numBlocks ), 1] );
blockIds = blockIds(:);
blockIds = blockIds(1 : numLabels);

L = zeros(numLabels, numLabels);
for i = 1 : numLabels
    for j = 1 : numLabels
        if i ~= j
            if blockIds(i) ~= blockIds(j)
                L(i, j) = 1;
            end
        end
    end
end

end

