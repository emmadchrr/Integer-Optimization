"""
Naive function that tries to shift the value of each separation of an univariate tree in order to improve its predictions

Input:
- T: the tree
- x,y: the training set
- classes: the different classes in the data
- clusters: the clusters

Output
- newT: a tree in which the value of 
"""
function naivelyShiftSeparations(T::Tree, x::Matrix{Float64}, y::Vector{}, classes::Vector{}, clusters::Vector{Cluster})

    newT = Tree()
    newT.a = Matrix{Float64}(copy(T.a))
    newT.b = Vector{Float64}(copy(T.b))
    newT.c = Vector{Int64}(copy(T.c))
    
    # For each separation
    for t in 1:length(T.b)

        # If the node performs a split
        if T.c[t] != -1

            # Get the number of correctly predicted data from the training set
            bestPrediction = prediction_errors(newT, x, y, classes)
            bestSplitValue = newT.b[t]

#                splitCount = getClusterSplitCount(newT, clusters)
#            @show bestSplitValue, bestPrediction, splitCount

            # Try to shift the split
            for currentSplitValue in 0:0.1:1

                # Change the split
                newT.b[t] = currentSplitValue

                # Get the number of correctly predicted data after the change
                currentPrediction = prediction_errors(newT, x, y, classes)

#                splitCount = getClusterSplitCount(newT, clusters)
#                @show currentSplitValue, currentPrediction, splitCount

                if currentPrediction < bestPrediction
                    println("Good predictions improved from ", bestPrediction, " to ", currentPrediction, " by shifting the split of node ", t , " from ", bestSplitValue,  " to ", currentSplitValue)
                    bestSplitValue = currentSplitValue
                    bestPrediction = currentPrediction
                end 
            end

            # Set back the best split value
            newT.b[t] = bestSplitValue

#            readline()
        end
    end
    
    return newT
end 
