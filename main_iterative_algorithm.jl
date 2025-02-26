include("building_tree.jl")
include("utilities.jl")
include("merge.jl")
include("main_merge.jl")
include("shift.jl")

function main_iterative()
    for dataSetName in ["iris", "seeds", "wine"]
        
        print("=== Dataset ", dataSetName)
        
        # Préparation des données
        include("../data/" * dataSetName * ".txt")
        
        # Ramener chaque caractéristique sur [0, 1]
        reducedX = Matrix{Float64}(X)
        for j in 1:size(X, 2)
            reducedX[:, j] .-= minimum(X[:, j])
            reducedX[:, j] ./= maximum(X[:, j])
        end

        train, test = train_test_indexes(length(Y))
        X_train = reducedX[train,:]
        Y_train = Y[train]
        X_test = reducedX[test,:]
        Y_test = Y[test]
        classes = unique(Y)

        println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2), ")")
        
        # Temps limite de la méthode de résolution en secondes        
        time_limit = 30

        for D in 2:4
            println("\tD = ", D)
            println("\t\tUnivarié")
            println("\t\t\t- Unsplittable clusters (FU)")
            testMerge(X_train, Y_train, X_test, Y_test, D, classes, time_limit = time_limit, isMultivariate = false)
            println("\t\t\t- Iterative heuristic (FhS)")
            testIterative(X_train, Y_train, X_test, Y_test, D, classes, time_limit = time_limit, isMultivariate = false, isExact=false)
            println("\t\t\t- Iterative heuristic (FhS) with shifts")
            testIterative(X_train, Y_train, X_test, Y_test, D, classes, time_limit = time_limit, isMultivariate = false, isExact=false, shiftSeparations=true)
            println("\t\t\t- Iterative exact (FeS)")
            testIterative(X_train, Y_train, X_test, Y_test, D, classes, time_limit = time_limit, isMultivariate = false, isExact=true)

#            # Do not apply to the multivariate case in the project             
#            println("\t\tMultivarié")
#            println("\t\t\t- Unsplittable clusters (FU)")
#            testMerge(X_train, Y_train, X_test, Y_test, D, classes, time_limit = time_limit, isMultivariate = true)
#            println("\t\t\t- Iterative heuristic (FhS)")
#            testIterative(X_train, Y_train, X_test, Y_test, D, classes, time_limit = time_limit, isMultivariate = true, isExact=false)
#            println("\t\t\t- Iterative exact (FeS)")
#            testIterative(X_train, Y_train, X_test, Y_test, D, classes, time_limit = time_limit, isMultivariate = true, isExact=true)
        end
    end
end 

function testIterative(X_train, Y_train, X_test, Y_test, D, classes; time_limit::Int=-1, isMultivariate::Bool = false, isExact::Bool=false, shiftSeparations::Bool=false)

    # Pour tout pourcentage de regroupement considéré
    println("\t\t\tGamma\t\t# clusters\tGap")
    for gamma in 0:0.2:0.8
        print("\t\t\t", gamma * 100, "%\t\t")
        clusters = simpleMerge(X_train, Y_train, gamma)
        print(length(clusters), " clusters\t")
        T, obj, resolution_time, gap, iterationCount = iteratively_build_tree(clusters, D, X_train, Y_train, classes, multivariate = isMultivariate, time_limit = time_limit, isExact=isExact, shiftSeparations = shiftSeparations)
        if gap == -1
            print("???%\t")
        else 
            print(round(gap, digits = 1), "%\t")
        end 
        print("Erreurs train/test : ", prediction_errors(T,X_train,Y_train, classes))
        print("/", prediction_errors(T,X_test,Y_test, classes), "\t")
        print(round(resolution_time, digits=1), "s\t")
        println(iterationCount, " iterations")

        if gap == -1
            println("Warning: there is no gap since when the time limit has been reached at the last iteration before CPLEX had found no feasible solution")
        end 
    end
    println() 
end 
