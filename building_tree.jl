include("struct/tree.jl")

"""
Construit un arbre de décision par résolution de la formulation F

Entrées :
- x : caractéristiques des données d'entraînement
- y : classe des données d'entraînement
- D : Nombre maximal de séparations d'une branche (profondeur de l'arbre - 1)
- multivariate (optionnel): vrai si les séparations sont multivariées; faux si elles sont univariées (faux par défaut)
- mu (optionnel, utilisé en multivarié): distance minimale à gauche d'une séparation où aucune donnée ne peut se trouver (i.e., pour la séparation ax <= b, il n'y aura aucune donnée dans ]b - ax - mu, b - ax[) (10^-4 par défaut)
- time_limits (optionnel) : temps maximal de résolution (-1 si le temps n'est pas limité) (-1 par défaut)
- classes : labels des classes figurant dans le dataset
"""
function build_tree(x::Matrix{Float64}, y::Vector{}, D::Int64, classes; multivariate::Bool=false, time_limit::Int64 = -1, mu::Float64=10^(-4))
    
    dataCount = length(y) # Nombre de données d'entraînement
    featuresCount = length(x[1, :]) # Nombre de caractéristiques
    classCount = length(classes) # Nombre de classes différentes
    sepCount = 2^D - 1 # Nombre de séparations de l'arbre
    leavesCount = 2^D # Nombre de feuilles de l'arbre

    m = Model(CPLEX.Optimizer) 
    set_silent(m)

    if time_limit!=-1
        set_time_limit_sec(m, time_limit)
    end

    # Plus petite différence entre deux données pour une caractéristique
    mu_min = 1.0 
    # Plus grande différence entre deux données pour une caractéristique
    mu_max = 0.0
    
    if !multivariate # calcul des constantes mu_min, mu_max et du vecteur mu

        # mu_vect[j] est la plus petite différence (>0) entre deux données, pour la caractéristiques j
        mu_vect = ones(Float64, featuresCount)
        for j in 1:featuresCount
            for i1 in 1:dataCount
                for i2 in (i1+1):dataCount
                    if abs(x[i1, j] - x[i2, j]) > 1E-4
                        mu_vect[j] = min(mu_vect[j], abs(x[i1, j] - x[i2, j]))
                    end
                end
            end
            mu_min = min(mu_min, mu_vect[j])
            mu_max = max(mu_max, mu_vect[j])
        end
    end

    ## Déclaration des variables
    if multivariate
        @variable(m, a[1:featuresCount, 1:sepCount], base_name="a_{j, t}")
        @variable(m, a_h[1:featuresCount, 1:sepCount], base_name="â_{j, t}")
        @variable(m, s[1:featuresCount, 1:sepCount], Bin, base_name="s_{j, t}")
        @variable(m, d[1:sepCount], Bin, base_name="d_t")
    else
        @variable(m, a[1:featuresCount, 1:sepCount], Bin, base_name="a")
    end 
    @variable(m, b[1:sepCount], base_name="b_t")
    @variable(m, c[1:classCount, 1:(sepCount+leavesCount)], Bin, base_name = "c_{k, t}")
    @variable(m, u_at[1:dataCount, 1:(sepCount+leavesCount)], Bin, base_name = "u^i_{a(t), t}")
    @variable(m, u_tw[1:dataCount, 1:(sepCount+leavesCount)], Bin, base_name = "u^i_{t, w}")

    ## Déclaration des contraintes

    # contraintes définissant la structure de l'arbre
    if multivariate
        @constraint(m, [t in 1:sepCount], d[t] + sum(c[k, t] for k in 1:classCount) == 1) # on s'assure que le noeud applique une règle de branchement OU attribue une classe
        @constraint(m, [t in 1:sepCount], b[t] <= d[t]) # b doit être nul si il n'y a pas de branchement 
        @constraint(m, [t in 1:sepCount], b[t] >= -d[t]) # b doit être nul si il n'y a pas de branchement 
        @constraint(m, [t in 1:sepCount], sum(a_h[j, t] for j in 1:featuresCount) <= d[t]) # on borne la norme du vecteur a
        @constraint(m, [t in 1:sepCount, j in 1:featuresCount], a[j, t] <= a_h[j, t]) # définition de â borne sup de la valeur absolu de a
        @constraint(m, [t in 1:sepCount, j in 1:featuresCount], a[j, t] >= -a_h[j, t]) # définition de â borne sup de la valeur absolu de a
        @constraint(m, [t in 1:sepCount, j in 1:featuresCount], a[j, t] <= s[j, t]) # définition de s, non nul ssi a non nul
        @constraint(m, [t in 1:sepCount, j in 1:featuresCount], a[j, t] >= -s[j, t]) # définition de s, non nul ssi a non nul
        @constraint(m, [t in 1:sepCount, j in 1:featuresCount], s[j, t] <= d[t]) # définition de d, non nul si il existe un coef non nul
        @constraint(m, [t in 1:sepCount], sum(s[j, t] for j in 1:featuresCount) >= d[t]) # définition de d, non nul si il existe un coef non nul
        @constraint(m, [t in 2:sepCount], d[t] <= d[t ÷ 2]) # on s'assure que si un noeud de branchement n'applique pas de règle de branchement, ses fils non plus
    else
        @constraint(m, [t in 1:sepCount], sum(a[j, t] for j in 1:featuresCount) + sum(c[k, t] for k in 1:classCount) == 1) # on s'assure que le noeud applique une règle de branchement OU attribue une classe
        @constraint(m, [t in 1:sepCount], b[t] <= sum(a[j, t] for j in 1:featuresCount)) # b doit être nul si il n'y a pas de branchement 
        @constraint(m, [t in 1:sepCount], b[t] >= 0) # b doit être positif
    end
    @constraint(m, [t in (sepCount+1):(sepCount+leavesCount)], sum(c[k, t] for k in 1:classCount) == 1) # on s'assure qu'on attribue une classe par feuille

    # contraintes de conservation du flot et contraintes de capacité
    @constraint(m, [i in 1:dataCount, t in 1:sepCount], u_at[i, t] == u_at[i, t*2] + u_at[i, t*2+1] + u_tw[i, t]) # conservation du flot dans les noeuds de branchement
    @constraint(m, [i in 1:dataCount, t in (sepCount+1):(sepCount+leavesCount)], u_at[i, t] == u_tw[i, t]) # conservation du flot dans les feuilles
    @constraint(m, [i in 1:dataCount, t in 1:(sepCount+leavesCount)], u_tw[i, t] <= c[findfirst(classes .== y[i]), t]) # contrainte de capacité qui impose le flot a etre nul si la classe de la feuille n'est pas la bonne
    if multivariate
        @constraint(m, [i in 1:dataCount, t in 1:sepCount], sum(a[j, t]*x[i, j] for j in 1:featuresCount) + mu <= b[t] + (2+mu)*(1-u_at[i, t*2])) # contrainte de capacité controlant le passage dans le noeud fils gauche
        @constraint(m, [i in 1:dataCount, t in 1:sepCount], sum(a[j, t]*x[i, j] for j in 1:featuresCount) >= b[t] - 2*(1-u_at[i, t*2 + 1])) # contrainte de capacité controlant le passage dans le noeud fils droit
        @constraint(m, [i in 1:dataCount, t in 1:sepCount], u_at[i, t*2+1] <= d[t]) # contrainte de capacité empechant les données de passer dans le fils droit d'un noeud n'appliquant pas de règle de branchement
    else
        @constraint(m, [i in 1:dataCount, t in 1:sepCount], sum(a[j, t]*(x[i, j]+mu_vect[j]-mu_min) for j in 1:featuresCount) + mu_min <= b[t] + (1+mu_max)*(1-u_at[i, t*2])) # contrainte de capacité controlant le passage dans le noeud fils gauche
        @constraint(m, [i in 1:dataCount, t in 1:sepCount], sum(a[j, t]*x[i, j] for j in 1:featuresCount) >= b[t] - (1-u_at[i, t*2 + 1])) # contrainte de capacité controlant le passage dans le noeud fils droit
        @constraint(m, [i in 1:dataCount, t in 1:sepCount], u_at[i, t*2+1] <= sum(a[j, t] for j in 1:featuresCount)) # contrainte de capacité empechant les données de passer dans le fils droit d'un noeud n'appliquant pas de règle de branchement
        @constraint(m, [i in 1:dataCount, t in 1:sepCount], u_at[i, t*2] <= sum(a[j, t] for j in 1:featuresCount)) # contrainte de capacité empechant les données de passer dans le fils gauche d'un noeud n'appliquant pas de règle de branchement
    end

    ## Déclaration de l'objectif
    @objective(m, Max,  sum(u_at[i, 1] for i in 1:dataCount)) 

    classif = @expression(m, sum(u_at[i, 1] for i in 1:dataCount))

    starting_time = time()
    optimize!(m)
    resolution_time = time() - starting_time
    
    gap = -1.0

    # Arbre obtenu (vide si le solveur n'a trouvé aucune solution)
    T = nothing
    objective = -1
    
    # Si une solution a été trouvée
    if primal_status(m) == MOI.FEASIBLE_POINT

        # class[t] : classe prédite par le sommet t
        class = Vector{Int64}(undef, sepCount+leavesCount)
        for t in 1:(sepCount+leavesCount)
            k = argmax(value.(c[:, t]))
            if value.(c[k, t])  >=  1.0 - 10^-4
                class[t] = k
            else
                class[t] = -1
            end
        end
        
        objective = JuMP.objective_value(m) 
        # Si une solution optimale a été trouvée
        if termination_status(m) == MOI.OPTIMAL
            gap = 0
        else
            # Calcul du gap relatif entre l'objectif de la meilleure solution entière et la borne continue en fin de résolution
            bound = JuMP.objective_bound(m)
            gap = 100.0 * abs(objective - bound) / (objective + 10^-4) # +10^-4 permet d'éviter de diviser par 0
        end   
        
        # Construction d'une variable de type Tree dans laquelle chaque séparation est recentrée
        if multivariate
            T = Tree(D, class, round.(Int, value.(u_at)), round.(Int, value.(s)), x)
        else
            T = Tree(D, value.(a), class, round.(Int, value.(u_at)), x)
        end
    end   

    return T, objective, resolution_time, gap
end

"""
FONCTION SIMILAIRE A LA PRECEDENTE UTILISEE UNIQUEMENT SI VOUS FAITES DES REGROUPEMENTS 

Construit un arbre de décision par résolution de :
-  la formulation F_U (si les paramètres useFhS et useFeS = false)
-  la formulation F^h_S (si le paramètre useFhS = true)
-  la formulation F^e_S (si le paramètre useFeS = true)

Entrées :
- clusters : partition des données d'entraînement (chaque cluster contient des données de même classe)
- D : Nombre maximal de séparations d'une branche (profondeur de l'arbre - 1)
- multivariate (optionnel): vrai si les séparations sont multivariées; faux si elles sont univariées (faux par défaut)
- mu (optionnel, utilisé en multivarié): distance minimale à gauche d'une séparation où aucune donnée ne peut se trouver (i.e., pour la séparation ax <= b, il n'y aura aucune donnée dans ]b - ax - mu, b - ax[) (10^-4 par défaut)
- time_limits (optionnel) : temps maximal de résolution (-1 si le temps n'est pas limité) (-1 par défaut)
- useFhS (optionnel): vrai si la formulation FhS est utilisée
- useFeS (optionnel): vrai si la formulation FeS est utilisée
"""
function build_tree(clusters::Vector{Cluster}, D::Int64, classes;multivariate::Bool=false, time_limit::Int64 = -1, mu::Float64=10^(-4), useFhS::Bool=false, useFeS::Bool=false)
    
    dataCount = sum(length(c.dataIds) for c in clusters) # Nombre de données d'entraînement
    clusterCount = length(clusters) # Nombre de données d'entraînement
    featuresCount = size(clusters[1].x, 2) # Nombre de caractéristiques
    classCount = length(classes) # Nombre de classes différentes
    sepCount = 2^D - 1 # Nombre de séparations de l'arbre
    leavesCount = 2^D # Nombre de feuilles de l'arbre
    
    m = Model(CPLEX.Optimizer) 

    set_silent(m) # Masque les sorties du solveur

    if time_limit!=-1
        set_time_limit_sec(m, time_limit)
    end

    # Plus petite différence entre deux données pour une caractéristique
    mu_min = 1.0 
    # Plus grande différence entre deux données pour une caractéristique
    mu_max = 0.0
    
    if !multivariate # calcul des constantes mu_min, mu_max et du vecteur mu
        mu_vect = ones(Float64, featuresCount)
        for j in 1:featuresCount
            for i1 in 1:clusterCount
                for i2 in (i1+1):clusterCount

                    if useFhS || useFeS
                        if abs(clusters[i1].barycenter[j] - clusters[i2].barycenter[j]) > 1E-4
                            mu_vect[j] = min(mu_vect[j], abs(clusters[i1].barycenter[j] - clusters[i2].barycenter[j]))
                        end
                    else  
                    v1 = clusters[i1].lBounds[j] - clusters[i2].uBounds[j]
                    v2 = clusters[i2].lBounds[j] - clusters[i1].uBounds[j]

                    # Si les clusters n'ont pas des intervalles pour la caractéristique j qui s'intersectent
                        if v1 > 1E-4 || v2 > 1E-4
                        vMin = min(abs(v1), abs(v2))
                        mu_vect[j] = min(mu_vect[j], vMin)
                        end
                    end
                end
            end
            mu_min = min(mu_min, mu_vect[j])
            mu_max = max(mu_max, mu_vect[j])
        end
    end

    ## Déclaraction des variables
    if multivariate
        @variable(m, a[1:featuresCount, 1:sepCount], base_name="a_{j, t}")
        @variable(m, a_h[1:featuresCount, 1:sepCount], base_name="â_{j, t}")
        @variable(m, s[1:featuresCount, 1:sepCount], Bin, base_name="s_{j, t}")
        @variable(m, d[1:sepCount], Bin, base_name="d_t")
    else
        @variable(m, a[1:featuresCount, 1:sepCount], Bin, base_name="a")
    end 
    @variable(m, b[1:sepCount], base_name="b_t")
    @variable(m, c[1:classCount, 1:(sepCount+leavesCount)], Bin, base_name = "c_{k, t}")
    @variable(m, u_at[1:clusterCount, 1:(sepCount+leavesCount)], Bin, base_name = "u^i_{a(t), t}")
    @variable(m, u_tw[1:clusterCount, 1:(sepCount+leavesCount)], Bin, base_name = "u^i_{t, w}")

    if useFeS
        @variable(m, r[1:dataCount], Bin)
        @constraint(m, [clusterId in 1:clusterCount], sum(r[i] for i in clusters[clusterId].dataIds) == 1)
    end 
    ## Déclaration des contraintes
    
    # Contraintes définissant la structure de l'arbre
    if multivariate
        @constraint(m, [t in 1:sepCount], d[t] + sum(c[k, t] for k in 1:classCount) == 1) # on s'assure que le noeud applique une règle de branchement OU attribue une classe
        @constraint(m, [t in 1:sepCount], b[t] <= d[t]) # b doit être nul si il n'y a pas de branchement 
        @constraint(m, [t in 1:sepCount], b[t] >= -d[t]) # b doit être nul si il n'y a pas de branchement 
        @constraint(m, [t in 1:sepCount], sum(a_h[j, t] for j in 1:featuresCount) <= d[t]) # on borne la norme du vecteur a
        @constraint(m, [t in 1:sepCount, j in 1:featuresCount], a[j, t] <= a_h[j, t]) # définition de â borne sup de la valeur absolu de a
        @constraint(m, [t in 1:sepCount, j in 1:featuresCount], a[j, t] >= -a_h[j, t]) # définition de â borne sup de la valeur absolu de a
        @constraint(m, [t in 1:sepCount, j in 1:featuresCount], a[j, t] <= s[j, t]) # définition de s, non nul ssi a non nul
        @constraint(m, [t in 1:sepCount, j in 1:featuresCount], a[j, t] >= -s[j, t]) # définition de s, non nul ssi a non nul
        @constraint(m, [t in 1:sepCount, j in 1:featuresCount], s[j, t] <= d[t]) # définition de d, non nul si il existe un coef non nul
        @constraint(m, [t in 1:sepCount], sum(s[j, t] for j in 1:featuresCount) >= d[t]) # définition de d, non nul si il existe un coef non nul
        @constraint(m, [t in 2:sepCount], d[t] <= d[t ÷ 2]) # on s'assure que si un noeud de branchement n'applique pas de règle de branchement, ses fils non plus
    else
        @constraint(m, [t in 1:sepCount], sum(a[j, t] for j in 1:featuresCount) + sum(c[k, t] for k in 1:classCount) == 1) # on s'assure que le noeud applique une règle de branchement OU attribue une classe
        @constraint(m, [t in 1:sepCount], b[t] <= sum(a[j, t] for j in 1:featuresCount)) # b doit être nul si il n'y a pas de branchement 
        @constraint(m, [t in 1:sepCount], b[t] >= 0) # b doit être positif
    end
    @constraint(m, [t in (sepCount+1):(sepCount+leavesCount)], sum(c[k, t] for k in 1:classCount) == 1) # on s'assure qu'on attribue une classe par feuille

    # contraintes de conservation du flot et contraintes de capacité
    @constraint(m, [i in 1:clusterCount, t in 1:sepCount], u_at[i, t] == u_at[i, t*2] + u_at[i, t*2+1] + u_tw[i, t]) # conservation du flot dans les noeuds de branchement
    @constraint(m, [i in 1:clusterCount, t in (sepCount+1):(sepCount+leavesCount)], u_at[i, t] == u_tw[i, t]) # conservation du flot dans les feuilles
    @constraint(m, [i in 1:clusterCount, t in 1:(sepCount+leavesCount)], u_tw[i, t] <= c[findfirst(classes .== clusters[i].class), t]) # contrainte de capacité qui impose le flot a etre nul si la classe de la feuille n'est pas la bonne
    if multivariate
        if useFhS
            @constraint(m, [i in 1:clusterCount, t in 1:sepCount], sum(a[j, t]*clusters[i].barycenter[j] for j in 1:featuresCount) + mu <= b[t] + (2+mu)*(1-u_at[i, t*2])) # contrainte de capacité controlant le passage dans le noeud fils gauche
            @constraint(m, [i in 1:clusterCount, t in 1:sepCount], sum(a[j, t]*clusters[i].barycenter[j] for j in 1:featuresCount) >= b[t] - 2*(1-u_at[i, t*2 + 1])) # contrainte de capacité controlant le passage dans le noeud fils droit
        elseif useFeS
            @constraint(m, [(clusterId, cluster) in enumerate(clusters), dataId in cluster.dataIds, t in 1:sepCount], sum(a[j, t]*cluster.x[dataId, j] for j in 1:featuresCount) + mu <= b[t] + (2+mu)*(2-u_at[clusterId, t*2]-r[dataId])) # contrainte de capacité controlant le passage dans le noeud fils gauche
            @constraint(m, [(clusterId, cluster) in enumerate(clusters), dataId in cluster.dataIds, t in 1:sepCount], sum(a[j, t]*cluster.x[dataId, j] for j in 1:featuresCount) >= b[t] - 2*(1-u_at[clusterId, t*2 + 1])) # contrainte de capacité controlant le passage dans le noeud fils droit
        
        else 
            @constraint(m, [i in 1:clusterCount, t in 1:sepCount, dataId in clusters[i].dataIds], sum(a[j, t]*clusters[i].x[dataId, j] for j in 1:featuresCount) + mu <= b[t] + (2+mu)*(1-u_at[i, t*2])) # contrainte de capacité controlant le passage dans le noeud fils gauche
            @constraint(m, [i in 1:clusterCount, t in 1:sepCount, dataId in clusters[i].dataIds], sum(a[j, t]*clusters[i].x[dataId, j] for j in 1:featuresCount) >= b[t] - 2*(1-u_at[i, t*2 + 1])) # contrainte de capacité controlant le passage dans le noeud fils droit
        end 
        @constraint(m, [i in 1:clusterCount, t in 1:sepCount], u_at[i, t*2+1] <= d[t]) # contrainte de capacité empechant les données de passer dans le fils droit d'un noeud n'appliquant pas de règle de branchement
    else
        if useFhS
            @constraint(m, [i in 1:clusterCount, t in 1:sepCount], sum(a[j, t]*(clusters[i].barycenter[j]+mu_vect[j]-mu_min) for j in 1:featuresCount) + mu_min <= b[t] + (1+mu_max)*(1-u_at[i, t*2])) # contrainte de capacité controlant le passage dans le noeud fils gauche
            @constraint(m, [i in 1:clusterCount, t in 1:sepCount], sum(a[j, t]*clusters[i].barycenter[j] for j in 1:featuresCount) >= b[t] - (1-u_at[i, t*2 + 1])) # contrainte de capacité controlant le passage dans le noeud fils droit
        elseif useFeS
            @constraint(m, [(clusterId, cluster) in enumerate(clusters), dataId in cluster.dataIds, t in 1:sepCount], sum(a[j, t]*(cluster.x[dataId, j]+mu_vect[j]-mu_min) for j in 1:featuresCount) + mu_min <= b[t] + (1+mu_max)*(2-u_at[clusterId, t*2]-r[dataId])) # contrainte de capacité controlant le passage dans le noeud fils gauche
            @constraint(m, [(clusterId, cluster) in enumerate(clusters), dataId in cluster.dataIds, t in 1:sepCount], sum(a[j, t]*cluster.x[dataId, j] for j in 1:featuresCount) >= b[t] - (2-u_at[clusterId, t*2 + 1]-r[dataId])) # contrainte de capacité controlant le passage dans le noeud fils droit
    else
        @constraint(m, [i in 1:clusterCount, t in 1:sepCount], sum(a[j, t]*(clusters[i].uBounds[j]+mu_vect[j]-mu_min) for j in 1:featuresCount) + mu_min <= b[t] + (1+mu_max)*(1-u_at[i, t*2])) # contrainte de capacité controlant le passage dans le noeud fils gauche
        @constraint(m, [i in 1:clusterCount, t in 1:sepCount], sum(a[j, t]*clusters[i].lBounds[j] for j in 1:featuresCount) >= b[t] - (1-u_at[i, t*2 + 1])) # contrainte de capacité controlant le passage dans le noeud fils droit
        end 
        @constraint(m, [i in 1:clusterCount, t in 1:sepCount], u_at[i, t*2+1] <= sum(a[j, t] for j in 1:featuresCount)) # contrainte de capacité empechant les données de passer dans le fils droit d'un noeud n'appliquant pas de règle de branchement
    end

    ## Déclaration de l'objectif
    @objective(m, Max, sum(length(clusters[i].dataIds) * u_at[i, 1] for i in 1:clusterCount)) 

    starting_time = time()
    optimize!(m)
    resolution_time = time() - starting_time

    gap = -1.0


    # Arbre obtenu (vide si le solveur n'a trouvé aucune solution)
    T = nothing
    objective = -1
    
    # Si une solution a été trouvée
    if primal_status(m) == MOI.FEASIBLE_POINT
    # class[t] : classe prédite par le sommet t
    class = Vector{Int64}(undef, sepCount+leavesCount)
    for t in 1:(sepCount+leavesCount)
        k = argmax(value.(c[:, t]))
        if value.(c[k, t]) >= 1.0 - 10^-4
            class[t] = k
        else
            class[t] = -1
        end
    end
        
        objective = JuMP.objective_value(m)

        # Si une solution optimale a été trouvée
        if termination_status(m) == MOI.OPTIMAL
            gap = 0
        else
            # Calcul du gap relatif entre l'objectif de la meilleure solution entière et la borne continue en fin de résolution
            bound = JuMP.objective_bound(m)
            gap = 100.0 * abs(objective - bound) / (objective + 10^-4) # +10^-4 permet d'éviter de diviser par 0
        end   
        
        # Construction d'une variable de type Tree dans laquelle chaque séparation est recentrée
        if multivariate
            T = Tree(D, class, round.(Int, value.(u_at)), round.(Int, value.(s)), clusters)
        else
            T = Tree(D, value.(a), class, round.(Int, value.(u_at)), clusters)
        end
    end   

    return T, objective, resolution_time, gap
end

"""
Construit un arbre de décision par résolution itérative des formulations F^e_S ou F^h_S

Entrées :
- clusters : partition des données d'entraînement (chaque cluster contient des données de même classe)
- D : Nombre maximal de séparations d'une branche (profondeur de l'arbre - 1)
- multivariate (optionnel): vrai si les séparations sont multivariées; faux si elles sont univariées (faux par défaut)
- mu (optionnel, utilisé en multivarié): distance minimale à gauche d'une séparation où aucune donnée ne peut se trouver (i.e., pour la séparation ax <= b, il n'y aura aucune donnée dans ]b - ax - mu, b - ax[) (10^-4 par défaut)
- time_limits (optionnel) : temps maximal de résolution (-1 si le temps n'est pas limité) (-1 par défaut)
- isExact (optionnel) : vrai si la formulation F^e_S est utilisée; la formulation F^h_S est utilisée sinon (false par défaut)
"""
function iteratively_build_tree(clusters::Vector{Cluster}, D::Int64, x::Matrix{Float64}, y::Vector{}, classes::Vector{};multivariate::Bool=false, time_limit::Int64 = -1, mu::Float64=10^(-4), isExact::Bool=false, shiftSeparations::Bool=false)

    startingTime = time()
    finalTime = startingTime + time_limit

    # Define variables used as return values
    # (otherwise they would not be defined outside of the while loop)
    lastObjective = nothing
    lastFeasibleT = nothing
    gap = nothing

    clusterSplit = true
    iterationCount = 0

    useFhS = !isExact
    useFeS = isExact

    # While cluster are split and the time limit is not reached
    while clusterSplit && (time_limit == -1 || time() < finalTime)

        iterationCount += 1
        remainingTime = round(Int, finalTime-time())
        
        # Solve with the current clusters
        T, objective, resolution_time, gap = build_tree(clusters, D, classes, multivariate=multivariate, time_limit=time_limit==-1 ? -1 : remainingTime, useFhS=useFhS, useFeS=useFeS)

        # If a solution has been obtained
        if objective != -1

            if shiftSeparations
                T = naivelyShiftSeparations(T, x, y, classes, clusters)
            end 
            
            # List of the clusters for the next iteration
            newClusters = Vector{Cluster}()

            # For each cluster
            for cluster in clusters

                # Split its data according to the leaves of tree T they reach
                newCurrentClusters = getSplitClusters(cluster, T)
                append!(newClusters, newCurrentClusters)
            end

            # If no cluster is split
            if length(clusters) == length(newClusters)
                clusterSplit = false
            else 
                clusters = newClusters
            end
            lastFeasibleT = T
            lastObjective = objective
        end 
    end

    resolution_time = time() - startingTime
    return lastFeasibleT, lastObjective, resolution_time, gap, iterationCount
end

