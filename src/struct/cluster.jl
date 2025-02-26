using Statistics

"""
Représente un regroupement de données
"""
mutable struct Cluster

    dataIds::Vector{Int}
    lBounds::Vector{Float64}
    uBounds::Vector{Float64}
    x::Matrix{Float64}
    class::Any
    barycenter::Vector{Float64}

    function Cluster()
        return new()
    end
end 

"""
Constructeur d'un cluster

Entrées :
- id : identifiant du premier élément du cluster
- x  : caractéristique des données d'entraînement
- y  : classe des données d'entraînement
"""
function Cluster(id::Int, x::Matrix{Float64}, y)

    c = Cluster()
    c.x = x
    c.class = y[id]
    c.dataIds = Vector{Int}([id])
    c.lBounds = Vector{Float64}(x[id, :])
    c.uBounds = Vector{Float64}(x[id, :])
    c.barycenter = getBarycenter(c)

    return c
    
end 
"""
Constructeur d'un cluster

Entrées :
- id : identifiant du premier élément du cluster
- x  : caractéristique des données d'entraînement
- y  : classe des données d'entraînement
"""
function Cluster(id::Vector{Int}, x::Matrix{Float64}, y)

    c = Cluster()
    c.x = x
    c.class = y
    c.dataIds = copy(id)
    c.barycenter = getBarycenter(c)

    # The bounds are not defined since this constructor is not used with formulation F_U
    
    return c
    
end 


"""
Get the barycenter of cluster c
"""
function getBarycenter(c::Cluster)
    return vec(mean(c.x[c.dataIds, :], dims=1))
end 

