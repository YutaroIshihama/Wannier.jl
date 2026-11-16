export TBOperator, simplify

"""
A struct representing a tight-binding operator in R-space.

# Fields
$(FIELDS)

!!! note

    We limit the type of `Rspace` to only [`BareRspace`](@ref), because
    this simplifies a lot of codes and compute faster.
    To covert a [`WignerSeitzRspace`](@ref) or [`MDRSRspace`](@ref)
    to [`BareRspace`](@ref), see [`simplify`](@ref).
    Note also that [`invfourier`](@ref) only accepts [`BareRspace`](@ref).
"""
struct TBOperator{M <: AbstractMatrix}
    """a concise name for the operator"""
    name::String

    """the R-space domain (or called R-vectors) on which the operator is defined"""
    Rspace::BareRspace

    """the actual operator defined on the `Rspace`, should be a
    length-`n_Rvectors` vector, where each element type `T` can be
    - a scaler
    - a matrix of scaler, e.g., `Matrix{ComplexF64}` for Hamiltonian
    - a matrix of vector, e.g., `Matrix{MVec3{ComplexF64}}` for position
        operator, spin operator, etc."""
    operator::Vector{M}
end

function n_wannier(tb::TBOperator)
    return isempty(tb.operator) ? 0 : size(tb.operator[1], 1)
end
n_Rvectors(tb::TBOperator) = n_Rvectors(tb.Rspace)
CrystalBase.real_lattice(tb::TBOperator) = real_lattice(tb.Rspace)

# the Rspace is so simple so we expose it to the user
function Base.propertynames(tb::TBOperator)
    return Tuple(
        [
            collect(fieldnames(typeof(tb.Rspace)))
            collect(fieldnames(typeof(tb)))
        ]
    )
end

function Base.getproperty(tb::TBOperator, sym::Symbol)
    type_R = typeof(getfield(tb, :Rspace))
    if sym ∈ fieldnames(type_R)
        return getfield(getfield(tb, :Rspace), sym)
    else
        # fallback
        return getfield(tb, sym)
    end
end

function Base.show(io::IO, ::MIME"text/plain", tb::TBOperator)
    @printf(io, "Tight-binding operator name  :  %s\n", tb.name)
    show(io, MIME"text/plain"(), tb.Rspace)
    println(io)
    return @printf(io, "n_wannier   =  %d", n_wannier(tb))
end

"""Index using `i` of Rspace"""
function Base.getindex(tb::TBOperator, i::Integer)
    return tb.operator[i]
end
function Base.getindex(tb::TBOperator, r::UnitRange{<:Integer})
    return tb.operator[r]
end

Base.lastindex(tb::TBOperator) = lastindex(tb.operator)
Base.length(tb::TBOperator) = length(tb.operator)

function Base.iterate(tb::TBOperator, state = 1)
    if state > length(tb.operator)
        return nothing
    else
        return (tb.operator[state], state + 1)
    end
end

"""Index using 3-vector of R-vectors coordinates"""
function Base.getindex(tb::TBOperator, R::Vec3{Int})
    return getindex(tb, R[1], R[2], R[3])
end

"""Index using 3-vector of R-vectors coordinates"""
function Base.getindex(tb::TBOperator, x::Integer, y::Integer, z::Integer)
    iR = tb.Rspace.xyz_iR[x, y, z]
    if iszero(iR)
        throw(BoundsError(tb, [x, y, z]))
    end
    return tb.operator[iR]
end

@inline function zeros_operator(T::DataType, nwann::Integer, nRvecs::Integer)
    return [zeros(T, (nwann, nwann)) for _ in 1:nRvecs]
end

@inline function zeros_operator(operator::AbstractVector{<:AbstractMatrix})
    @assert length(operator) > 0 "empty operator"
    T = eltype(operator[1])
    n = size(operator[1], 1)
    return zeros_operator(T, n, length(operator))
end

@inline function Base.zeros(tb::TBOperator)
    return TBOperator("zeros", tb.Rspace, zeros_operator(tb.operator))
end

function Base.fill!(tb::TBOperator, x)
    for O in tb.operator
        if eltype(O) <: AbstractArray
            for Oi in O
                fill!(Oi, x)
            end
        else
            O .= x
        end
    end
    return
end

function Base.isapprox(a::TBOperator, b::TBOperator; kwargs...)
    return isapprox_struct(a, b; kwargs...)
end

"""
    $(SIGNATURES)

Simplify the R-vectors of a tight-binding operator, such that the inverse
Fourier transform is a simple sum (and is faster).

- If the R-space domain is a `MDRSRspace`, expand the R-vectors to a new
    set of R-vectors `R̃ = R + T`, where `T` is the translation vectors;
    also divide the operator by the degeneracies of `R` and `T` vectors.
- If the R-space domain is a `WignerSeitzRspace`, divide the operator by the
    R-vector degeneracy.
- If the R-space domain is a `BareRspace`, do nothing.

# Return
- a `BareRspace` with simplified Rvectors
- a vector for the simplified `operator` defined on the `BareRspace`
"""
function simplify end

"""
    RvectorReducer

Functor that simplifies R-vectors, replacing the removed `WannierIO.RvectorReducer`.

# Fields
- `Rvectors`: simplified R-vectors (`R` for Wigner-Seitz, `R + T` for MDRS)
- `contribs`: for each simplified R-vector index, list of
  `(old_index, m, n, weight)` contributions. `m == n == 0` means
  "whole operator at `old_index`, scaled by `weight`"
  (Wigner-Seitz case); otherwise only element `(m, n)` of the
  operator at `old_index` contributes.
"""
struct RvectorReducer
    Rvectors::Vector{Vec3{Int}}
    contribs::Vector{Vector{Tuple{Int, Int, Int, Float64}}}
end

function RvectorReducer(Rvectors::AbstractVector, n_Rdegens::AbstractVector)
    Rvecs = Vector{Vec3{Int}}(Rvectors)
    contribs = map(enumerate(n_Rdegens)) do (i, d)
        [(i, 0, 0, 1.0 / d)]
    end
    return RvectorReducer(Rvecs, contribs)
end

function RvectorReducer(
        Rvectors::AbstractVector,
        n_Rdegens::AbstractVector,
        Tvectors::AbstractVector,
        n_Tdegens::AbstractVector,
    )
    # Expand R̃ = R + T, deduplicate in first-appearance order.
    new_Rvectors = Vec3{Int}[]
    index_of = Dict{Vec3{Int}, Int}()
    contribs = Vector{Tuple{Int, Int, Int, Float64}}[]
    for (iR, R) in enumerate(Rvectors)
        Tv = Tvectors[iR]
        Td = n_Tdegens[iR]
        nwann_m, nwann_n = size(Tv)
        for m in 1:nwann_m, n in 1:nwann_n
            w = 1.0 / (n_Rdegens[iR] * Td[m, n])
            for T in Tv[m, n]
                Rt = R + T
                j = get(index_of, Rt, 0)
                if iszero(j)
                    push!(new_Rvectors, Rt)
                    push!(contribs, Tuple{Int, Int, Int, Float64}[])
                    j = length(new_Rvectors)
                    index_of[Rt] = j
                end
                push!(contribs[j], (iR, m, n, w))
            end
        end
    end
    return RvectorReducer(new_Rvectors, contribs)
end

RvectorReducer(Rspace::MDRSRspace) = RvectorReducer(
    Rspace.Rvectors, Rspace.n_Rdegens, Rspace.Tvectors, Rspace.n_Tdegens
)
RvectorReducer(Rspace::WignerSeitzRspace) =
    RvectorReducer(Rspace.Rvectors, Rspace.n_Rdegens)

function (reducer::RvectorReducer)(operators::AbstractVector)
    # Zero-filled output, same element type as input.
    out = [zero.(O) for O in [operators[1] for _ in 1:length(reducer.Rvectors)]]
    # `zero.(O)` above broadcasts; for plain matrices it is `zero(O)`.
    # Rebuild correctly for the common matrix case:
    out = [zero(operators[1]) for _ in 1:length(reducer.Rvectors)]
    for (j, cs) in enumerate(reducer.contribs)
        for (iR, m, n, w) in cs
            O = operators[iR]
            if m == 0 && n == 0
                out[j] = out[j] + O .* w
            else
                out[j][m, n] = out[j][m, n] + O[m, n] * w
            end
        end
    end
    return out
end

function simplify(Rspace::Union{MDRSRspace,WignerSeitzRspace}, operators::AbstractVector...)
    reducer = RvectorReducer(Rspace)
    bare_Rspace = BareRspace(Rspace.lattice, reducer.Rvectors)
    bare_operators = map(reducer, operators)
    return bare_Rspace, bare_operators...
end

function simplify(Rspace::BareRspace, operators::AbstractVector...)
    return Rspace, operators...
end

"""
    $(SIGNATURES)

Sort a Rspace operator (e.g. Hamiltonian ``H(\\mathbf{R})``) by the norm of
``\\mathbf{R}``-vectors.

Since the operator decays ~expoentially with ``\\mathbf{R}``, the resulting
operator should be in descending order.

# Return
- `idx`: the index to sort the operator, i.e. `Oᴿ[idx]` is the sorted operator
- `normR`: the norm of R vectors, `normR[idx]` is the sorted `R` vectors in ascending order
- `normO`: the norm of the operator, `normO[idx]` is the sorted `Oᴿ` in descending order

# Example
```julia
hamiltonian, position = read_w90_tb("mos2")
idx, normR, normH = Wannier.sort_by_Rnorm(hamiltonian)
```
"""
function sort_by_Rnorm(tb::TBOperator)
    normO = map(norm, tb.operator)
    normR = map(tb.Rvectors) do R
        norm(tb.lattice * R)
    end
    idx = sortperm(normR)

    @printf("# No.    ||R|| (Å)    ||H(R)||\n")
    for i in 1:20
        @printf("%2d       %.5f       %.5f\n", i - 1, normR[idx[i]], normO[idx[i]])
    end
    return idx, normR, normO
end

"""
    $(SIGNATURES)

Cut R-space operator (e.g. Hamiltonian) by the given cutoff radius.

# Arguments
- `Rcut`: cutoff radius in angstrom, the operator `O(R)` with R-vector norm
    larger than `Rcut` will removed.

# Example
```julia
hamiltonian, position = read_w90_tb("mos2")
H_cut = Wannier.cut(hamiltonian)
```
"""
function cut(tb::TBOperator, Rcut::Real)
    println("Sorting by R norm...")
    _, normR, _ = sort_by_Rnorm(tb)
    println("")
    idx_keep = normR .<= Rcut
    Rspace = BareRspace(tb.Rspace.lattice, deepcopy(tb.Rvectors[idx_keep]))
    return TBOperator(tb.name, Rspace, deepcopy(tb.operator[idx_keep]))
end

"""An abstract type that interpolate physical quantities from some
[`TBOperator`](@ref)s

Each subtype should define a function
`function (interp::name_of_Interpolator)(kpoints::AbstractVector{<:AbstractVector})`
that returns the interpolated physical quantities for the given list of `kpoints`.

Since it interpolates back to Bloch gauge, almost always the 1st field is
`hamiltonian::TBHamiltonian`.
"""
abstract type AbstractTBInterpolator <: Function end

@inline function (interp::AbstractTBInterpolator)(
        kpoint::AbstractVector{<:Real}, args...; kwargs...
    )
    # unwrap results
    result = interp([kpoint], args...; kwargs...)
    if length(result) == 1
        return result[1]
    else
        return Tuple(map(x -> x[1], result))
    end
end

n_wannier(interp::AbstractTBInterpolator) = n_wannier(interp.hamiltonian)
n_Rvectors(interp::AbstractTBInterpolator) = n_Rvectors(interp.hamiltonian)
CrystalBase.real_lattice(interp::AbstractTBInterpolator) = real_lattice(interp.hamiltonian)

function Base.show(io::IO, ::MIME"text/plain", interp::AbstractTBInterpolator)
    @printf(io, "Tight-binding interpolator name  :  %s\n", nameof(interp))
    ops = []
    itps = []
    for p in propertynames(interp)
        f = getproperty(interp, p)
        if f isa TBOperator
            push!(ops, f.name)
        elseif f isa AbstractTBInterpolator
            push!(itps, nameof(typeof(f)))
        else
            continue
        end
    end
    if length(itps) > 0
        println(io, "\nList of contained interpolators:")
        for it in itps
            println(io, "  ", it)
        end
    end
    if length(ops) > 0
        println(io, "\nList of contained operators:")
        for op in ops
            println(io, "  ", op)
        end
    end
    println(io, "\nSummary:")
    length(itps) > 0 && @printf(io, "  n_interpolators =  %d\n", length(itps))
    length(ops) > 0 && @printf(io, "  n_operators     =  %d\n", length(ops))
    @printf(io, "  n_Rvectors      =  %d\n", n_Rvectors(interp))
    return @printf(io, "  n_wannier       =  %d", n_wannier(interp))
end
