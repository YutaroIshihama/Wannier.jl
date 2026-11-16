"""
    $(SIGNATURES)

Compute the so-called `uIu` matrix, LVTS12 Eq. xx:
``\\langle u_{m, k + b_1} | u_{n, k + b_2} \\rangle``,
approximately from the overlap matrices
``\\langle u_{m, k} | u_{n, k + b} \\rangle``.

Use the following formula:

```
\\langle u_{m, k + b_1} | u_{n, k + b_2} \\rangle
= \\sum_{i} \\langle u_{m, k + b_1} | u_{i, k} \\rangle
    \\langle u_{i, k} | u_{n, k + b_2} \\rangle
= M_{k, b}^{\\dagger} M_{k, b}
```

This is an approximation to the true `uIu` matrix that should be computed
e.g. in the DFT plane-wave basis.

Note the `overlaps` should be the outputs from a DFT to Wannier interface code,
i.e., they should contain `n_bands`-sized matrices instead of Wannier-gauge
`n_wannier`-sized matrices, to minimize the effect of band truncation.
"""
function compute_uIu(
        overlaps::AbstractVector,
    )
    uIu = map(overlaps) do Mₖ
        # note I need to use reshape to convert it into a row vector,
        # cannot use transpose since that one is recursive
        adjoint.(Mₖ) * reshape(Mₖ, (1, :))
    end
    return uIu
end

"""
    $(SIGNATURES)

Compute the so-called `uHu` matrix, LVTS12 Eq. 96:
``\\langle u_{m, k + b_1}| H | u_{n, k + b_2} \\rangle``,
approximately from the overlap matrices
``\\langle u_{m, k} | u_{n, k + b} \\rangle``
and Hamiltonian.

Use the following formula:

```
\\langle u_{m, k + b_1}| H | u_{n, k + b_2} \\rangle
= \\sum_{i,j} \\langle u_{m, k + b_1} | u_{i, k} \\rangle
    \\langle u_{i, k} | H | u_{j, k} \\rangle
    \\langle u_{j, k} | u_{n, k + b_2} \\rangle
= M_{k, b}^{\\dagger} H_{k} M_{k, b}
```

This is an approximation to the true `uHu` matrix that should be computed
e.g. in the DFT plane-wave basis.

Note the `overlaps` and `eigenvalues` should be the outputs from a DFT to
Wannier interface code, i.e., they should contain `n_bands`-sized matrices
instead of Wannier-gauge `n_wannier`-sized matrices, to minimize the effect
of band truncation.
"""

# Dimension check for the nested layout used by Wannier.jl:
# overlaps[ik][ib] is n_bands x n_bands, kpb_k[ik][ib] / kpb_G[ik][ib] one entry per b-vector.
function _check_dimensions_M_kpb_nested(overlaps, kpb_k, kpb_G)
    nkpts = length(kpb_k)
    length(overlaps) == nkpts || throw(DimensionMismatch("overlaps and kpb_k have different n_kpts"))
    length(kpb_G) == nkpts || throw(DimensionMismatch("kpb_G and kpb_k have different n_kpts"))
    for ik in 1:nkpts
        nbvecs = length(kpb_k[ik])
        length(overlaps[ik]) == nbvecs || throw(DimensionMismatch("overlaps[ik] and kpb_k[ik] have different n_bvecs"))
        length(kpb_G[ik]) == nbvecs || throw(DimensionMismatch("kpb_G[ik] and kpb_k[ik] have different n_bvecs"))
        for ib in 1:nbvecs
            n_bands = size(overlaps[ik][ib], 1)
            n_bands > 0 || throw(ArgumentError("Empty overlaps, n_bands = 0"))
            size(overlaps[ik][ib], 2) == n_bands || throw(DimensionMismatch("overlaps[ik][ib] is not a square matrix"))
            length(kpb_G[ik][ib]) == 3 || throw(DimensionMismatch("kpb_G entries are not 3-vectors"))
        end
    end
    return nothing
end

function compute_uHu(
        overlaps::AbstractVector,
        kpb_k::AbstractVector,
        kpb_G::AbstractVector,
        eigenvalues::AbstractVector,
    )
    _check_dimensions_M_kpb_nested(overlaps, kpb_k, kpb_G)
    nkpts = length(kpb_k)
    @assert nkpts == length(eigenvalues) "wrong n_kpoints in eigenvalues"

    uHu = map(zip(overlaps, eigenvalues)) do (Mₖ, εₖ)
        # note I need to use reshape to convert it into a row vector,
        # cannot use transpose since that one is recursive
        (adjoint.(Mₖ) .* Ref(Diagonal(εₖ))) * reshape(Mₖ, (1, :))
    end
    return uHu
end

"""
Compute H_kb = U'_k H_k M_kb U_kb, LVTS12 Eq. 86 and 91.

```
\\langle u_{m, k}| H | u_{n, k + b} \\rangle
= \\sum_{i} \\langle u_{m, k} | H | u_{i, k} \\rangle
    \\langle u_{i, k} | u_{n, k + b} \\rangle
= H_{k} M_{k, b}
```

In principle, this should be computed in the DFT plane-wave basis, similar to
[`compute_uHu`](@ref); however, no DFT codes implement this, so we approximate
it using the Hamiltonian and overlap matrices.

Note the `overlaps` and `eigenvalues` should be the outputs from a DFT to
Wannier interface code, i.e., they should contain `n_bands`-sized matrices
instead of Wannier-gauge `n_wannier`-sized matrices, to minimize the effect
of band truncation.
"""
function compute_hamiltonian_times_position_kspace(
        overlaps::AbstractVector,
        kpb_k::AbstractVector,
        kpb_G::AbstractVector,
        eigenvalues::AbstractVector,
    )
    _check_dimensions_M_kpb_nested(overlaps, kpb_k, kpb_G)
    nkpts = length(kpb_k)
    @assert nkpts == length(eigenvalues) "wrong n_kpoints in eigenvalues"

    Hkb = map(zip(eigenvalues, overlaps)) do (εₖ, Mₖ)
        Ref(Diagonal(εₖ)) .* Mₖ
    end
    # Hkb can be indexed by Hkb[ik][ib][m, n]
    return Hkb
end
