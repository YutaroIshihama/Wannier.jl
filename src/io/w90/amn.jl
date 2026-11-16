export read_amn_ortho

"""
    $(SIGNATURES)

Read `amn` file and Lowdin orthonormalize the unitary matrices.

The `U` matrix for Wannier functions must be unitary or semi-unitary.
Thus, in most cases, this function should be used instead of `WannierIO.read_amn`,
where the latter one just parse the `amn` file and return whatever is in it.
"""
function read_amn_ortho(filename::AbstractString)
    # WannierIO 0.3: A is Array{Complex,3} of size (nbands, nwann, nkpts)
    # Wannier.jl Model expects Vector{Matrix}, one per k-point.
    A = WannierIO.read_amn(filename).A
    U = [Matrix(A[:, :, ik]) for ik in axes(A, 3)]
    for ik in eachindex(U)
        U[ik] = orthonorm_lowdin(U[ik])
    end
    return U
end
