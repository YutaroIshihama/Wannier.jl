export read_w90, read_w90_with_chk, write_w90

"""
    $(SIGNATURES)

Read `win` and `mmn` files, and read `amn`/`eig` files if they exist.

# Keyword arguments
- ortho_amn: Lowdin orthonormalization after reading `amn` matrices.
    Should be `true` for most cases, since usually the input `amn` matrices are
    not guaranteed to be unitary or semi-unitary.
- use_mmn_bvecs: use the b-vectors in `mmn` file instead of regenerating them.
- kstencil_algo: algorithm to generate `KspaceStencil` if `use_mmn_bvecs` is `false`.
    Default is `generate_kspace_stencil`.
"""
function read_w90(prefix::AbstractString; ortho_amn::Bool = true, use_mmn_bvecs::Bool = true, kstencil_algo::KspaceStencilAlgorithm = default_kstencil_algo())
    win = read_win(prefix * ".win")
    nbands = win["num_bands"]
    nwann = win["num_wann"]
    lattice = win["unit_cell_cart"]
    recip_lattice = reciprocal_lattice(lattice)
    nkpts = length(win["kpoints"])

    isfile(prefix * ".mmn") || error("$(prefix).mmn file does not exist")
    mmn = read_mmn(prefix * ".mmn")
    # WannierIO 0.3: M is Array{Complex,4} (nbands, nbands, nbvecs, nkpts),
    # kpb_k / kpb_G are Matrix (nbvecs, nkpts).
    # Wannier.jl Model / KspaceStencil expect Vector-of-Vector layout.
    M4 = mmn.M
    nbvecs = size(M4, 3)
    overlaps = [[Matrix(M4[:, :, ib, ik]) for ib in 1:nbvecs] for ik in 1:nkpts]
    kpb_k = [[mmn.kpb_k[ib, ik] for ib in 1:nbvecs] for ik in 1:nkpts]
    kpb_G = [[mmn.kpb_G[ib, ik] for ib in 1:nbvecs] for ik in 1:nkpts]
    # check consistency for mmn
    @assert nkpts == length(overlaps) "different n_kpoints in mmn and win files"
    @assert (nbands, nbands) == size(overlaps[1][1]) "different n_bands in mmn and win files"

    if use_mmn_bvecs
        kstencil = KspaceStencil(recip_lattice, win["kpoints"], kpb_k, kpb_G)
    else
        atol = get(win, "kmesh_tol", default_w90_kmesh_tol())
        kstencil = generate_kspace_stencil(
            recip_lattice, win["mp_grid"], win["kpoints"], kstencil_algo; atol
        )
        @assert n_bvectors(kstencil) == nbvecs "different n_bvectors in mmn and win files"
        @assert kstencil.kpb_k == kpb_k "auto generated kpb_k are different from mmn file"
        @assert kstencil.kpb_G == kpb_G "auto generated kpb_G are different from mmn file"
        # overlaps = zeros_overlap(ComplexF64, nkpts, nbvecs, nbands)
        # @warn "$prefix.mmn file does not exist, set M to zeros"
    end

    if isfile(prefix * ".amn")
        if ortho_amn
            gauges = read_amn_ortho("$prefix.amn")
        else
            A3 = read_amn("$prefix.amn").A
            gauges = [Matrix(A3[:, :, ik]) for ik in axes(A3, 3)]
        end
        @assert nkpts == length(gauges) "different n_kpoints in amn and win files"
        @assert (nbands, nwann) == size(gauges[1]) "different n_bands or n_wannier in amn and win files"
    else
        gauges = zeros_gauge(ComplexF64, nkpts, nbands, nwann)
        @warn "$prefix.amn file does not exist, set U to zeros"
    end

    disentangle = nbands != nwann
    if isfile(prefix * ".eig")
        Emat = read_eig(prefix * ".eig")
        # WannierIO 0.3: Matrix (nbands, nkpts) -> Vector{Vector}
        eigenvalues = [Vector(Emat[:, ik]) for ik in axes(Emat, 2)]
        @assert nkpts == length(eigenvalues) "different n_kpoints in eig and win files"
        @assert nbands == length(eigenvalues[1]) "different n_bands in eig and win files"
    else
        @assert !disentangle "eig file is required for disentanglement"
        eigenvalues = [zeros(Float64, nbands) for _ in 1:nkpts]
        @warn "$prefix.eig file does not exist, set eigenvalues to zeros"
    end

    if disentangle
        dis_froz_max = get(win, "dis_froz_max", nothing)
        dis_froz_min = get(win, "dis_froz_min", -Inf)
        if isnothing(dis_froz_max)
            frozen_bands = [falses(nbands) for _ in 1:nkpts]
        else
            frozen_bands = get_frozen_bands(eigenvalues, dis_froz_max, dis_froz_min)
        end
    else
        frozen_bands = [falses(nbands) for _ in 1:nkpts]
    end

    atom_labels = map(x -> string(x.first), win["atoms_frac"])
    atom_positions = map(x -> x.second, win["atoms_frac"])

    return Model(
        lattice,
        atom_positions,
        atom_labels,
        kstencil,
        overlaps,
        gauges,
        eigenvalues,
        frozen_bands,
    )
end

"""
    $(SIGNATURES)

Return a `Model` with U matrix filled by that from a chk file.

# Arguments
- chk: path of chk file to get the unitary matrices.
"""
function read_w90_with_chk(prefix::AbstractString, chk::AbstractString = "$prefix.chk")
    model = read_w90(prefix)
    fchk = read_chk(chk)
    U3 = WannierIO.gauge_matrices(fchk)
    model.gauges .= [Matrix(U3[:, :, ik]) for ik in axes(U3, 3)]
    return model
end

"""
    $(SIGNATURES)

Write `Model` into `eig`, `mmn`, `amn` files.

# Keyword arguments
- `binary`: write `eig`, `mmn`, and `amn` in Fortran binary format
"""
function write_w90(prefix::AbstractString, model::Model; binary::Bool = false)
    outname(suffix) = "$prefix.$suffix"

    # Model (Vector layout) -> WannierIO 0.3 (dense array / Mmn struct)
    Emat = hcat(model.eigenvalues...)
    A3 = cat(model.gauges...; dims = 3)
    nkpts_w = length(model.overlaps)
    nbvecs_w = length(model.overlaps[1])
    nbands_w = size(model.overlaps[1][1], 1)
    M4 = zeros(ComplexF64, nbands_w, nbands_w, nbvecs_w, nkpts_w)
    for ik in 1:nkpts_w, ib in 1:nbvecs_w
        M4[:, :, ib, ik] .= model.overlaps[ik][ib]
    end
    kpb_k_mat = zeros(Int, nbvecs_w, nkpts_w)
    kpb_G_mat = zeros(WannierIO.Vec3{Int}, nbvecs_w, nkpts_w)
    for ik in 1:nkpts_w, ib in 1:nbvecs_w
        kpb_k_mat[ib, ik] = model.kstencil.kpb_k[ik][ib]
        kpb_G_mat[ib, ik] = model.kstencil.kpb_G[ik][ib]
    end
    mmn_out = WannierIO.Mmn("Written by Wannier.jl", M4, kpb_k_mat, kpb_G_mat)
    write_eig(outname("eig"), Emat; binary)
    write_mmn(outname("mmn"), mmn_out; binary)
    write_amn(outname("amn"), A3; binary)
    return nothing
end

