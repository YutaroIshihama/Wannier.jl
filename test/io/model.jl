@testitem "read_w90" begin
    using Wannier.Datasets
    model = load_dataset("Si2")

    @test n_bands(model) == 16
    @test n_wannier(model) == 8
    @test n_kpoints(model) == 9^3
end

@testitem "write_w90" begin
    using Wannier.Datasets
    model = load_dataset("Si2")

    outdir = mktempdir(; cleanup = true)
    outprefix = joinpath(outdir, "silicon")
    write_w90(outprefix, model)

    gauges = read_amn_ortho("$outprefix.amn")
    @test gauges ≈ model.gauges

    eigenvalues = read_eig("$outprefix.eig")
    # read_eig is dense (nbands x nk), model.eigenvalues is nested per k
    @test eigenvalues ≈ hcat(model.eigenvalues...)

    mmn = read_mmn("$outprefix.mmn")
    # mmn.M is dense (nb x nb x nbvec x nk), model.overlaps nested per k, per b
    M_dense = zeros(ComplexF64, size(mmn.M))
    for (ik, per_k) in enumerate(model.overlaps), (ib, m) in enumerate(per_k)
        M_dense[:, :, ib, ik] .= m
    end
    @test mmn.M ≈ M_dense
    # mmn.kpb_* are dense (nbvec x nk), model.kpb_* nested per k
    @test mmn.kpb_k == hcat(model.kpb_k...)
    @test mmn.kpb_G == hcat(model.kpb_G...)
end

@testitem "read_w90_with_chk" begin
    using Wannier.Datasets
    model = read_w90_with_chk(dataset"Si2/Si2", dataset"Si2/outputs/Si2.chk")

    @test model isa Wannier.Model
    @test n_wannier(model) == 8
end

@testitem "write chk from Model" begin
    using LinearAlgebra
    using Wannier.Datasets
    model = read_w90_with_chk(dataset"Si2/Si2", dataset"Si2/outputs/Si2.chk.fmt")
    tmpfile = tempname(; cleanup = true)
    write_chk(tmpfile, model)

    chk = read_chk(tmpfile)

    # header contains date, always different
    # @test chk.header == chk2.header
    @test isempty(chk.exclude_bands)
    @test model.lattice ≈ chk.lattice
    @test model.recip_lattice ≈ chk.recip_lattice
    @test model.kgrid_size == chk.kgrid
    @test model.kpoints ≈ chk.kpoints
    @test true == chk.have_disentangled
    @test BitMatrix(hcat(model.entangled_bands...)) == chk.dis_bands
    # the Hamiltonian rotated by Udis must be diagonal, according to W90 convention
    _Udis = WannierIO.gauge_matrices_dis(chk)
    H = transform_gauge(model.eigenvalues, [Matrix(_Udis[:, :, ik]) for ik in axes(_Udis, 3)])
    # this is too strict, even
    #   norm(H[:, :, ik] - Hdiag[:, :, ik]) ≈ 1e-14
    # is still false
    # @test all(isdiag(H[:, :, ik]) for ik in axes(H, 3))
    Hdiag = map(H) do h
        Hermitian(Diagonal(h))
    end
    @test H ≈ Hdiag
    # the unitary matrix should be the same
    @test cat(model.gauges...; dims = 3) ≈ WannierIO.gauge_matrices(chk)

    M = transform_gauge(model.overlaps, model.kpb_k, model.gauges)
    # M nested per k, per b; chk.M dense (nb x nb x nbvec x nk)
    M_dense = zeros(ComplexF64, size(chk.M))
    for (ik, per_k) in enumerate(M), (ib, m) in enumerate(per_k)
        M_dense[:, :, ib, ik] .= m
    end
    @test M_dense ≈ chk.M
    Ω = omega(model, model.gauges)
    @test Ω.ΩI ≈ chk.ΩI
    @test Ω.r ≈ chk.r
    @test Ω.ω ≈ chk.ω
end

@testitem "Model from chk" begin
    using Wannier.Datasets
    chk = read_chk(dataset"Si2/outputs/Si2.chk")
    model = Wannier.Model(chk)

    @test chk.lattice ≈ model.lattice
    @test chk.recip_lattice ≈ reciprocal_lattice(model)
    @test chk.kgrid == model.kgrid_size
    @test chk.kpoints ≈ model.kpoints
    # chk.M dense, model.overlaps nested
    _M_dense = zeros(ComplexF64, size(chk.M))
    for (ik, per_k) in enumerate(model.overlaps), (ib, m) in enumerate(per_k)
        _M_dense[:, :, ib, ik] .= m
    end
    @test chk.M ≈ _M_dense
end
