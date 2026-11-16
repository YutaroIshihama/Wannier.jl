@testitem "fourier" begin
    using Wannier.Datasets
    tbdat = read_w90_tb_dat(dataset"Si2_valence/outputs/WS/Si2_valence_tb.dat")
    ref_Hᴿ = tbdat.H
    Rspace = Wannier.WignerSeitzRspace(tbdat.lattice, tbdat.Rvectors, tbdat.Rdegens)

    model = read_w90_with_chk(
        dataset"Si2_valence/Si2_valence", dataset"Si2_valence/outputs/Si2_valence.chk.fmt"
    )
    Hᵏ = transform_gauge(model.eigenvalues, model.gauges)
    Hᴿ = fourier(model.kpoints, Hᵏ, Rspace)
    # ref_Hᴿ is dense (nwann x nwann x nR), Hᴿ is nested
    Hᴿ_dense = cat(Hᴿ...; dims = 3)
    @test all(isapprox.(ref_Hᴿ, Hᴿ_dense; atol = 5.0e-7))
end

@testitem "invfourier" begin
    using Wannier.Datasets
    hamiltonian, position = read_w90_tb(dataset"Si2_valence/outputs/WS/Si2_valence")
    ref_band = read_w90_band(dataset"Si2_valence/outputs/WS/Si2_valence")
    kpoints = ref_band.kpoints

    Hᵏ = invfourier(hamiltonian, kpoints)
    Hᵏ_nested = Hᵏ isa AbstractArray{<:Any, 3} ? [Matrix(Hᵏ[:, :, ik]) for ik in axes(Hᵏ, 3)] : Hᵏ
    eigenvals = hcat([eigen(m).values for m in Hᵏ_nested]...)
    @test all(isapprox.(eigenvals, ref_band.eigenvalues; atol = 2.0e-5))
end
