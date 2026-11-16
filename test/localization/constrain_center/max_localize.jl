@testmodule MaxlocCenterEnv begin
    # This code runs once and the module is cached
    using Wannier
    using Wannier: Vec3
    using Wannier.Datasets
    export model, fg!, p

    model = read_w90(dataset"Si2_valence_coarse/Si2")
    r₀ = [Vec3(0.0, 0.0, 0.0) for i in 1:n_wannier(model)]
    λ = 10.0
    p = CenterSpreadPenalty(r₀, λ)
    fg! = Wannier.get_fg!_maxloc(p, model)
end

@testitem "constraint center maxloc spread gradient" setup = [MaxlocCenterEnv] begin
    using NLSolversBase

    U0 = stack(model.gauges)

    # analytical gradient
    G = similar(U0)
    fg!(nothing, G, U0)

    # finite diff gradient
    d = OnceDifferentiable(x -> fg!(1.0, nothing, x), U0, zero(eltype(real(U0))))
    G_ref = NLSolversBase.gradient!(d, U0)

    # I am using a looser tolerance here
    @test isapprox(G, G_ref; atol = 1.0e-6)

    # Test 2nd iteration
    U1 = Wannier.max_localize(p, model; max_iter = 1)
    U1 = stack(U1)

    fg!(nothing, G, U1)
    d = OnceDifferentiable(x -> fg!(1.0, nothing, x), U1, zero(eltype(real(U1))))
    G_ref = NLSolversBase.gradient!(d, U1)
    @test isapprox(G, G_ref; atol = 1.0e-6)
end

@testitem "constraint center maxloc valence" setup = [MaxlocCenterEnv] begin
    Umin = Wannier.max_localize(p, model; max_iter = 4)
    Ω = Wannier.omega(p, model.kstencil, model.overlaps, Umin)

    # display(Ω)
    @test Ω.Ω ≈ Ω.ΩI + Ω.Ω̃
    @test Ω.Ω̃ ≈ Ω.ΩOD + Ω.ΩD
    @test isapprox(Ω.Ω, 30.212541786847897; atol = 1.0e-7)
    @test isapprox(Ω.ΩI, 3.706376531801815; atol = 1.0e-7)
    @test isapprox(Ω.ΩOD, 6.166406390333415; atol = 1.0e-7)
    @test isapprox(Ω.ΩD, 20.339758877909578; atol = 1.0e-7)

    @test isapprox(
        Ω.ω,
        [7.281420298699552, 7.517458772963187, 8.485999692493612, 6.927663022691542];
        atol = 1.0e-7,
    )
    @test isapprox(
        Ω.r,
        [
            [-0.07121670117433276, -0.06864981990353497, -0.1700526914145371],
            [-0.1584367148089656, -0.7074236492675717, 0.7030443914449187],
            [-0.12106701884890392, -0.10699903912721295, -0.07752478152918568],
            [0.007623180718356191, -0.16940228475664315, 0.5054673077101564],
        ];
        atol = 1.0e-7,
    )

    @test Ω.Ωt ≈ Ω.Ω + Ω.Ωc
    @test isapprox(Ω.Ωc, 13.748929179479767; atol = 1.0e-7)
    @test isapprox(Ω.Ωt, 43.96147096632767; atol = 1.0e-7)
    @test isapprox(
        Ω.ωc,
        [0.3870253415626978, 10.198218284846618, 0.3211610917825568, 2.842524461287894];
        atol = 1.0e-7,
    )
    @test isapprox(
        Ω.ωt,
        [7.66844564026225, 17.715677057809806, 8.807160784276169, 9.770187483979436];
        atol = 1.0e-7,
    )
    @test Ω.ωt ≈ Ω.ω + Ω.ωc
end
