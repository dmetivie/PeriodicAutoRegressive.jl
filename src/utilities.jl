interleave2(args...) = collect(Iterators.flatten(zip(args...))) # merge two vector with alternate elements
"""
    n_to_t(N::Int, T::Int)

    This function transforms all index of the chain `n` into their periodic counterpart `t`.
"""
function n_to_t(N::Int, T::Int)
    return [repeat(1:T, N ÷ T); remaining(N - T * (N ÷ T))]
end

remaining(N::Int) = N > 0 ? range(1, length=N) : Int64[]

function polynomial_trigo(t::Number, β, T)
    d = (length(β) - 1) ÷ 2
    if d == 0
        return β[1]
    else
        f = 2π / T
        # everything is shifted from 1 from usual notation due to array starting at 1
        return β[1] + sum(β[2*l] * cos(f * l * t) + β[2*l+1] * sin(f * l * t) for l = 1:d)
    end
end

function polynomial_trigo(t::AbstractArray, β, T)
    d = (length(β) - 1) ÷ 2
    if d == 0
        return β[1]
    else
        f = 2π / T
        # everything is shifted from 1 from usual notation due to array starting at 1
        return β[1] .+ sum(β[2*l] * cos.(f * l * t) + β[2*l+1] * sin.(f * l * t) for l = 1:d)
    end
end

μₜ(t, θ::AbstractArray, T) = polynomial_trigo(t, θ[:], T)
ρₜ(t, θ::AbstractArray, T) = 2 / (1 + exp(-polynomial_trigo(t, θ[:], T))) - 1
σₜ(t, θ::AbstractArray, T) = exp(polynomial_trigo(t, θ[:], T))