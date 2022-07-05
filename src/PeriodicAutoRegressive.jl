module PeriodicAutoRegressive

using JuMP, Ipopt
using Distributions

import Base: rand

include("utilities.jl")
include("AR1.jl")

export AR1
export model_for_loglikelihood_AR1, initialvalue_optimize!, model_for_loglikelihood_AR1_full
export μₜ, ρₜ, σₜ

end
