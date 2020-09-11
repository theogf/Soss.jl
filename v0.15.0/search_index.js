var documenterSearchIndex = {"docs":
[{"location":"api/","page":"Soss API","title":"Soss API","text":"CurrentModule = Soss","category":"page"},{"location":"api/#API","page":"Soss API","title":"API","text":"","category":"section"},{"location":"api/","page":"Soss API","title":"Soss API","text":"","category":"page"},{"location":"api/","page":"Soss API","title":"Soss API","text":"Modules = [Soss]","category":"page"},{"location":"api/#Soss.MarkovChain","page":"Soss API","title":"Soss.MarkovChain","text":"MarkovChain\n\nMarkovChain(pars, step) defines a Markov Chain with global parameters pars and transition kernel step. Here, pars is a named tuple, and step is a Soss model that takes arguments (pars, state) and returns a next value containing the new pars and state.\n\nNOTE: This is experimental, and may change in the near future.\n\nmstep = @model pars,state begin\n    σ = pars.σ\n    x0 = state.x\n    x ~ Normal(x0, σ)\n    next = (pars=pars, state=(x=x,))\nend;\n\nm = @model s0 begin\n    σ ~ Exponential()\n    pars = (σ=σ,)\n    chain ~ MarkovChain(pars, mstep(pars=pars, state=s0))\nend;\n\nr = rand(m(s0=(x=2,),));\n\nfor s in Iterators.take(r.chain,3)\n    println(s)\nend\n\n# output\n\n(x = -6.596883394256064,)\n(x = 0.48200039561318864,)\n(x = -2.838556784903994,)\n\n\n\n\n\n","category":"type"},{"location":"api/#Soss.Do-Tuple{Model,Vararg{Any,N} where N}","page":"Soss API","title":"Soss.Do","text":"Do(m, xs...)\n\nReturns a model transformed by adding xs... to arguments. The remainder of the body remains the same, consistent with Judea Pearl's \"Do\" operator. Unneeded arguments are trimmed.\n\nExamples\n\nm = @model (n, k) begin\n    β ~ Gamma()\n    α ~ Gamma()\n    θ ~ Beta(α, β)\n    x ~ Binomial(n, θ)\n    z ~ Binomial(k, α / (α + β))\nend;\nDo(m, :θ)\n\n# output\n@model (n, k, θ) begin\n        β ~ Gamma()\n        α ~ Gamma()\n        x ~ Binomial(n, θ)\n        z ~ Binomial(k, α / (α + β))\n    end\n\n\n\n\n\n","category":"method"},{"location":"api/#Soss.advancedHMC-Union{Tuple{B}, Tuple{A}, Tuple{Soss.JointDistribution{A,B,B1,M} where M where B1,Any}, Tuple{Soss.JointDistribution{A,B,B1,M} where M where B1,Any,Any}} where B where A","page":"Soss API","title":"Soss.advancedHMC","text":"advancedHMC(m, data, N = 1000; n_adapts = 1000)\n\nDraw N samples from the posterior distribution of parameters defined in Soss model m, conditional on data. Samples are drawn using Hamiltonial Monte Carlo (HMC) from the advancedHMC.jl package.\n\nKeywords\n\nn_adapts = 1000: The number of interations used to set HMC parameters.\n\nReturns a tuple of length 2:\n\nSamples from the posterior distribution of parameters.\nSample summary statistics.\n\nExample\n\nm = @model x begin\n    β ~ Normal()\n    yhat = β .* x\n    y ~ For(eachindex(x)) do j\n        Normal(yhat[j], 2.0)\n    end\nend\n\nx = randn(10);\ntruth = [0.61, -0.34, -1.74];\n\npost = advancedHMC(m(x=x), (y=truth,));\nE_β = mean(post[1])[1]\n\nprintln(\"Posterior mean β: \" * string(round(E_β, digits=2)))\n\n\n\n\n\n","category":"method"},{"location":"api/#Soss.after-Tuple{Model,Vararg{Any,N} where N}","page":"Soss API","title":"Soss.after","text":"after(m::Model, xs...; strict=false)\n\nTransforms m by moving xs to arguments. If strict=true, only descendants of xs are retained in the body. Otherwise, the remaining variables in the body are unmodified. Unused arguments are trimmed.\n\npredictive(m::Model, xs...) = after(m, xs..., strict = true)\n\nDo(m::Model, xs...) = after(m, xs..., strict = false)\n\nExample\n\nm = @model (n, k) begin\n    β ~ Gamma()\n    α ~ Gamma()\n    θ ~ Beta(α, β)\n    x ~ Binomial(n, θ)\n    z ~ Binomial(k, α / (α + β))\nend;\nSoss.after(m, :α)\n\n# output\n@model (n, k, α) begin\n        β ~ Gamma()\n        θ ~ Beta(α, β)\n        x ~ Binomial(n, θ)\n        z ~ Binomial(k, α / (α + β))\n    end\n\n\n\n\n\n","category":"method"},{"location":"api/#Soss.before-Tuple{Model,Vararg{Any,N} where N}","page":"Soss API","title":"Soss.before","text":"before(m::Model, xs...; inclusive=true, strict=true)\n\nTransforms m by retaining all ancestors of any of xs if strict=true; if strict=false, retains all variables that are not descendants of any xs. Note that adding more variables to xs cannot result in a larger model. If inclusive=true, xs is considered to be an ancestor of itself and is always included in the returned Model. Unneeded arguments are trimmed.\n\nprune(m::Model, xs...) = before(m, xs..., inclusive = false, strict = false)\n\nprior(m::Model, xs...) = before(m, xs..., inclusive = true, strict = true)\n\nExamples\n\nm = @model (n, k) begin\n    β ~ Gamma()\n    α ~ Gamma()\n    θ ~ Beta(α, β)\n    x ~ Binomial(n, θ)\n    z ~ Binomial(k, α / (α + β))\nend;\nSoss.before(m, :θ, inclusive = true, strict = false)\n\n# output\n@model k begin\n        β ~ Gamma()\n        α ~ Gamma()\n        θ ~ Beta(α, β)\n        z ~ Binomial(k, α / (α + β))\n    end\n\n\n\n\n\n","category":"method"},{"location":"api/#Soss.dynamicHMC","page":"Soss API","title":"Soss.dynamicHMC","text":"dynamicHMC(\n    rng::AbstractRNG,\n    m::JointDistribution,\n    _data,\n    N::Int = 1000;\n    method = logpdf,\n    ad_backend = Val(:ForwardDiff),\n    reporter = DynamicHMC.NoProgressReport(),\n    kwargs...)\n\nDraw N samples from the posterior distribution of parameters defined in Soss model m, conditional on _data. Samples are drawn using Hamiltonial Monte Carlo (HMC) from the DynamicHMC.jl package.\n\nThis function is essentially a wrapper around DynamicHMC.mcmc_with_warmup(). Arguments reporter, ad_backend DynamicHMC docs here)\n\nArguments\n\nrng: Random number generator.\nm: Soss model.\n_data: NamedTuple of data to condition on.\n\nKeyword Arguments\n\nN = 1000: Number of samples to draw.\nmethod = logpdf: How to compute the log-density. Options are logpdf (delegates to logpdf of each component) or codegen (symbolic simplification and code generation).\nad_backend = Val(:ForwardDiff): Automatic differentiation backend.\nreporter = DynamicHMC.NoProgressReport(): Specify logging during sampling. Default: do not log progress.\nkwargs: Additional keyword arguments passed to core sampling function DynamicHMC.mcmc_with_warmup().\n\nReturns an Array of Namedtuple of length N. Each entry in the array is a sample of parameters indexed by the parameter symbol.\n\nExample\n\nusing StableRNGs\nrng = StableRNG(42);\n\nm = @model x begin\n    β ~ Normal()\n    yhat = β .* x\n    y ~ For(eachindex(x)) do j\n        Normal(yhat[j], 2.0)\n    end\nend\n\nx = randn(rng, 3);\ntruth = [-0.41, 1.21, 0.11];\n\npost = dynamicHMC(rng, m(x=x), (y=truth,));\nE_β = mean(getfield.(post, :β))\n\nprintln(\"Posterior mean β: \" * string(round(E_β, digits=2)))\n\n# output\nPosterior mean β: 0.25\n\n\n\n\n\n","category":"function"},{"location":"api/#Soss.predictive-Tuple{Model,Vararg{Any,N} where N}","page":"Soss API","title":"Soss.predictive","text":"predictive(m, xs...)\n\nReturns a model transformed by adding xs... to arguments with a body containing only statements that depend on xs, or statements that are depended upon by children of xs through an open path. Unneeded arguments are trimmed.\n\nExamples\n\nm = @model (n, k) begin\n    β ~ Gamma()\n    α ~ Gamma()\n    θ ~ Beta(α, β)\n    x ~ Binomial(n, θ)\n    z ~ Binomial(k, α / (α + β))\nend;\npredictive(m, :θ)\n\n# output\n@model (n, θ) begin\n        x ~ Binomial(n, θ)\n    end\n\n\n\n\n\n","category":"method"},{"location":"api/#Soss.prior-Tuple{Model,Vararg{Any,N} where N}","page":"Soss API","title":"Soss.prior","text":"prior(m, xs...)\n\nReturns the minimal model required to sample random variables xs.... Useful for extracting a prior distribution from a joint model m by designating xs... and the variables they depend on as the prior and hyperpriors.\n\nExample\n\nm = @model n begin\n    α ~ Gamma()\n    β ~ Gamma()\n    θ ~ Beta(α,β)\n    x ~ Binomial(n, θ)\nend;\nSoss.prior(m, :θ)\n\n# output\n@model begin\n        β ~ Gamma()\n        α ~ Gamma()\n        θ ~ Beta(α, β)\n    end\n\n\n\n\n\n","category":"method"},{"location":"api/#Soss.prune-Tuple{Model,Vararg{Any,N} where N}","page":"Soss API","title":"Soss.prune","text":"prune(m, xs...)\n\nReturns a model transformed by removing xs... and all variables that depend on xs.... Unneeded arguments are also removed.\n\nExamples\n\nm = @model n begin\n    α ~ Gamma()\n    β ~ Gamma()\n    θ ~ Beta(α,β)\n    x ~ Binomial(n, θ)\nend;\nprune(m, :θ)\n\n# output\n@model begin\n        β ~ Gamma()\n        α ~ Gamma()\n    end\n\nm = @model n begin\n    α ~ Gamma()\n    β ~ Gamma()\n    θ ~ Beta(α,β)\n    x ~ Binomial(n, θ)\nend;\nprune(m, :n)\n\n# output\n@model begin\n        β ~ Gamma()\n        α ~ Gamma()\n        θ ~ Beta(α, β)\n    end\n\n\n\n\n\n","category":"method"},{"location":"api/#Soss.withdistributions-Tuple{Model}","page":"Soss API","title":"Soss.withdistributions","text":"withdistributions(m::Model) -> Model\n\njulia> m = @model begin     σ ~ HalfNormal()     y ~ For(10) do j         Normal(0,σ)     end end;\n\njulia> mdists = Soss.withdistributions(m) @model begin         _σdist = HalfNormal()         σ ~ σdist         ydist = For(10) do j                 Normal(0, σ)             end         y ~ ydist     end\n\njulia> ydist = rand(mdists()).y_dist For{GeneralizedGenerated.Closure{function = (σ, M, j;) -> begin     M.Normal(0, σ) end,Tuple{Float64,Module}},Tuple{Int64},Normal{Float64},Float64}(GeneralizedGenerated.Closure{function = (σ, M, j;) -> begin     M.Normal(0, σ) end,Tuple{Float64,Module}}((0.031328640120683524, Main)), (10,))\n\njulia> rand(ydist) 10-element Array{Float64,1}:   0.03454891487870426   0.008832782323408313  -0.007395186925623771  -0.030669004243492004  -0.01728630026691135   0.011892877715064682   0.025576319363013512  -0.029323425779917773  -0.020502677724193594   0.04612690097957398\n\n\n\n\n\n","category":"method"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"EditURL = \"https://github.com/cscherrer/Soss.jl/blob/master/examples/example-linear-regression.jl\"","category":"page"},{"location":"example-linear-regression/#Example:-Linear-regression","page":"Linear regression","title":"Example: Linear regression","text":"","category":"section"},{"location":"example-linear-regression/#Defining-the-linear-regression-model","page":"Linear regression","title":"Defining the linear regression model","text":"","category":"section"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"In this example, we fit a Bayesian linear regression model with the canonical link function.","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"Suppose that we are given a matrix of features X and a column vector of labels y. X has n rows and p columns. y has n elements. We assume that our observation vector y is a realization of a random variable Y. We define μ (mu) as the expected value of Y, i.e. μ := E[Y]. Our model comprises three components:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"The probability distribution of Y: for linear regression, we assume that each Yᵢ follows a normal distribution with mean μᵢ and variance σ².\nThe systematic component, which consists of linear predictor η (eta), which we define as η := α + Xβ, where α is the scalar intercept and β is the column vector of p coefficients.\nThe link function g, which provides the following relationship: g(E[Y]) = g(μ) = η = Xβ. It follows that μ = g⁻¹(η), where g⁻¹ denotes the inverse of g. For linear regression, the canonical link function is the identity function. Therefore, when using the canonical link function, μ = g⁻¹(η) = η.","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"In this model, the parameters that we want to estimate are α, β, and σ. We need to select prior distributions for these parameters. For α, we choose a normal distribution with zero mean and unit variance. For each βᵢ, we choose a normal distribution with zero mean and unit variance. Here, βᵢ denotes the ith component of β. For σ, we will choose a half-normal distribution with unit variance.","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"We define this model using Soss:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"using Soss\nusing Random\n\nmodel = @model X begin\n    p = size(X, 2) # number of features\n    α ~ Normal(0, 1) # intercept\n    β ~ Normal(0, 1) |> iid(p) # coefficients\n    σ ~ HalfNormal(1) # dispersion\n    η = α .+ X * β # linear predictor\n    μ = η # `μ = g⁻¹(η) = η`\n    y ~ For(eachindex(μ)) do j\n        Normal(μ[j], σ) # `Yᵢ ~ Normal(mean=μᵢ, variance=σ²)`\n    end\nend;\nnothing #hide","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"In Soss, models are first-class and function-like, and applying a model to its arguments gives a joint distribution.","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"Just a few of the things we can do in Soss:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"Sample from the forward model\nCondition a joint distribution on a subset of parameters\nHave arbitrary Julia values (yes, even other models) as inputs or outputs of a model\nBuild a new model for the predictive distribution, for assigning parameters to particular values","category":"page"},{"location":"example-linear-regression/#Sampling-from-the-forward-model","page":"Linear regression","title":"Sampling from the forward model","text":"","category":"section"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"First, create some fake data:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"X = randn(6,2)","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"Now, sample from the forward model:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"forward_sample = rand(model(X=X))","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"The pairs function can make this a little easier to read:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"pairs(forward_sample)","category":"page"},{"location":"example-linear-regression/#Use-MCMC-to-sample-from-the-posterior-distribution","page":"Linear regression","title":"Use MCMC to sample from the posterior distribution","text":"","category":"section"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"First, generate some fake data:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"num_rows = 1_000\nnum_features = 2\nX = randn(num_rows, num_features)","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"Pick the true values for our coefficients β:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"β_true = [2.0, -1.0]","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"We also need to pick a true value for the intercept α:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"α_true = 1.0","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"And we also need to pick a true value for the dispersion parameter σ","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"σ_true = 0.5","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"Now, generate the true labels:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"η_true = α_true .+ X * β_true\nμ_true = η_true\nnoise = randn(num_rows) .* σ_true\ny_true = μ_true .+ noise","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"Now we use MCMC (specifically, the No-U-turn sampler) to sample from the posterior distribution:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"posterior = dynamicHMC(model(X=X), (y=y_true,))","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"Often, the posterior distributions are easier to work with in terms of particles (built using MonteCarloMeasurements.jl):","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"particles(posterior)","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"Again, the pairs function can make this a little easier to read:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"pairs(particles(posterior))","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"Compare the posterior distributions on σ, α, and β to the true values:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"@show σ_true; @show α_true; @show β_true;\nnothing #hide","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"We did a pretty good job at recovering the true parameter values!","category":"page"},{"location":"example-linear-regression/#Construct-the-posterior-predictive-distribution","page":"Linear regression","title":"Construct the posterior predictive distribution","text":"","category":"section"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"For model diagnostics and prediction, we need the posterior predictive distribution:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"posterior_predictive = predictive(model, :β)","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"This requires X and β as inputs, so we can do something like this to do a posterior predictive check (PPC)","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"y_ppc = [rand(posterior_predictive(;X=X, p...)).y for p in posterior]","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"We can compare the posterior predictive distribution on y to the true values of y:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"y_true - particles(y_ppc)","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"These play a role similar to that of residuals in a non-Bayesian approach (there's plenty more detail to go into, but that's for another time).","category":"page"},{"location":"example-linear-regression/#So,-what's-really-happening-here?","page":"Linear regression","title":"So, what's really happening here?","text":"","category":"section"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"Under the hood, rand and logpdf specify different ways of \"running\" the model.","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"rand  turns each v ~ dist into v = rand(dist), finally outputting the NamedTuple of all values it has seen.","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"logpdf steps through the same program, but instead accumulates a log-density. It begins by initializing _ℓ = 0.0. Then at each step, it turns v ~ dist into _ℓ += logpdf(dist, v), before finally returning _ℓ.","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"Note that I said \"turns into\" instead of \"interprets\". Soss uses GG.jl to generate specialized code for a given model, inference primitive (like rand and logpdf), and type of data.","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"This idea can be used in much more complex ways. weightedSample is a sort of hybrid between rand and logpdf. For data that are provided, it increments a _ℓ using logpdf. Unknown values are sampled using rand.","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"ℓ, proposal = weightedSample(model(X=X), (y=y_true,));\nnothing #hide","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"ℓ:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"ℓ","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"proposal.β:","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"proposal.β","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"Again, there's no runtime check needed for this. Each of these is compiled the first time it is called, so future calls are very fast. Functions like this are great to use in tight loops.","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"","category":"page"},{"location":"example-linear-regression/","page":"Linear regression","title":"Linear regression","text":"This page was generated using Literate.jl.","category":"page"},{"location":"misc/#Models-and-JointDistributions","page":"Miscellaneous","title":"Models and JointDistributions","text":"","category":"section"},{"location":"misc/","page":"Miscellaneous","title":"Miscellaneous","text":"A Model in Soss","category":"page"},{"location":"misc/#Model-Combinators","page":"Miscellaneous","title":"Model Combinators","text":"","category":"section"},{"location":"misc/#Building-Inference-Algorithms","page":"Miscellaneous","title":"Building Inference Algorithms","text":"","category":"section"},{"location":"misc/#Inference-Primitives","page":"Miscellaneous","title":"Inference Primitives","text":"","category":"section"},{"location":"misc/","page":"Miscellaneous","title":"Miscellaneous","text":"At its core, Soss is about source code generation. Instances of this are referred to as inference primitives, or simply \"primitives\". As a general rule, new primitives are rarely needed. A wide variety of inference algorithms can be built using what's provided.","category":"page"},{"location":"misc/","page":"Miscellaneous","title":"Miscellaneous","text":"To easily find all available inference primitives, enter Soss.source<TAB> at a REPL. Currently this returns this result:","category":"page"},{"location":"misc/","page":"Miscellaneous","title":"Miscellaneous","text":"julia> Soss.source\nsourceLogpdf         sourceRand            sourceXform\nsourceParticles      sourceWeightedSample","category":"page"},{"location":"misc/","page":"Miscellaneous","title":"Miscellaneous","text":"The general pattern is that a primitive sourceFoo specifies how code is generated for an inference function foo.","category":"page"},{"location":"misc/","page":"Miscellaneous","title":"Miscellaneous","text":"For more details on inference primitives, see the Internals section.","category":"page"},{"location":"misc/#Inference-Functions","page":"Miscellaneous","title":"Inference Functions","text":"","category":"section"},{"location":"misc/","page":"Miscellaneous","title":"Miscellaneous","text":"An inference function is a function that takes a JointDistribution as an argument, and calls at least one inference primitive (not necessarily directly). The wrapper around each primitive is a special case of this, but most inference functions work at a higher level of abstraction.","category":"page"},{"location":"misc/","page":"Miscellaneous","title":"Miscellaneous","text":"There's some variability , but is often of the form","category":"page"},{"location":"misc/","page":"Miscellaneous","title":"Miscellaneous","text":"foo(d::JointDistribution, data::NamedTuple)","category":"page"},{"location":"misc/","page":"Miscellaneous","title":"Miscellaneous","text":"For example, advancedHMC uses TuringLang/AdvancedHMC.jl , which needs a logpdf and its gradient.","category":"page"},{"location":"misc/","page":"Miscellaneous","title":"Miscellaneous","text":"Most inference algorithms can be expressed in terms of inference primitives.","category":"page"},{"location":"misc/#Chain-Combinators","page":"Miscellaneous","title":"Chain Combinators","text":"","category":"section"},{"location":"misc/#Internals","page":"Miscellaneous","title":"Internals","text":"","category":"section"},{"location":"misc/#Models","page":"Miscellaneous","title":"Models","text":"","category":"section"},{"location":"misc/","page":"Miscellaneous","title":"Miscellaneous","text":"struct Model{A,B}\n    args  :: Vector{Symbol}\n    vals  :: NamedTuple\n    dists :: NamedTuple\n    retn  :: Union{Nothing, Symbol, Expr}\nend","category":"page"},{"location":"misc/","page":"Miscellaneous","title":"Miscellaneous","text":"function sourceWeightedSample(_data)\n    function(_m::Model)\n\n        _datakeys = getntkeys(_data)\n        proc(_m, st :: Assign)     = :($(st.x) = $(st.rhs))\n        proc(_m, st :: Return)     = nothing\n        proc(_m, st :: LineNumber) = nothing\n\n        function proc(_m, st :: Sample)\n            st.x ∈ _datakeys && return :(_ℓ += logpdf($(st.rhs), $(st.x)))\n            return :($(st.x) = rand($(st.rhs)))\n        end\n\n        vals = map(x -> Expr(:(=), x,x),variables(_m))\n\n        wrap(kernel) = @q begin\n            _ℓ = 0.0\n            $kernel\n\n            return (_ℓ, $(Expr(:tuple, vals...)))\n        end\n\n        buildSource(_m, proc, wrap) |> flatten\n    end\nend\n","category":"page"},{"location":"to-do-list/","page":"To-Do List","title":"To-Do List","text":"CurrentModule = Soss","category":"page"},{"location":"to-do-list/#To-Do-List","page":"To-Do List","title":"To-Do List","text":"","category":"section"},{"location":"to-do-list/","page":"To-Do List","title":"To-Do List","text":"We need a way to \"lift\" a \"Distribution\" (without parameters, so really a family) to a Model, or one with parameters to a JointDistribution","category":"page"},{"location":"to-do-list/","page":"To-Do List","title":"To-Do List","text":"Models are \"function-like\", so a JointDistribution should be sometimes usable as a value. m1(m2(args)) should work.","category":"page"},{"location":"to-do-list/","page":"To-Do List","title":"To-Do List","text":"This also means m1 ∘ m2 should be fine","category":"page"},{"location":"to-do-list/","page":"To-Do List","title":"To-Do List","text":"Since inference primitives are specialized for the type of data, we can include methods for Union{Missing, T} data. PyMC3 has something like this, but for us it will be better since we know at compile time whether any data are missing.","category":"page"},{"location":"to-do-list/","page":"To-Do List","title":"To-Do List","text":"There's a return available in case you want a result other than a NamedTuple, but it's a little fiddly still. I think whether the return is respected or ignored should depend on the inference primitive. And some will also modify it, similar to how a state monad works. Likelihood weighting is an example of this.","category":"page"},{"location":"to-do-list/","page":"To-Do List","title":"To-Do List","text":"Rather than having lots of functions for inference, anything that's not a primitive should (I think for now at least) be a method of... let's call it sample. This should always return an iterator, so we can combine results after the fact using tools like IterTools, ResumableFunctions, and Transducers.","category":"page"},{"location":"to-do-list/","page":"To-Do List","title":"To-Do List","text":"This situation just described is for generating a sequence of samples from a single distribution. But we may also have models with a sequence of distributions, either observed or sampled, or a mix. This can be something like Haskell's iterateM, though we need to think carefully about the specifics.","category":"page"},{"location":"to-do-list/","page":"To-Do List","title":"To-Do List","text":"We already have a way to merge models, we should look into intersection as well.","category":"page"},{"location":"to-do-list/","page":"To-Do List","title":"To-Do List","text":"We need ways to interact with Turing and Gen. Some ideas:","category":"page"},{"location":"to-do-list/","page":"To-Do List","title":"To-Do List","text":"Turn a Soss model into an \"outside\" (Turing or Gen) model\nEmbed outside models as a black box in a Soss model, using their methods for inference","category":"page"},{"location":"to-do-list/","page":"To-Do List","title":"To-Do List","text":"We are working on the SossMLJ package, which will provide an interface between Soss and the MLJ machine learning framework.","category":"page"},{"location":"installing-soss/","page":"Installing Soss","title":"Installing Soss","text":"CurrentModule = Soss","category":"page"},{"location":"installing-soss/#Getting-Started","page":"Installing Soss","title":"Getting Started","text":"","category":"section"},{"location":"installing-soss/","page":"Installing Soss","title":"Installing Soss","text":"Soss is an officially registered package, so to add it to your project you can type","category":"page"},{"location":"installing-soss/","page":"Installing Soss","title":"Installing Soss","text":"julia> import Pkg; Pkg.add(\"Soss\")","category":"page"},{"location":"installing-soss/","page":"Installing Soss","title":"Installing Soss","text":"within the julia REPL and you are ready for using Soss. If it fails to precompile, it could be due to one of the following:","category":"page"},{"location":"installing-soss/","page":"Installing Soss","title":"Installing Soss","text":"You have gotten an old version due to compatibility restrictions with your current environment.","category":"page"},{"location":"installing-soss/","page":"Installing Soss","title":"Installing Soss","text":"Should that happen, create a new folder for your Soss project, launch a julia session within, type","category":"page"},{"location":"installing-soss/","page":"Installing Soss","title":"Installing Soss","text":"julia> import Pkg; Pkg.activate(pwd())","category":"page"},{"location":"installing-soss/","page":"Installing Soss","title":"Installing Soss","text":"and start again. More information on julia projects here.","category":"page"},{"location":"installing-soss/","page":"Installing Soss","title":"Installing Soss","text":"You have set up PyCall to use a python distribution provided by yourself. If that is the case, make sure to install the missing python dependencies, as listed in the precompilation error. More information on PyCall's python version here.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Soss","category":"page"},{"location":"#Soss","page":"Home","title":"Soss","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Soss is a library for probabilistic programming.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The source code is available in the GitHub repository.","category":"page"},{"location":"internals/","page":"Internals","title":"Internals","text":"Soss needs the body of a model to be of the form","category":"page"},{"location":"internals/","page":"Internals","title":"Internals","text":"begin\n    line_1\n    ⋮\n    line_n\nend","category":"page"},{"location":"internals/","page":"Internals","title":"Internals","text":"Each line is syntactically translated into a Statement. This is an abstract type, with subtypes Assign and Sample. For example,","category":"page"},{"location":"internals/","page":"Internals","title":"Internals","text":"x ~ Normal(μ,σ)","category":"page"},{"location":"internals/","page":"Internals","title":"Internals","text":"becomes","category":"page"},{"location":"internals/","page":"Internals","title":"Internals","text":"Sample(:x, :(Normal(μ,σ)))","category":"page"},{"location":"internals/","page":"Internals","title":"Internals","text":"Next, all of the Samples are brought together to build a named tuple mapping each Symbol to its Expr. This becomes the dists field for a Model.","category":"page"},{"location":"internals/","page":"Internals","title":"Internals","text":"Because all of this is entirely syntactic, translating into another form only helps when its done on the right side of ~ or =. Otherwise we need another way to represent this information.","category":"page"},{"location":"sossmlj/","page":"SossMLJ.jl","title":"SossMLJ.jl","text":"CurrentModule = Soss","category":"page"},{"location":"sossmlj/#SossMLJ.jl","page":"SossMLJ.jl","title":"SossMLJ.jl","text":"","category":"section"},{"location":"sossmlj/","page":"SossMLJ.jl","title":"SossMLJ.jl","text":"The SossMLJ.jl package integrates Soss.jl into the MLJ.jl machine learning framework.","category":"page"},{"location":"sossmlj/","page":"SossMLJ.jl","title":"SossMLJ.jl","text":"More details are available in the SossMLJ.jl GitHub repository.","category":"page"}]
}
