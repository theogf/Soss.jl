using Soss

get_distname(x::Symbol) = Symbol(:_, x, :_dist)

"""
    withmeasures(m::Model) -> Model

```@repl
m = @model begin
    σ ~ HalfNormal()
    y ~ For(10) do j
        Normal(0,σ)
    end
end;

m_dists = Soss.withmeasures(m)
@model begin
        _σ_dist = HalfNormal()
        σ ~ _σ_dist
        _y_dist = For(10) do j
                Normal(0, σ)
            end
        y ~ _y_dist
    end

ydist = rand(m_dists())._y_dist

rand(ydist)
```
"""
function withmeasures(m::Model)
    theModule = getmodule(m)
    m_init = Model(theModule, m.args, NamedTuple(), NamedTuple(), nothing)

    function proc(st::Sample) 
        distname = get_distname(st.x)
        assgn = Model(theModule, Assign(distname, st.rhs))
        sampl = Model(theModule, Sample(st.x, distname))
        return merge(assgn, sampl)
    end
    
    proc(st::Arg) = nothing
    proc(st) = Model(theModule, st)

    # Rewrite the statements of the model one by one. 
    m_new = foldl(statements(m); init=m_init) do m0,st
        merge(m0, proc(st))
    end
    return m_new
end

function withmeasures(d::ConditionalModel)
    withmeasures(Model(d))(argvals(d)) | observations(d)
end

# TODO: Finish this
# function predict_measure(rng::AbstractRNG, d::ConditionalModel, post::AbstractVector{<:NamedTuple{N}}) where {N}
