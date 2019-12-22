using LinearAlgebra: inv, rank
using SparseArrays: spzeros

"""
    typename(system::AbstractSystem)

Returns the base type of `system` without parameter information.

### Input

- `system` -- `AbstractSystem`

### Output

Returns base-type of `system`.

"""
function typename(system::AbstractSystem)
    return Base.typename(typeof(system)).wrapper
end

"""
    _complementary_type(system_type::Type{<:AbstractSystem})

Return the complementary type of a system type `system_type`.

For a `system_type<:AbstractDiscreteSystem`, the complementary
`AbstractContinuousSystem` type is returned and vice versa.

### Input

- `system_type` -- type of `AbstractSystem`

### Ouput

Return complementary type of `system_type`.

### Note
To get the `_complementary_type` of a `system<:AbstractSystem` use
`_complementary_type(typename(system))`.
"""
@generated function _complementary_type(type::Type{<:AbstractSystem})
    # type is a Type{<:AbstractSystem}, e.g. Type{AffineContinuousSystem}
    # extract the system type information with
    system_type = type.parameters[1]
    type_string = string(system_type)
    if supertype(system_type) == AbstractDiscreteSystem
        type_string = replace(type_string, "Discrete"=>"Continuous")
    elseif supertype(system_type) == AbstractContinuousSystem
        type_string = replace(type_string, "Continuous"=>"Discrete")
    else
        error("$system_type <: $(supertype(system_type)) is neither discrete nor continuous")
    end
    return Meta.parse(type_string)
end


"""
     discretize(system::AbstractContinuousSystem, ΔT::Real; algorithm=:default)

Discretization of a `isaffine` `AbstractContinuousSystem` to a
`AbstractDiscreteSystem` with discretization time `ΔT` using the exact
discretization algorithm if possible.

### Input

- `system` -- a affine continuous system
- `ΔT` -- discretization time
- `algorithm` -- (optional, default=`:default`) discretization algorithm

### Output

Returns a discretization of the input system `system` with discretization time `ΔT`.

### Algorithm

Consider a `NoisyAffineControlledContinuousSystem` with system dynamics
`x' = Ax + Bu + c + Du`.

If A is invertible:
The exact discretization is calculated by solving the integral for
`t = [t, t+ΔT]` for a fixed input `u` and fixed noise realisation `w` which
writes as `x⁺ = Aᵈx + Bᵈu + cᵈ + Dᵈu` where `Aᵈ = exp(A⋅ΔT)`,
`Bᵈ = inv(A)⋅(Aᵈ - I)⋅B`, `cᵈ = inv(A)⋅(Aᵈ - I)⋅c` and `Dᵈ = inv(A)⋅(Aᵈ - I)⋅D`.


If A is not invertible:
A first order approximation of the exact discretiziation, the euler
discretization, can be applied, which writes as `x⁺ = Aᵈx + Bᵈu + cᵈ + Dᵈu`
where  `Aᵈ = I + ΔT⋅A`, `Bᵈ = ΔT⋅B`, `cᵈ = ΔT⋅c` and `Dᵈ = ΔT⋅D`.

"""
function discretize(system::AbstractContinuousSystem, ΔT::Real; algorithm=:default)
    noset(x) = x ∉ [:X,:U,:W]
    # get all fields from system
    fields = collect(fieldnames(typeof(system)))
    # get fields of system that are parameter of the system dynamics (no sets)
    # i.e., all fields that needs to be discretized
    values_cont = [getfield(system, f) for f in filter(noset, fields)]
    if algorithm == :default
        if rank(system.A) == size(system.A, 1)
          # A is invertible, use exact discretizaion
          algorithm = :exact
        else
          # A is not invertible, use approximative discretizaion
          algorithm = :euler
        end
    end
    # compute discretized values of dynamics_params_c
    values_disc = _discretize(values_cont..., ΔT; algorithm=algorithm)
    # get fields of system that are sets
    set_values = [getfield(system, f) for f in filter(!noset, fields)]
    # get corresponding discrete type of system
    discrete_type = _complementary_type(typename(system))
    # build the new discrete type with the discretized and set values
    return discrete_type(values_disc..., set_values...)
end

"""
    _discretize(A::AbstractMatrix, B::AbstractMatrix, c::AbstractVector,
                D::AbstractMatrix, ΔT::Real; algorithm=:exact)

Implementation of the discretization algorithm used in `discretize`.

See [`discretize`](@ref) for more details.
"""
function _discretize(A::AbstractMatrix,
                     B::AbstractMatrix,
                     c::AbstractVector,
                     D::AbstractMatrix, ΔT::Real; algorithm=:exact)
    if algorithm == :exact
        A_d = exp(A*ΔT)
        Matr = inv(A)*(A_d - I)
        B_d = Matr*B
        c_d = Matr*c
        D_d = Matr*D
    elseif algorithm == :euler
        A_d = I + ΔT*A
        B_d = ΔT*B
        c_d = ΔT*c
        D_d = ΔT*D
    else
        error("discretization algorithm $algorithm is not known")
    end
    return [A_d, B_d, c_d, D_d]
end

function _discretize(A::AbstractMatrix, ΔT::Real; algorithm=:exact)
    n = size(A,1)
    mzero = spzeros(n, n)
    vzero = spzeros(n)
    A_d, _, _, _ = _discretize(A, mzero, vzero, mzero, ΔT; algorithm=algorithm)
    return [A_d]
end

# works for (:A,:D) and (:A, :B)
function _discretize(A::AbstractMatrix,
                     B::AbstractMatrix, ΔT::Real; algorithm=:exact)
    n = size(A,1)
    mzero = spzeros(n, n)
    vzero = spzeros(n)
    A_d, B_d, _, _ = _discretize(A, B, vzero, mzero, ΔT; algorithm=algorithm)
    return [A_d, B_d]
end

function _discretize(A::AbstractMatrix,
                     c::AbstractVector, ΔT::Real; algorithm=:exact)
    n = size(A,1)
    mzero = spzeros(n, n)
    A_d, _, c_d, _ = _discretize(A, mzero, c, mzero, ΔT; algorithm=algorithm)
    return [A_d, c_d]
end

function _discretize(A::AbstractMatrix,
                     B::AbstractMatrix,
                     c::AbstractVector, ΔT::Real; algorithm=:exact)
    n = size(A,1)
    mzero = spzeros(n, n)
    A_d, B_d, c_d, _ = _discretize(A, B, c, mzero, ΔT; algorithm=algorithm)
    return [A_d, B_d, c_d]
end

function _discretize(A::AbstractMatrix,
                     B::AbstractMatrix,
                     D::AbstractMatrix, ΔT::Real; algorithm=:exact)
    n = size(A,1)
    vzero = spzeros(n)
    A_d, B_d, _, D_d = _discretize(A, B, vzero, D, ΔT; algorithm=algorithm)
    return [A_d, B_d, D_d]
end
