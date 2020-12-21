using Espresso: matchex
using LinearAlgebra: I
using MacroTools: @capture
using InteractiveUtils: subtypes

import Base: sort

# ========================
# @map macro
# ========================

"""
    map(ex, args)

Return an instance of the map type corresponding to the given expression.

### Input

- `ex`   -- an expression defining the map, in the form of an anonymous function
- `args` -- additional optional arguments

### Output

A map that best matches the given expression.

### Examples

Let us first create a linear map using this macro:

```jldoctest
julia> @map x -> [1 0; 0 0]*x
LinearMap{Int64,Array{Int64,2}}([1 0; 0 0])
```

We can create an affine system as well:

```jldoctest
julia> @map x -> [1 0; 0 0]*x + [2, 0]
AffineMap{Int64,Array{Int64,2},Array{Int64,1}}([1 0; 0 0], [2, 0])
```

Additional arguments can be passed to `@map` using the function-call form, i.e.
separating the arguments by commas, and using parentheses around the macro call.
For example, an identity map of dimension 5 can be defined as:

```jldoctest
julia> @map(x -> x, dim=5)
IdentityMap(5)
```
A state constraint on such map can be specified passing the additional argument
`x ∈ X`.

An identity map can alternatively be created by giving a the size of the identity
matrix as `I(n)`, for example:

```jldoctest
julia> @map x -> I(5)*x
IdentityMap(5)
```
"""
macro map(ex, args...)
    dimension = nothing
    x = (ex.args)[1]
    rhs = (ex.args)[2].args[2]
    if x == rhs
        # identity map, eg: @map(x -> x, dim=2)
        if @capture(args[1], (dim = dim_) | (dim: dim_) )
            dimension = dim
        else
            throw(ArgumentError("cannot parse dimension of identity map"))
        end

        # constrained identity map: @map(x -> x, dim=2, x ∈ X)
        if length(args) > 1 && @capture(args[2], x ∈ X_)
            return Expr(:call, ConstrainedIdentityMap, esc(:($(dimension))), esc(:($(X))))
        else
            return Expr(:call, IdentityMap, esc(:($(dimension))))
        end
    end

    throw(ArgumentError("unable to match the given expression to a known map type"))
end

macro map(ex)
    x = (ex.args)[1]
    rhs = (ex.args)[2].args[2]

    # x -> I(n)*x
    # (this rule is more specific than x -> Ax so it should come before it)
    MT = IdentityMap
    pat = Meta.parse("I(_n) * $x")
    matched = matchex(pat, rhs)
    matched != nothing &&
        return Expr(:call, :($MT), esc(:($(matched[:_n]))))

    # x -> Ax
    MT = LinearMap
    pat = Meta.parse("_A * $x")
    matched = matchex(pat, rhs)
    matched != nothing &&
        return Expr(:call, :($MT), esc(:($(matched[:_A]))))

    # x -> Ax + b
    MT = AffineMap
    pat = Meta.parse("_A * $x + _b")
    matched = matchex(pat, rhs)
    matched != nothing &&
        return Expr(:call, :($MT), esc(:($(matched[:_A]))), esc(:($(matched[:_b]))))

    throw(ArgumentError("unable to match the given expression to a known map type"))
end

# ========================
# @system macro
# ========================

const TYPES_CONTINUOUS = subtypes(AbstractContinuousSystem)
const FIELDS_CONTINUOUS = fieldnames.(TYPES_CONTINUOUS)

const TYPES_DISCRETE = subtypes(AbstractDiscreteSystem)
const FIELDS_DISCRETE = fieldnames.(TYPES_DISCRETE)

types_table(::Type{AbstractContinuousSystem}) = TYPES_CONTINUOUS
types_table(::Type{AbstractDiscreteSystem}) = TYPES_DISCRETE

fields_table(::Type{AbstractContinuousSystem}) = FIELDS_CONTINUOUS
fields_table(::Type{AbstractDiscreteSystem}) = FIELDS_DISCRETE

"""
    _corresponding_type(AT::Type{<:AbstractSystem}, fields::Tuple)

Return the system type whose field names match those in `fields`.

### Input

- `AT`     -- abstract system type, which can be either `AbstractContinuousSystem`
              or `AbstractDiscreSystem`
- `fields` -- tuple of field names

### Output

The system type (either discrete or continous, depending on `AT`) whose fields
names correspond to those in `fields`, or an error if the `fields` do not match
any known system type.

### Examples

```jldoctest
julia> using MathematicalSystems: _corresponding_type

julia> _corresponding_type(AbstractContinuousSystem, ((:A),))
LinearContinuousSystem

julia> _corresponding_type(AbstractContinuousSystem, ((:A), (:B), (:X), (:U)))
ConstrainedLinearControlContinuousSystem
```
"""
function _corresponding_type(AT::Type{<:AbstractSystem}, fields::Tuple)
    TYPES = types_table(AT)
    FIELDS = fields_table(AT)
    idx = findall(x -> issetequal(fields, x), FIELDS)
    if isempty(idx)
        throw(ArgumentError("the entry $(fields) does not match a " *
                            "`MathematicalSystems.jl` structure"))
    end
    @assert length(idx) == 1 "found more than one compatible system type"
    return TYPES[idx][1]
end

"""
    _capture_dim(expr)

Return the tuple containing the dimension(s) in `expr`.

### Input

- `expr` -- symbolic expression that can be of any of the following forms:

    - `:x` or `:(x)` -- state dimension
    - `:(x, u)`      -- state and input dimension
    - `:(x, u, w)`   -- state, input and noise dimensions

### Output

- The scalar `x` if `expr` specifies the state dimension.
- The vector `[x, u]` if `expr` specifies state and input dimension.
- The vector `[x, u, w]` if `expr` specifies state, input and noise dimensions.
"""
function _capture_dim(expr)
    # the order of the `if` clauses is important, as e.g.
    # `@capture(expr, (x_, u_, w_))` is a particular case of `@capture(expr, (x_))`.
    if @capture(expr, (x_, u_, w_))
        dims = [x, u, w]
    elseif @capture(expr, (x_, u_))
        dims = [x, u]
    elseif @capture(expr, (x_))
        dims = x
    else
        throw(ArgumentError("the dimensions in expression $expr could not be parsed; " *
            "see the documentation for valid examples"))
    end
    return dims
end


"""
    is_equation(expr)

Return `true` if the given expression `expr` corresponds to an equation `lhs = rhs`
and `false` otherwise. This function just detects the presence of the symbol `=`.

### Input

- `expr` -- expression

### Output

A `Bool` indicating whether `expr` is an equation or not.
"""
function is_equation(expr)
    return @capture(expr, lhs_ = rhs_)
end

# returns `(equation, AT, state)` if `expr` if the given expression `expr`
# corresponds to a dynamic equation, either of the form `x⁺ = rhs` or `x' = rhs`
# in the former case `AT = AbstractDiscreteSystem` and in the latter
# `AT = AbstractContinuousSystem`. `equation` is `x = rhs` where the superscript
# is stripped from the expression and the state corresponds to `x`,
# otherwise, return `(nothing, nothing, nothing)`
# Additionally, the `lhs` can be in the form `E*x'` or `E*x⁺` where `E` can be
# any variable name.
function strip_dynamic_equation(expr)
    expr_str = string(expr)
    stripped_equation = ""
    if occursin("'", expr_str)
        AT = AbstractContinuousSystem
        stripped_equation =  replace(expr_str, "'" => "")
    elseif occursin("⁺", expr_str)
        AT = AbstractDiscreteSystem
        stripped_equation =  replace(expr_str, "⁺" => "")
    end
    stripped_expr = Meta.parse(stripped_equation)
    # check if `stripped_expr` is an equation and extract `lhs` and `rhs` if so
    !@capture(stripped_expr, lhs_ = rhs_) && return (nothing, nothing, nothing)

    # extract the name of the state variable from the lhs
    if @capture(lhs, (E_*x_)) || @capture(lhs, (x_))
        state = x
        return (stripped_expr, AT, state)
    end
    return (nothing, nothing, nothing)
end

function _parse_system(expr::Expr)
    return _parse_system((expr,))
end

function _parse_system(exprs::NTuple{N, Expr}) where {N}
    # define default dynamic equation, unknown abstract system type,
    # and empty list of constraints
    dynamic_equation = nothing
    AT = nothing
    constraints = Vector{Expr}()

    # define defaults for state, noise and input symbol and default dimension
    default_input_var = :u
    default_noise_var = :w
    state_var = nothing
    input_var = nothing
    noise_var = nothing
    dimension = nothing
    initial_state = nothing # for initial-value problems

    # main loop to parse the subexpressions in exprs
    for ex in exprs

        if is_equation(ex)  # parse an equation
            (stripped, abstract_system_type, subject) = strip_dynamic_equation(ex)
            if subject != nothing
                dynamic_equation = stripped
                state_var = subject
                AT = abstract_system_type
                # if the stripped system has the structure x_ = A_*x_ + B_*u_ or
                # one of the other patterns, handle u_ as input variable
                if @capture(stripped, (x_ = A_*x_ + B_*u_) |
                                      (x_ = x_ + B_*u_) |
                                      (x_ = A_*x_ + B_*u_ + c_) |
                                      (x_ = x_ + B_*u_ + c_) |
                                      (x_ = f_(x_, u_)) )
                    if (f == :+) || (f == :-) || (f == :*)
                        # pattern x_ = f_(x_, u_) also catches the cases:
                        # x_ = x_ + u_, x_ = x_ - u_ and x_ = x_*u_
                        # where u_ doesn't necessarily need to be the input
                    else
                        input_var = u
                    end
                end

            elseif @capture(ex, (dim = (f_dims_)) | (dims = (f_dims_)))
                dimension = _capture_dim(f_dims)

            elseif @capture(ex, (input = u_) | (u_ = input))
                input_var = u

            elseif @capture(ex, (noise = w_) | (w_ = noise))
                noise_var = w

            else
                throw(ArgumentError("could not properly parse the equation $ex; " *
                                    "see the documentation for valid examples"))
            end

        elseif @capture(ex, x_(0) ∈ X0_)
            # TODO? handle equality, || @capture(ex, x_(0) = X0_)
            if x != state_var
                throw(ArgumentError("the initial state assignment, $x(0), does "*
                "not correspond to the state variable $state_var"))
            end
            initial_state = X0

        elseif @capture(ex, state_ ∈ Set_)  # parse a constraint
            push!(constraints, ex)

        elseif @capture(ex, (input: u_) | (u_: input))  # parse an input symbol
            input_var = u

        elseif @capture(ex, (noise: w_) | (w_: noise))  # parse a noise symbol
            noise_var = w

        elseif @capture(ex, (dim: (f_dims_)) | (dims: (f_dims_)))  # parse a dimension
            dimension = _capture_dim(f_dims)

        else
            throw(ArgumentError("the expression $ex could not be parsed; " *
                                "see the documentation for valid examples"))
        end
    end

    # error handling for the given equations
    dynamic_equation == nothing && throw(ArgumentError("the dynamic equation was not found"))

    # error handling for the given set constraints
    nsets = length(constraints)
    nsets > 3 && throw(ArgumentError("cannot parse $nsets set constraints"))

    # error handling for variable names
    state_var == nothing && throw(ArgumentError("the state variable was not found"))
    got_input_var = input_var != nothing
    got_noise_var = noise_var != nothing
    if got_input_var && (state_var == input_var)
         throw(ArgumentError("state and input variables have the same name `$(state_var)`"))
    elseif got_noise_var && (state_var == noise_var)
         throw(ArgumentError("state and noise variables have the same name `$(state_var)`"))
    elseif got_input_var && got_noise_var && (input_var == noise_var)
         throw(ArgumentError("input and noise variables have the same name `$(input_var)`"))
    end

    # assign default values
    if !got_input_var
        input_var = default_input_var
    end
    if !got_noise_var
        noise_var = default_noise_var
    end

    return dynamic_equation, AT, constraints,
           state_var, input_var, noise_var, dimension, initial_state
end

"""
    extract_dyn_equation_parameters(equation, state, input, noise, dim, AT)

Extract the value and field parameter from the dynamic equation `equation` according
to the variable `state`, `input` and `noise`.

For the right-hand side of the dynamic equation, this function returns a vector of tuples
containing some elements from the list
- `(:A_user, :A)`
- `(:B_user, :B)`
- `(:c_user, :c)`
- `(:D_user, :D)`
- `(:f_user, :f)`
- `(:statedim_user :statedim)`
- `(:inputdim_user :inputdim)`
- `(:noisedim_user :noisedim)`
and for the left-hand side, it returns either an empty vector `Any[]` or
`[(:E_user, :E)]` where the first argument of the tuple corresponds to the value
and the second argument of the tuple corresponds to the field parameter.

### Input

- `equation` -- dynamic equation
- `state`    -- state variable
- `input`    -- input variable
- `noise`    -- noise variable
- `dim`      -- dimensionality
- `AT`       -- abstract system type

### Output

Two arrays of tuples containing the value and field parameters for the right-hand
and left-hand side of the dynamic equation `equation`.
"""
function extract_dyn_equation_parameters(equation, state, input, noise, dim, AT)
    @capture(equation, lhs_ = rhscode_)
    lhs_params = Vector{Tuple{Any, Symbol}}()
    rhs_params = Vector{Tuple{Any, Symbol}}()
    # if a * is used on the lhs, the rhs is a code-block
    if  @capture(lhs, E_*x_)
        push!(lhs_params, (E, :E))
        rhs = rhscode.args[end]
    else
        rhs = rhscode
    end

    # if rhs is parsed as addition => affine system which is controlled, noisy or both
    if @capture(rhs, A_ + B__)
        # parse summands of rhs and add * if needed
        summands = add_asterisk.([A, B...], Ref(state), Ref(input), Ref(noise))
        push!(rhs_params, extract_sum(summands, state, input, noise)...)
    # if rhs is a function call except `*` or `-` => black-box system
    elseif @capture(rhs, f_(a__)) && f != :(*) && f != :(-)
        # the dimension argument needs to be a iterable
        (dim == nothing) && throw(ArgumentError("for a blackbox system, the dimension has to be defined"))
        dim_vec = [dim...]
        push!(rhs_params, extract_blackbox_parameter(rhs, dim_vec)...)

    # if rhs is a single term => affine systm (e.g. A*x, Ax, 2x, x or 0)
    else
        # if not specified, assume dim = 1
        dim = (dim == nothing) ? 1 : dim
        if rhs == state  # => rhs = x
            if AT == AbstractDiscreteSystem
                push!(rhs_params, (dim, :statedim))
            elseif AT == AbstractContinuousSystem
                push!(rhs_params, (I(dim), :A))
            end
        elseif rhs == :(0) && AT == AbstractContinuousSystem # x' = 0
            push!(rhs_params, (dim, :statedim))
        else
            if @capture(rhs, -var_) # => rhs = -x
                if state == var
                    push!(rhs_params, (-1.0*I(dim), :A))
                end
            else
                rhs = add_asterisk(rhs, state, input, noise)
                if @capture(rhs, array_ * var_)
                    if state == var # rhs = A_x_ or rhs= A_*x_
                        value = tryparse(Float64, string(array))
                        if value == nothing # e.g. => rhs = Ax
                            push!(rhs_params, (array, :A))
                        else # => e.g., rhs = 2x
                            push!(rhs_params, (value*I(dim), :A))
                        end
                    else
                        throw(ArgumentError("if there is only one term on the "*
                                  "right-hand side, the state needs to be contained"))
                    end
                end
            end
        end
    end
    return lhs_params, rhs_params
end

"""
    add_asterisk(summand, state::Symbol, input::Symbol, noise::Symbol)

Checks if expression `summand` contains `state`, `input` or `noise` at its end.
If so, a multiplication expression, e.g. `Expr(:call, :*, :A, :x) is created.
If not, `summand` is returned.

### Input

- `summand` -- expressions
- `state`   -- state variable
- `input`   -- input variable
- `noise`   -- noise variable

### Output

Multiplication expression or symbol.

### Example

```jldoctest
julia> using MathematicalSystems: add_asterisk

julia> add_asterisk(:(A1*x), :x, :u, :w)
:(A1 * x)

julia> add_asterisk(:(c1), :x, :u, :w)
:c1

julia> add_asterisk(:(Ax1), :x1, :u, :w)
:(A * x1)

julia> add_asterisk(:(Awb), :x1, :u, :wb)
:(A * wb)

julia> add_asterisk(:(A1u), :x, :u, :w)
:(A1 * u)

julia> add_asterisk(:(A1ub), :x, :u, :w)
:A1ub
```
"""
function add_asterisk(summand, state::Symbol, input::Symbol, noise::Symbol)
    if @capture(summand, A_ * x_)
        return summand
    end

    str = string(summand)
    statestr = string(state); lenstate = length(statestr)
    inputstr = string(input); leninput = length(inputstr)
    noisestr = string(noise); lennoise = length(noisestr)

    # if summand contains the state, input or noise variable at the end and has
    # one or more additional characters (i.e. length(str) > length(state) for the state)
    # a `*` is added in between
    if lenstate < length(str) && str[(end-lenstate+1):end] == statestr
        return Meta.parse(str[1:end-length(statestr)]*"*$state")

    elseif leninput < length(str) && str[(end-leninput+1):end] == inputstr
        return Meta.parse(str[1:end-length(inputstr)]*"*$input")

    elseif lennoise < length(str) && str[(end-lennoise+1):end] == noisestr
        return Meta.parse(str[1:end-length(noisestr)]*"*$noise")

    else # summand is returned which is either a constant term or the state, input or noise variable
        return summand
    end
end

"""
    extract_sum(summands, state::Symbol, input::Symbol, noise::Symbol)

Extract the variable name and field name for every element of `summands` which
corresponds to the elements of the right-hand side of an affine system.

### Input

- `summands` -- array of expressions
- `state`    -- state variable
- `input`    -- input variable
- `noise`    -- noise variable

### Output

Array of tuples of symbols with variable name and field name.

### Example

```jldoctest
julia> using MathematicalSystems: extract_sum

julia> extract_sum([:(A1*x)], :x, :u, :w)
1-element Array{Tuple{Any,Symbol},1}:
 (:(hcat(A1)), :A)

julia> extract_sum([:(A1*x), :(B1*u), :c], :x, :u, :w)
3-element Array{Tuple{Any,Symbol},1}:
 (:(hcat(A1)), :A)
 (:(hcat(B1)), :B)
 (:(vcat(c)), :c)

julia> extract_sum([:(A1*x7), :( B1*u7), :( B2*w7)], :x7, :u7, :w7)
3-element Array{Tuple{Any,Symbol},1}:
 (:(hcat(A1)), :A)
 (:(hcat(B1)), :B)
 (:(hcat(B2)), :D)
```

### Notes

If an element of `summands` is a multiplication expression `lhs*rhs`,
return `lhs` as variable name and `:A` as field name if `rhs==state`,
`:B` as field name if `rhs==input` and `:D` as field name if `rhs==noise`.

If an element of `summands` is a symbol, and not equal to `input` or `noise`,
the symbol is the variable name and the field name is `:c`. If it is equal to
`input`, the variable name is a `IdentityMultiple(I,state_dim)` where `state_dim`
is extracted from the state matrix (i.e. take the symbol `lhs` of `lhs*rhs` where
`rhs==state` which corresponds to the state matrix and generate the expression
`state_dim = size(lhs,1)` which is evaluated in the scope where `@system` is called)
and the field name is `:B`.

Similiarily, if the element is equal to `noise`, the variable name is
`IdentityMultiple(I, state_dim)` and the field name is `:D`.
"""
function extract_sum(summands, state::Symbol, input::Symbol, noise::Symbol)
    params = Tuple{Any,Symbol}[]
    state_dim = 1
    got_state_dim = false

    num_state_assignments = 0
    num_input_assignments = 0
    num_noise_assignments = 0
    num_const_assignments = 0

    for summand in summands
        if @capture(summand, array_ * var_)
            if state == var
                push!(params, (Expr(:call, :hcat, array), :A))
                # obtain "state_dim" for later using in IdentityMultiple
                state_dim =  Expr(:call, :size, :($array), 1)
                got_state_dim = true
                num_state_assignments += 1

            elseif input == var
                push!(params, (Expr(:call, :hcat, array), :B))
                 num_input_assignments += 1

            elseif noise == var
                push!(params, (Expr(:call, :hcat, array), :D))
                num_noise_assignments += 1

            else
                throw(ArgumentError("in the dynamic equation, the expression "*
                "$summand does not contain the state $state, the input $input "*
                "or the noise term $noise"))
            end
        elseif @capture(summand, array_)
            identity = :(I($state_dim))
            # if array == variable: field value equals identity
            if state == array
                push!(params, (identity, :A))
                num_state_assignments += 1
            elseif input == array
                push!(params, (identity, :B))
                num_input_assignments += 1
            elseif noise == array
                push!(params, (identity, :D))
                num_noise_assignments += 1
            else
                push!(params, (Expr(:call, :vcat, array), :c))
                num_const_assignments += 1
            end
        end
    end
    num_const_assignments > 1 && throw(ArgumentError("there is more than one constant term"))
    num_state_assignments > 1 && throw(ArgumentError("there is more than one state term `$state`"))
    num_input_assignments > 1 && throw(ArgumentError("there is more than one input term `$input`"))
    num_noise_assignments > 1 && throw(ArgumentError("there is more than one noise term `$noise`"))

    return params
end

# Extract the variable name of the function of the rhs of an equation if it has
# the form `f_(x_)`, `f_(x_,u_)` or `f_(x_,u_,w_)`.
# In addition to the variable name of the function and the field name `:f`,
# depending on the number of input arguments, the field names `:statedim`, `:inputdim`
# and `:noisedim` with corresponding variable names (which are the number of state,
# input and noise dimensions provided as input to the macro) is returned.
function extract_blackbox_parameter(rhs, dim::AbstractVector)
    if @capture(rhs, f_(x_))
        @assert length(dim) == 1
        return [(f, :f), (dim[1], :statedim)]
    elseif @capture(rhs, f_(x_,u_))
        @assert length(dim) == 2
        return [(f, :f), (dim[1], :statedim),
                         (dim[2], :inputdim)]
    elseif @capture(rhs, f_(x_,u_,w_))
        @assert length(dim) == 3
        return [(f, :f), (dim[1], :statedim),
                         (dim[2], :inputdim),
                         (dim[3], :noisedim)]
    end
end

# Take the vectors of tuples providing the variable and fields names for the
# lhs, the rhs and the sets, combine them, and return a vector of field names
# and a vector of variable names.
function constructor_input(lhs, rhs, set)
    rhs_fields = [tuple[2] for tuple in rhs]
    lhs_fields = [tuple[2] for tuple in lhs]
    set_fields = [tuple[2] for tuple in set]
    (length(unique(set_fields)) != length(set_fields)) &&
        throw(ArgumentError("There is some ambiguity in the set definition"))
    rhs_var_names = [tuple[1] for tuple in rhs]
    lhs_var_names = [tuple[1] for tuple in lhs]
    set_var_names = [tuple[1] for tuple in set]
    field_names = (rhs_fields..., lhs_fields..., set_fields...)
    var_names = (rhs_var_names..., lhs_var_names..., set_var_names...)
    return field_names, var_names
end

# extract the variable name and the field name for a set expression. The method
# checks wether the set belongs to the `state`, `input` or `noise` and returns a
# tuple of symbols where the field name is either `:X`, `:U` or `:W` and
# the variable name is the value parsed as Set_
function extract_set_parameter(expr, state, input, noise) # input => to check set definitions
    if @capture(expr, x_(0) ∈ Set_)
        return Set, :x0

    elseif @capture(expr, x_ ∈ Set_)
        if x == state
            return  Set, :X
        elseif x == input
            return  Set, :U
        elseif x == noise
            return  Set, :W
        else
            error("$expr is not a valid set constraint definition; it does not contain"*
                   "the state $state, the input $input or noise term $noise")
        end
    end
    throw(ArgumentError("the set entry $(expr) does not have the correct form `x_ ∈ X_`"))
end

"""
    sort(parameters::Vector, order::Tuple)

Filter and sort the vector `parameters` according to `order`.

### Input

- `parameters` -- vector of tuples
- `order`      -- tuple of symbols

### Output

A new vector of tuples corresponding to `parameters` filtered and sorted according
to `order`.

### Examples

```jldoctest
julia> parameters= [(:U1, :U), (:X1, :X), (:W1, :W)];

julia> sort(parameters, (:X, :U, :W))
3-element Array{Tuple{Any,Symbol},1}:
 (:X1, :X)
 (:U1, :U)
 (:W1, :W)

julia>  parameters = [(:const, :c), (:A, :A)];

julia> sort(parameters, (:A, :B, :c, :D))
2-element Array{Tuple{Any,Symbol},1}:
 (:A, :A)
 (:const, :c)
```

### Notes

`parameters` is a vector that contains tuples whose second element
is considered for the sorting according to `order`.

If a value of `order` is not contained in `parameters`, the corresponding entry of
`order` will be omitted.
"""
function sort(parameters::Vector{<:Tuple{Any, Symbol}}, order::NTuple{N, Symbol}) where {N}
    order_parameters = Vector{Tuple{Any, Symbol}}()
    for ordered_element in order
        for tuple in parameters
            if tuple[2] == ordered_element
                push!(order_parameters, tuple)
            end
        end
    end
    return order_parameters
end

function _get_system_type(dyn_eq, AT, constr, state, input, noise, dim)
    lhs, rhs = extract_dyn_equation_parameters(dyn_eq, state, input, noise, dim, AT)
    ordered_rhs = sort(rhs, (:A, :B, :c, :D, :f, :statedim, :inputdim, :noisedim))
    ordered_set = sort(extract_set_parameter.(constr, state, input, noise), (:X, :U, :W))
    field_names, var_names = constructor_input(lhs, ordered_rhs, ordered_set)
    sys_type = _corresponding_type(AT, field_names)
    return sys_type, var_names
end

"""
    system(expr...)

Return an instance of the system type corresponding to the given expressions.

### Input

- `expr` -- expressions separated by commas which define the dynamic equation,
            the constraint sets or the dimensionality of the system

### Output

A system that best matches the given expressions.

### Notes

**Terms.** The expression `expr` contains one or more of the following sub-expressions:

- dynamic equation, either continuous, e.g.`x' = Ax`, or discrete, e.g. `x⁺ = Ax`
- set constraints, e.g. `x ∈ X`
- input constraints, e.g. `u ∈ U`
- dimensionality, e.g. `dim: (2,1)` or `dim = 1`
- specification of the input variable, e.g. `input: u` or `input = u`
- specification of the noise variable, e,g, `noise: w` or `noise = w`

The macro call is then formed by separating the previous sub-expressions
(which we simply call *terms* hereafter), as in:

```julia
@system(dynamic eq., set constr., input constr., input specif., noise spec., dimens.)
```

The different terms that compose the system's definition do not have to appear in
any particular order. Moreover, the only mandatory term is the dynamic equation;
the other terms are optional and default values may apply depending on the system
type; this is explained next.

**Dynamic equation.** The time derivative in a continuous system is specified
by using `'`, as in `x' = A*x`. A discrete system is specified using `⁺` (which
can be written with the combination of keys `\\^+[TAB]`), as in `x⁺ = A*x`.
Moreover, the asterisk denoting matrix-vector products is optional. For instance,
both `x' = Ax` and `x' = A*x` are parsed as the linear continuous system whose
state matrix is `A`. The matrix is supposed to be defined at the call site.

**Default values.** When the dynamic equation is parsed, the variable on the
left-hand side is interpreted as the state variable. The input variable is by
default `u` and the noise variable is by default `w`. If we want to change the
default name of the input variable, this can be done by adding the term
`input: var` (or equivalently, `input=var)` where `var` corresponds to the new
name of the input variable, eg. `@system(x' = A*x + B*v, input:v)`.
Similarly, a noise variable is specified with `noise: var` or `noise=var`.

**Exceptions.** The following exceptions and particular cases apply:

- If the right-hand side has the form `A*x + B*foo`, `A*x + B*foo + c` or
  `f(x, foo)`, the equation is parsed as a controlled linear (affine) or controlled
  black-box system with input `foo`. Note that in this case, `input` variable does
  not correspond to the default value of `u`, but `foo` is parsed as being the input.

- If the left-hand side contains a multiplicative term in the form `E*x⁺` or `E*x'`,
  the equation is parsed as an algebraic system. In this case, the asterisk
  `*` operator is mandatory.

- Systems of the form `x' = α*x` where `α` is a scalar are parsed as linear
  systems. The default dimension is `1` and `α` is parsed as `Float64`;
  if the system is higher-dimensional, use `dim`, as in `x' = 2x, dim=3`.

### Examples

Let us first create a continuous linear system using this macro:

```jldoctest system_macro
julia> A = [1. 0; 0 1.];

julia> @system(x' = A*x)
LinearContinuousSystem{Float64,Array{Float64,2}}([1.0 0.0; 0.0 1.0])
```

A discrete system is defined by using `⁺`:

```jldoctest system_macro
julia> @system(x⁺ = A*x)
LinearDiscreteSystem{Float64,Array{Float64,2}}([1.0 0.0; 0.0 1.0])
```

Additionally, a set definition `x ∈ X` can be added to create a constrained system.
For example, a discrete controlled affine system with constrained states and
inputs writes as:

```jldoctest system_macro
julia> using LazySets

julia> B = Matrix([1. 0.5]');

julia> c = [1., 1.5];

julia> X = BallInf(zeros(2), 10.);

julia> U = BallInf(zeros(1), 2.);

julia> @system(x' = A*x + B*u + c, x ∈ X, u ∈ U)
ConstrainedAffineControlContinuousSystem{Float64,Array{Float64,2},Array{Float64,2},Array{Float64,1},BallInf{Float64,Array{Float64,1}},BallInf{Float64,Array{Float64,1}}}([1.0 0.0; 0.0 1.0], [1.0; 0.5], [1.0, 1.5], BallInf{Float64,Array{Float64,1}}([0.0, 0.0], 10.0), BallInf{Float64,Array{Float64,1}}([0.0], 2.0))
```
For the creation of a black-box system, the state, input and noise dimensions have
to be defined separately. For a constrained controlled black-box system, the macro
writes as

```jldoctest system_macro
julia> f(x, u) = x + u;

julia> @system(x⁺ = f(x, u), x ∈ X, u ∈ U, dim: (2,2))
ConstrainedBlackBoxControlDiscreteSystem{typeof(f),BallInf{Float64,Array{Float64,1}},BallInf{Float64,Array{Float64,1}}}(f, 2, 2, BallInf{Float64,Array{Float64,1}}([0.0, 0.0], 10.0), BallInf{Float64,Array{Float64,1}}([0.0], 2.0))
```
"""
macro system(expr...)
    try
        dyn_eq, AT, constr, state, input, noise, dim, x0 = _parse_system(expr)
        if @capture(expr[1], (M_*x_'' + C_*x_' + K_*x_ = 0))
            variableNames = :(M, C, K)
            sys = Expr(:call, SecondOrderLinearContinuousSystem, :($variableNames...))
        else
            sys_type, var_names = _get_system_type(dyn_eq, AT, constr, state, input, noise, dim)
            sys = Expr(:call, :($sys_type), :($(var_names...)))
        end
        if x0 == nothing
            return esc(sys)
        else
            ivp = Expr(:call, InitialValueProblem, :($sys), :($x0))
            return esc(ivp)
        end
    catch ex
        if  isa(ex, ArgumentError)
            return :(throw($ex))
        else
            throw(ex)
        end
    end
end



"""
    ivp(expr...)

Return an instance of the initial-value problem type corresponding to the given
expressions.

### Input

- `expr` -- expressions separated by commas which define the dynamic equation,
            the constraint sets or the dimensionality of the system, and the set
            of initial states (required)

### Output

An initial-value problem that best matches the given expressions.

### Notes

This macro behaves like the `@system` macro, the sole difference being that in
`@ivp` the constraint on the set of initial states is mandatory. For the technical
details we refer to the documentation of [`@system`](@ref).

The macro can also be called with a `system` argument of type `AbstractSystem`
in the form `@ivp(system, state(0) ∈ initial_set)`.

### Examples

```jldoctest ivp_macro
julia> p = @ivp(x' = -x, x(0) ∈ [1.0]);

julia> typeof(p)
InitialValueProblem{LinearContinuousSystem{Float64,IdentityMultiple{Float64}},Array{Float64,1}}

julia> initial_state(p)
1-element Array{Float64,1}:
 1.0

julia> sys = @system(x' = [1 0; 0 1] * x);

julia> @ivp(sys, x(0) ∈ [-1, 1])
InitialValueProblem{LinearContinuousSystem{Int64,Array{Int64,2}},Array{Int64,1}}(LinearContinuousSystem{Int64,Array{Int64,2}}([1 0; 0 1]), [-1, 1])
```
"""
macro ivp(expr...)
    try
        # check if the @ivp macro is called with an AbstractSystem argument
        if parses_as_system(expr[1])
            sys = expr[1]
            @capture(expr[2], x_(0) ∈ x0_) || throw(ArgumentError("malformed epxpression")) # TODO handle equality
            ivp = Expr(:call, InitialValueProblem, :($(expr[1])), :($x0))
            return esc(ivp)
        else
            dyn_eq, AT, constr, state, input, noise, dim, x0 = _parse_system(expr)
            sys_type, var_names = _get_system_type(dyn_eq, AT, constr, state, input, noise, dim)
            sys = Expr(:call, :($sys_type), :($(var_names...)))
            if x0 == nothing
                return throw(ArgumentError("an initial-value problem should define the " *
                            "initial states, but such expression was not found"))
            else
                ivp = Expr(:call, InitialValueProblem, :($sys), :($x0))
                return esc(ivp)
            end
        end
    catch ex
        if  isa(ex, ArgumentError)
            return :(throw($ex))
        else
            throw(ex)
        end
    end
end

function parses_as_system(expr)
    # The argument is a system if
    # 1) it is directly defined using @system in the macro call
    if @capture(expr, @system(def_))
        return true
    end
    # 2) it is a variable, i.e. it is not a dynamic equation
    if !@capture(expr, a_ = b_)
        return true
    end
    return false
end
