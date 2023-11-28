using ITensors

# Is it an MPS…?
# It's only half of an MPO, so I guess?
mutable struct MPDO <: ITensors.AbstractMPS
  data::Vector{ITensor}
  llim::Int
  rlim::Int
end

function MPDO(ψ::MPS)
  tensors = ITensor[]
  for (i, ψᵢ) in enumerate(ψ)
    push!(tensors, ITensor(ITensors.array(ψᵢ), (inds(ψᵢ)..., Index(1, "kraus,n=$i"))))
  end
  return MPDO(tensors, ψ.llim, ψ.rlim)
end

function MPDO(A::Vector{<:ITensor}; ortho_lims::UnitRange=1:length(A))
  return MPDO(A, first(ortho_lims) - 1, last(ortho_lims) + 1)
end

function setindex!(
  ψ::MPDO, A::ITensor, r::UnitRange{Int}; orthocenter::Integer=last(r), kwargs...
)
  @assert :perm ∉ keys(kwargs) "Permutation not implemented atm."

  # Replace the sites of ITensor ψ
  # with the tensor A, splitting up A
  # into MPS tensors
  firstsite = first(r)
  lastsite = last(r)
  @assert firstsite ≤ ITensors.orthocenter(ψ) ≤ lastsite
  @assert firstsite ≤ ITensors.leftlim(ψ) + 1
  @assert ITensors.rightlim(ψ) - 1 ≤ lastsite

  # TODO: allow orthocenter outside of this
  # range, and orthogonalize/truncate as needed
  @assert firstsite ≤ orthocenter ≤ lastsite

  lind = linkind(ψ, firstsite - 1)

  #sites = [siteinds(ψ, j) for j ∈ firstsite:lastsite]
  sites = [filter(inds(A), "n=$j") for j in firstsite:lastsite]

  ψA = MPDO(A, sites; leftinds=lind, orthocenter=orthocenter - first(r) + 1, kwargs...)

  ψ[firstsite:lastsite] = ψA

  return ψ
end

function runcircuit_MPDO(
  hilbert::Vector{<:Index},
  circuit::Vector,
  noise;
  eltype=nothing,
  device=identity,
  kwargs...,
)

  # this step is required to check whether there is already noise in the circuit
  # which was added using the `insertnoise` function. If so, one should call directly
  # the `choimatrix` function.
  circuit_tensors = buildcircuit(hilbert, circuit; noise, device, eltype)
  if circuit_tensors isa Vector{<:ITensor}
    inds_sizes = [length(inds(g)) for g in circuit_tensors]
  else
    inds_sizes = vcat([[length(inds(g)) for g in layer] for layer in circuit_tensors]...)
  end

  M₀ = productstate(hilbert; eltype, device)
  return runcircuit(MPDO(M₀), circuit_tensors; noise=noise, kwargs...)
end

function runcircuit(
  M::MPDO,
  circuit_tensors::Vector{<:ITensor};
  cutoff=1e-15,
  maxdim=10_000,
  max_kraus_dim=50,
  svd_alg="divide_and_conquer",
  move_sites_back::Bool=true,
  eltype=nothing,
  device=identity,
  kwargs...,
)
  M = device(_convert_leaf_eltype(eltype, M))
  circuit_tensors = device(_convert_leaf_eltype(eltype, circuit_tensors))

  # Check if gate_tensors contains Kraus operators
  inds_sizes = [length(inds(g)) for g in circuit_tensors]
  noiseflag = any(x -> x % 2 == 1, inds_sizes)

  # Noisy evolution: MPS/MPO -> MPO
  if noiseflag
    # If M is an MPS, |ψ⟩ -> ρ = |ψ⟩⟨ψ| (MPS -> MPO)
    # ρ -> ε(ρ) (MPO -> MPO, conjugate evolution)
    return apply(
      circuit_tensors, M; cutoff, maxdim, max_kraus_dim, svd_alg, move_sites_back
    )
  else
    # Pure state evolution
    error("Use an MPS for noiseless evolution.")
  end
end

function ITensors.product(
  As::Vector{ITensor},
  ψ::MPDO;
  move_sites_back_between_gates::Bool=true,
  move_sites_back::Bool=true,
  kwargs...,
)
  Aψ = ψ
  for A in As
    Aψ = product(A, Aψ; move_sites_back=move_sites_back_between_gates, kwargs...)
  end
  if !move_sites_back_between_gates && move_sites_back
    s = siteinds(Aψ)
    ns = 1:length(ψ)
    ñs = [findsite(ψ, i) for i in s]
    Aψ = movesites(Aψ, ns .=> ñs; kwargs...)
  end
  return Aψ
end

function ITensors.product(
  o::ITensor, ψ::MPDO, ns=findsites(ψ, o); move_sites_back::Bool=true, kwargs...
)
  N = length(ns)
  ns = sort(ns)

  # TODO: make this smarter by minimizing
  # distance to orthogonalization.
  # For example, if ITensors.orthocenter(ψ) > ns[end],
  # set to ns[end].
  ψ = orthogonalize(ψ, ns[1])
  diff_ns = diff(ns)
  ns′ = ns
  if any(!=(1), diff_ns)
    ns′ = [ns[1] + n - 1 for n in 1:N]
    ψ = movesites(ψ, ns .=> ns′; kwargs...)
  end
  ϕ = ψ[ns′[1]]
  for n in 2:N
    ϕ *= ψ[ns′[n]]
  end
  ϕ = product(o, ϕ)

  kraus_kwargs = copy(kwargs)
  kraus_kwargs[:maxdim] = pop!(kraus_kwargs, :max_kraus_dim)

  for idx in ns
    k_idxs = filter(krausinds(ϕ), "n=$idx")
    if length(k_idxs) > 1
      # combine
      cmb = combiner(k_idxs)
      ϕ *= cmb

      # truncate
      _, ϕ = factorize(
        ϕ, combinedind(cmb); kraus_kwargs..., tags="kraus,n=$idx", ortho="left"
      )
    end
  end

  kwargs = delete!(copy(kwargs), :max_kraus_dim)
  ψ[ns′[1]:ns′[end], kwargs...] = ϕ
  if move_sites_back
    # Move the sites back to their original positions
    ψ = movesites(ψ, ns′ .=> ns; kwargs...)
  end
  return ψ
end

krausinds(A::ITensor) = filter(inds(A), "kraus")

krausind(A::ITensor) = getfirst(inds(A), "kraus")
