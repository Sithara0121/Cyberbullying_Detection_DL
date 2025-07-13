using Flux
using Distributions

# RBM Definition
struct RBM
    W::Matrix{Float32}
    vbias::Vector{Float32}
    hbias::Vector{Float32}
end

# Initialize RBM
function init_rbm(n_visible::Int, n_hidden::Int)
    W = 0.01f0 * randn(Float32, n_hidden, n_visible)
    vbias = zeros(Float32, n_visible)
    hbias = zeros(Float32, n_hidden)
    return RBM(W, vbias, hbias)
end

# Sigmoid with elementwise support
extended_sigmoid(x) = 1.0 ./ (1.0 .+ exp.(-x))

# Sample from Bernoulli distribution
function sample_bernoulli(p::AbstractMatrix{Float32})
    map(x -> rand(Bernoulli(x)), p)
end

# Contrastive Divergence (CD-1)
function cd1!(rbm::RBM, data::Matrix{Float32}, lr::Float32)
    batch_size = size(data, 2)

    # Positive phase
    h_prob = extended_sigmoid(rbm.W * data .+ rbm.hbias)
    h_sample = h_prob .> rand(Float32, size(h_prob))

    # Negative phase
    v_recon_prob = extended_sigmoid(rbm.W' * h_sample .+ rbm.vbias)
    h_recon_prob = extended_sigmoid(rbm.W * v_recon_prob .+ rbm.hbias)

    # Update
    rbm.W .+= lr .* ((h_prob * data') .- (h_recon_prob * v_recon_prob')) ./ batch_size
    rbm.vbias .+= lr .* vec(sum(data .- v_recon_prob, dims=2)) ./ batch_size
    rbm.hbias .+= lr .* vec(sum(h_prob .- h_recon_prob, dims=2)) ./ batch_size
end

# Train RBM
function train_rbm!(data::Matrix{Float32}, n_hidden::Int; epochs::Int=10, lr::Float32=0.01f0)
    n_visible = size(data, 2)
    rbm = init_rbm(n_visible, n_hidden)

    for epoch in 1:epochs
        cd1!(rbm, data, lr)
        println("RBM Epoch $epoch done")
    end
    return rbm
end



# Use trained RBM to transform data
function transform(rbm::RBM, data::Matrix{Float32})
    extended_sigmoid(rbm.W * data .+ rbm.hbias)
end
