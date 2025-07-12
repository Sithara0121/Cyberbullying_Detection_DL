using CSV
using DataFrames
using TextAnalysis
using MLDataUtils
using Flux
using Statistics
using Random
using StatsBase
using EvalMetrics
using Printf
using LinearAlgebra
using Dates

# Step 1: Load dataset
println("Step 1: Loading dataset...")
df = CSV.read("tweets.csv", DataFrame; ignoreemptylines=true)
println("Dataset loaded with $(nrow(df)) rows.")

# Step 2: Preprocess tweet text
println("Step 2: Preprocessing tweet text...")
function preprocess_text(text)
    try
        text = lowercase(String(text))
        text = replace(text, r"http\S+" => "")
        text = replace(text, r"[^a-z\s]" => "")
        text = replace(text, r"\s+" => " ")
        return strip(text)
    catch e
        return ""
    end
end

df.tweet_text = preprocess_text.(df.tweet_text)
df = dropmissing(df, :tweet_text)

# Convert multi-class to binary classification
# 0 = not cyberbullying, 1 = cyberbullying
df.label = ifelse.(df.cyberbullying_type .== "not_cyberbullying", 0, 1)
println("Label distribution after binary conversion:")
println(combine(groupby(df, :label), nrow => :Count))

# Filter out rows with missing or empty text after preprocessing
df = filter(row -> !ismissing(row.tweet_text) && !isempty(row.tweet_text), df)
println("Text preprocessing complete. Remaining samples: $(nrow(df))")

# Step 3: Vectorize text using TF-IDF
println("Step 3: Vectorizing text using TF-IDF...")


function tfidf_vectorize(texts)
    docs = [StringDocument(t) for t in texts]
    for doc in docs
        prepare!(doc, StripPunctuation())
        prepare!(doc, StripStopWords())
    end
    corpus = Corpus(docs)
    dtm = DocumentTermMatrix(corpus)
    tfidf_model = TFIDF(dtm)
    return Array(tfidf_model)
end



X = tfidf_vectorize(df.tweet_text)
println("Vectorization complete. Feature matrix size: $(size(X))")

# Step 4: Prepare labels
y = convert(Array{Float32}, df.label)
println("Label encoding complete. Number of classes: $(length(unique(y)))")

# Step 5: Train-test split
function train_test_split(X, y, test_ratio=0.2)
    n = size(X, 1)
    idx = collect(1:n)
    Random.seed!(42)
    shuffle!(idx)
    test_size = round(Int, n * test_ratio)
    test_idx = idx[1:test_size]
    train_idx = idx[(test_size+1):end]
    return X[train_idx, :], y[train_idx], X[test_idx, :], y[test_idx]
end

X_train, y_train, X_test, y_test = train_test_split(X, y)

# RBM implementation
struct RBM
    W::Matrix{Float32}
    hbias::Vector{Float32}
    vbias::Vector{Float32}
end

function init_rbm(n_visible, n_hidden)
    W = 0.01f0 * randn(Float32, n_hidden, n_visible)
    hbias = zeros(Float32, n_hidden)
    vbias = zeros(Float32, n_visible)
    return RBM(W, hbias, vbias)
end

function sigmoid_fn(x)
    return 1.0 ./ (1.0 .+ exp.(-x))
end

function sample_bernoulli(p)
    return Float32.(p .> rand(size(p)))
end

function contrastive_divergence(rbm::RBM, v_input, lr=0.01f0, k=1)
    v0 = v_input
    h0_prob = sigmoid_fn(rbm.W * v0 .+ rbm.hbias)
    h0_sample = sample_bernoulli(h0_prob)

    vk = v0
    hk = h0_sample
    for _ in 1:k
        vk_prob = sigmoid_fn(rbm.W' * hk .+ rbm.vbias)
        vk = sample_bernoulli(vk_prob)
        hk_prob = sigmoid_fn(rbm.W * vk .+ rbm.hbias)
        hk = sample_bernoulli(hk_prob)
    end

    rbm.W += lr .* ((h0_prob * v0') .- (hk_prob * vk'))
    rbm.vbias += lr .* (v0 .- vk)
    rbm.hbias += lr .* (h0_prob .- hk_prob)
end

function train_rbm(rbm::RBM, data, epochs=5, lr=0.01f0)
    for epoch in 1:epochs
        for i in 1:size(data, 1)
            v = data[i, :] |> vec
            contrastive_divergence(rbm, v, lr)
        end
    end
end

function pretrain_dbn(X, layer_sizes)
    input_data = X
    rbms = RBM[]
    for i in 1:length(layer_sizes)-1
        println("Pretraining RBM layer $i...")
        rbm = init_rbm(size(input_data, 2), layer_sizes[i+1])
        train_rbm(rbm, input_data)
        h_probs = sigmoid_fn.(rbm.W * input_data' .+ rbm.hbias)'
        push!(rbms, rbm)
        input_data = h_probs
    end
    return rbms
end

# Step 6: Define fitness function
function evaluate_fitness(hyperparams)
    try
        h1, h2, lr = hyperparams
        layer_sizes = [size(X_train, 2), round(Int, h1), round(Int, h2)]
        rbms = pretrain_dbn(X_train, layer_sizes)
        model = Chain(
            Dense(layer_sizes[1], layer_sizes[2], relu),
            Dense(layer_sizes[2], layer_sizes[3], relu),
            Dense(layer_sizes[3], 1),
            sigmoid
        )
        loss(x, y) = Flux.binarycrossentropy(model(x), y)
        opt = Flux.Adam(lr)
        data = [(X_train[i, :], y_train[i]) for i in 1:length(y_train)]
        Flux.train!(loss, Flux.params(model), data, opt, cb = () -> nothing)
        y_pred = model(X_test') |> vec
        y_pred_class = round.(y_pred)
        acc = accuracy(y_pred_class, y_test)
        return acc
    catch e
        println("Fitness evaluation failed: $e")
        return 0.0
    end
end

# Step 7: Salp Swarm Algorithm (SSA)
println("Step 6: Starting Salp Swarm Algorithm (SSA)...")
function salp_swarm_algorithm(iterations, n_salps)
    dim = 3
    lb = [10.0, 10.0, 0.0001]
    ub = [100.0, 100.0, 0.1]
    salps = [lb .+ rand(dim) .* (ub .- lb) for _ in 1:n_salps]
    fitness = [evaluate_fitness(s) for s in salps]
    best_idx = argmax(fitness)
    best_pos = copy(salps[best_idx])
    best_fit = fitness[best_idx]

    for iter in 1:iterations
        c1 = 2 * exp(-((4 * iter / iterations)^2))
        for i in 1:n_salps
            for d in 1:dim
                if i == 1
                    c2 = rand()
                    c3 = rand()
                    if c3 < 0.5
                        salps[i][d] = best_pos[d] + c1 * ((ub[d] - lb[d]) * c2 + lb[d])
                    else
                        salps[i][d] = best_pos[d] - c1 * ((ub[d] - lb[d]) * c2 + lb[d])
                    end
                else
                    salps[i][d] = (salps[i][d] + salps[i-1][d]) / 2
                end
                salps[i][d] = clamp(salps[i][d], lb[d], ub[d])
            end
        end
        fitness = [evaluate_fitness(s) for s in salps]
        best_idx = argmax(fitness)
        if fitness[best_idx] > best_fit
            best_fit = fitness[best_idx]
            best_pos = copy(salps[best_idx])
        end
        println("Iteration $iter - Best fitness: $best_fit")
    end
    return best_pos
end

best_hyperparams = salp_swarm_algorithm(5, 15)
println("Best hyperparameters found: $best_hyperparams")

# Step 8: Train final model
println("Step 7: Training final DBN with best hyperparameters...")
h1, h2, lr = best_hyperparams
layer_sizes = [size(X_train, 2), round(Int, h1), round(Int, h2)]
rbms = pretrain_dbn(X_train, layer_sizes)
model = Chain(
    Dense(layer_sizes[1], layer_sizes[2], relu),
    Dense(layer_sizes[2], layer_sizes[3], relu),
    Dense(layer_sizes[3], 1),
    sigmoid
)
loss(x, y) = Flux.binarycrossentropy(model(x), y)
opt = Flux.Adam(lr)
data = [(X_train[i, :], y_train[i]) for i in 1:length(y_train)]
for epoch in 1:10
    Flux.train!(loss, Flux.params(model), data, opt)
end

# Step 9: Evaluate model
y_pred = model(X_test') |> vec
y_pred_class = round.(y_pred)
acc = accuracy(y_pred_class, y_test)
println("Test Accuracy: $(acc)")
