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
df = CSV.read("tweets.csv", DataFrame; ignoreemptyrows=true)
#df = first(df, 2000)  # Limit to the first 2000 rows
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

# Step 3: Vectorize text
println("Step 3: Vectorizing text using hash-based vectorizer...")
function vectorize_text(texts, dim=3000)
    mat = zeros(Float32, length(texts), dim)
    for (i, text) in enumerate(texts)
        words = split(text)
        for word in words
            h = hash(word) % dim + 1
            mat[i, h] += 1
        end
    end
    return mat
end

X = vectorize_text(df.tweet_text)
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

# Step 6: Define fitness function
function evaluate_fitness(hyperparams)
    try
        h1, h2, lr = hyperparams
        model = Chain(
            Dense(size(X_train, 2), round(Int, h1), relu),
            Dense(round(Int, h1), round(Int, h2), relu),
            Dense(round(Int, h2), 1),
            sigmoid
        )
        loss(x, y) = Flux.binarycrossentropy(model(x), y)
        opt = Flux.Optimise.Descent(lr)
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
model = Chain(
    Dense(size(X_train, 2), round(Int, h1), relu),
    Dense(round(Int, h1), round(Int, h2), relu),
    Dense(round(Int, h2), 1),
    sigmoid
)
loss(x, y) = Flux.binarycrossentropy(model(x), y)
opt = Flux.Optimise.Descent(lr)
data = [(X_train[i, :], y_train[i]) for i in 1:length(y_train)]
Flux.train!(loss, Flux.params(model), data, opt)

# Step 9: Evaluate model
y_pred = model(X_test') |> vec
y_pred_class = round.(y_pred)
acc = accuracy(y_pred_class, y_test)
println("Test Accuracy: $(acc)")