using CSV
using DataFrames
using Flux
using Random
using Statistics
using Printf
using LinearAlgebra
using Dates

# Include scripts
include("src/data.jl")
include("src/utils.jl")
include("src/optimizer.jl")
include("src/model.jl")
include("src/rbm.jl")
include("src/plot_utils.jl")

# Step 1: Load dataset
println("Step 1: Loading dataset...")
df = load_dataset("data/tweets.csv")
println("Dataset loaded with $(nrow(df)) rows.")

# Step 2: Preprocess tweet text
println("Step 2: Preprocessing tweet text...")
df = preprocess_text(df)

# Step 3: Vectorizing text
println("Step 3: Vectorizing text using hash-based vectorizer...")
X, y = vectorize_text(df)
X = Float32.(X)
X_matrix = Matrix(X)

# Step 4: Train-test split
X_train, y_train, X_test, y_test = train_test_split(X_matrix, y)

# Step 5: Salp Swarm Algorithm (SSA)
println("Step 5: Starting Salp Swarm Algorithm (SSA)...")
best_hyperparams = salp_swarm_algorithm(X_train, y_train, X_test, y_test, 5, 15)
println("Best hyperparameters found: $best_hyperparams")

# Extract hidden layer sizes
layer_dims = [round(Int, best_hyperparams[1]), round(Int, best_hyperparams[2])]

# Step 6: Pretraining DBN using RBMs
println("Step 6: Pretraining DBN using RBMs...")
X_train_T = Matrix(X_train')  # (features, samples)
rbms = pretrain_dbn(X_train_T, layer_dims; epochs=5, batch_size=64)

# Step 7: Fine-tune DBN
println("Step 7: Fine-tuning DBN with labels...")
model = build_dbn_with_softmax_from_rbms(rbms, 2)

# Define loss function
loss(model, x, y) = Flux.crossentropy(model(x), y)
opt = Flux.setup(Descent(0.01), model)

# One-hot encode y
y_train_onehot = Flux.onehotbatch(y_train, [false, true])

# Prepare training data: use (features, samples)
data = [(X_train_T[:, i], y_train_onehot[:, i]) for i in 1:size(X_train_T, 2)]

# Debug info
@info "Model expects input of size: $(size(rbms[1].W, 2))"
@info "Each x sample size: $(size(data[1][1]))"

# Train
Flux.train!(loss, model, data, opt)

# Step 8: Evaluate
println("Step 8: Evaluating on test set...")
X_test_T = Matrix(X_test')  # (features, samples)
y_pred = model(X_test_T)
pred_classes = Flux.onecold(y_pred, [false, true])
acc = sum(pred_classes .== y_test) / length(y_test)
println("Test Accuracy: $acc")
