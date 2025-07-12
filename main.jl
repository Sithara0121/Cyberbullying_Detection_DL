# main.jl
using CSV
using DataFrames
using Flux
using Random
using Statistics
using Printf
using LinearAlgebra
using Dates

# Include all necessary scripts
include("src/data.jl")
include("src/utils.jl")
include("src/optimizer.jl")
include("src/model.jl")
include("src/plot_utils.jl")

# Step 1: Load dataset
println("Step 1: Loading dataset...")
df = load_dataset("tweets.csv")
println("Dataset loaded with $(nrow(df)) rows.")

# Step 2: Preprocess tweet text
println("Step 2: Preprocessing tweet text...")
df = preprocess_text(df)

# Step 3: Vectorize text
println("Step 3: Vectorizing text using hash-based vectorizer...")
X, y = vectorize_text(df)

# Step 4: Train-test split
X_train, y_train, X_test, y_test = train_test_split(X, y)

# Step 5: Salp Swarm Algorithm (SSA)
println("Step 5: Starting Salp Swarm Algorithm (SSA)...")
best_hyperparams = salp_swarm_algorithm(X_train, y_train, X_test, y_test, 5, 15)
println("Best hyperparameters found: $best_hyperparams")

# Step 6: Train final model
println("Step 6: Training final model with best hyperparameters...")
model = build_model(best_hyperparams, size(X_train, 2))
train_model!(model, X_train, y_train, best_hyperparams[3])

# Step 7: Evaluate model
println("Step 7: Evaluating model...")
model_accuracy = evaluate_model(model, X_test, y_test)
println("Test Accuracy: $model_accuracy")

# Step 8: Plotting the results (Optional, add this functionality in `plot_utils.jl`)
plot_accuracy(model_accuracy)
