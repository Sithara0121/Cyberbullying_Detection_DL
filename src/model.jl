using Statistics
using Flux
using Plots
include("rbm.jl")


function pretrain_dbn(X::Matrix{Float32}, layer_dims::Vector{Int}; epochs::Int=5, batch_size::Int=64)
    input = X
    rbms = Vector{RBM}()

    for n_hidden in layer_dims
        println("Pretraining RBM layer with $n_hidden hidden units")

        n_features = size(input, 1)
        n_samples = size(input, 2)
        lr = 0.1f0
        rbm = init_rbm(n_features, n_hidden)

        for epoch in 1:epochs
            for i in 1:batch_size:n_samples
                batch_end = min(i + batch_size - 1, n_samples)
                x_batch = input[:, i:batch_end]  # select columns, not rows
                cd1!(rbm, Matrix(x_batch), lr)   # no need to transpose
            end
            println("RBM Epoch $epoch done")
        end

        push!(rbms, rbm)
        input = sigmoid.(rbm.W * input .+ rbm.hbias)  # forward to next layer
    end

    return rbms
end



using Flux

using Flux

function build_dbn_from_rbms(rbms::Vector{RBM}, output_dim::Int)
    layers = []

    # First layer: input_dim should be rbms[1].W |> size |> second
    input_dim = size(rbms[1].W, 2)
    for rbm in rbms
        output_dim_rbm = size(rbm.W, 1)
        push!(layers, Dense(input_dim, output_dim_rbm, relu))
        input_dim = output_dim_rbm
    end

    # Output layer (e.g., for classification)
    push!(layers, Dense(input_dim, output_dim))  # softmax added in loss function

    return Chain(layers...)
end

function build_dbn_with_softmax_from_rbms(rbms::Vector{RBM}, output_dim::Int)
    layers = []
    for rbm in rbms
        layer = Dense(size(rbm.W, 2), size(rbm.W, 1), relu)
        layer.weight .= rbm.W
        layer.bias .= rbm.hbias
        push!(layers, layer)
    end
    last_hidden = size(rbms[end].W, 1)
    push!(layers, Dense(last_hidden, output_dim))  # final layer before softmax
    push!(layers, Flux.softmax)
    return Chain(layers...)
end




function build_model(hyperparams, input_dim)
    h1, h2, _ = hyperparams
    return Chain(
        Dense(input_dim, round(Int, h1), relu),
        Dense(round(Int, h1), round(Int, h2), relu),
        Dense(round(Int, h2), 1),
        sigmoid
    )
end

function evaluate_model(model, X_test, y_test)
    y_pred = model(X_test') |> vec
    y_pred_class = y_pred .>= 0.45


    acc = accuracy(y_pred_class, y_test)
    println("\nModel Evaluation Metrics:")
    println("  Accuracy: ", round(acc, digits=4))

    # Confusion matrix
    tp = sum((y_test .== 1) .& (y_pred_class .== 1))
    tn = sum((y_test .== 0) .& (y_pred_class .== 0))
    fp = sum((y_test .== 0) .& (y_pred_class .== 1))
    fn = sum((y_test .== 1) .& (y_pred_class .== 0))

    println("  True Positives: $tp")
    println("  True Negatives: $tn")
    println("  False Positives: $fp")
    println("  False Negatives: $fn")

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    println("  Precision: ", round(precision, digits=4))
    println("  Recall:    ", round(recall, digits=4))
    println("  F1 Score:  ", round(f1, digits=4))

    # Confusion matrix plot


# Define confusion matrix values
# tp = 127
# tn = 151
# fp = 23
# fn = 84

    cm = [tn fp; fn tp]  # Actual (rows) vs Predicted (columns)
    labels = ["Negative", "Positive"]

    # Convert numbers to strings for annotation
    annotations = [string(cm[i, j]) for i in 1:2, j in 1:2]

    # Plot
    heatmap(
        labels,
        labels,
        cm,
        xlabel = "Predicted",
        ylabel = "Actual",
        title = "Confusion Matrix",
        annotations = (1:2, 1:2, annotations),
        cbar_title = "Count",
        color = :blues,
        size = (500, 400)
    )

    # Save or display
    savefig("output/confusion_matrix.png")
    println("Saved confusion matrix to output/confusion_matrix.png")



    return acc
end
