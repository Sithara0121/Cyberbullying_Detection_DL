using Statistics
using Plots

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
