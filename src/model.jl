# src/model.jl
using Flux

# Build the neural network model
function build_model(hyperparams, input_size)
    h1, h2, lr = hyperparams
    model = Chain(
        Dense(input_size, round(Int, h1), relu),
        Dense(round(Int, h1), round(Int, h2), relu),
        Dense(round(Int, h2), 1),
        sigmoid
    )
    return model
end

# Evaluate model accuracy
function evaluate_model(model, X_test, y_test)
    y_pred = model(X_test') |> vec
    y_pred_class = round.(y_pred)
    acc = accuracy(y_pred_class, y_test)
    return acc
end
