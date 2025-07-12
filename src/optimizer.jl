using Flux

# Training the model with given learning rate
function train_model!(model, X_train, y_train, lr)
    println("\nTraining model...")
    println("  Learning rate: $lr")
    println("  Training samples: ", size(X_train, 1))

    loss(x, y) = Flux.binarycrossentropy(model(x), y)

    opt = Flux.Optimise.Descent(lr)
    data = [(X_train[i, :], y_train[i]) for i in 1:length(y_train)]

    # Log initial loss
    initial_losses = [loss(d[1], d[2]) for d in data[1:min(end, 10)]]
    println("  Avg initial loss (first 10 samples): ", round(mean(initial_losses), digits=4))

    Flux.train!(loss, Flux.params(model), data, opt)

    # Log final loss
    final_losses = [loss(d[1], d[2]) for d in data[1:min(end, 10)]]
    println("  Avg final loss (first 10 samples): ", round(mean(final_losses), digits=4))

    # Log predictions after training
    example_preds = model(X_train[1:10, :]') |> vec
    println("  First 10 predictions after training: ", round.(example_preds, digits=3))
end
