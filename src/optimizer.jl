# src/optimizer.jl
using Flux

# Training the model with given learning rate
function train_model!(model, X_train, y_train, lr)
    loss(x, y) = Flux.binarycrossentropy(model(x), y)
    opt = Flux.Optimise.Descent(lr)
    data = [(X_train[i, :], y_train[i]) for i in 1:length(y_train)]
    Flux.train!(loss, Flux.params(model), data, opt)
end
