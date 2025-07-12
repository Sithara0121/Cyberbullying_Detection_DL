# src/utils.jl
using Random
using Flux

# Train-test split function
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

# Salp Swarm Algorithm (SSA)
function salp_swarm_algorithm(X_train, y_train, X_test, y_test, iterations, n_salps)
    dim = 3
    lb = [10.0, 10.0, 0.0001]
    ub = [100.0, 100.0, 0.1]
    salps = [lb .+ rand(dim) .* (ub .- lb) for _ in 1:n_salps]
    fitness = [evaluate_fitness(X_train, y_train, X_test, y_test, s) for s in salps]
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
        fitness = [evaluate_fitness(X_train, y_train, X_test, y_test, s) for s in salps]
        best_idx = argmax(fitness)
        if fitness[best_idx] > best_fit
            best_fit = fitness[best_idx]
            best_pos = copy(salps[best_idx])
        end
        println("Iteration $iter - Best fitness: $best_fit")
    end
    return best_pos
end

# Fitness evaluation function
function evaluate_fitness(X_train, y_train, X_test, y_test, hyperparams)
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
