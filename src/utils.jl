using Random
using Flux
using Plots

function accuracy(y_pred, y_true)
    return sum(y_pred .== y_true) / length(y_true)
end

# Train-test split function
function train_test_split(X, y, test_ratio=0.2)
    n = size(X, 1)
    idx = collect(1:n)
    Random.seed!(42)
    shuffle!(idx)
    test_size = round(Int, n * test_ratio)
    test_idx = idx[1:test_size]
    train_idx = idx[(test_size+1):end]

    println("Train-test split:")
    println("  Total samples: $n")
    println("  Train size: ", length(train_idx))
    println("  Test size: ", length(test_idx))

    return X[train_idx, :], y[train_idx], X[test_idx, :], y[test_idx]
end

# Salp Swarm Algorithm (SSA)
function salp_swarm_algorithm(X_train, y_train, X_test, y_test, iterations, n_salps)
    println("\nRunning Salp Swarm Algorithm (SSA)")
    dim = 3
    lb = [10.0, 10.0, 0.0001]   # Lower bounds: [hidden1, hidden2, learning_rate]
    ub = [200.0, 100.0, 0.05]   # Upper bounds extended for learning rate

    salps = [lb .+ rand(dim) .* (ub .- lb) for _ in 1:n_salps]
    fitness = [evaluate_fitness(X_train, y_train, X_test, y_test, s) for s in salps]
    best_idx = argmax(fitness)
    best_pos = copy(salps[best_idx])
    best_fit = fitness[best_idx]

    fitness_over_time = Float64[]

    for iter in 1:iterations
        c1 = 2 * exp(-((4 * iter / iterations)^2))

        for i in 1:n_salps
            for d in 1:dim
                if i == 1
                    # Leader update
                    c2, c3 = rand(), rand()
                    move = c1 * ((ub[d] - lb[d]) * c2 + lb[d])
                    salps[i][d] = best_pos[d] + move * (c3 < 0.5 ? 1 : -1)
                else
                    # Follower update
                    salps[i][d] = (salps[i][d] + salps[i - 1][d]) / 2
                end

                # Add small Gaussian noise (mutation)
                salps[i][d] += randn() * 0.5

                # Clip to bounds
                salps[i][d] = clamp(salps[i][d], lb[d], ub[d])
            end
        end

        fitness = [evaluate_fitness(X_train, y_train, X_test, y_test, s) for s in salps]
        best_idx = argmax(fitness)
        if fitness[best_idx] > best_fit
            best_fit = fitness[best_idx]
            best_pos = copy(salps[best_idx])
        end

        push!(fitness_over_time, best_fit)

        println("Epoch $iter:")
       # println("  Best fit: ", round(best_fit, digits=4))
       # println("  Best hyperparams: ", round.(best_pos, digits=4))
    end

    # Plot fitness over time
    plot(1:iterations, fitness_over_time,
         title="SSA Optimization Progress",
         xlabel="Iteration",
         ylabel="Best Fitness",
         legend=false)
    savefig("output/ssa_fitness_plot.png")
    println("SSA fitness plot saved to output/ssa_fitness_plot.png")

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
        #y_pred_class = round.(y_pred)
        y_pred_class = y_pred .>= 0.45

        acc = accuracy(y_pred_class, y_test)

        #println("    Evaluated hyperparams: h1=$(round(h1)), h2=$(round(h2)), lr=$(round(lr, sigdigits=3)) -> Accuracy=$(round(acc, digits=4))")

        return acc
    catch e
        println("    Fitness evaluation failed: $e")
        return 0.0
    end
end
