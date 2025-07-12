using Plots

function plot_accuracy(accuracy::Float64)
    if !isdir("output")
        mkpath("output")
    end

    bar(["Test Accuracy"], [accuracy],
        title="Model Accuracy",
        ylabel="Accuracy",
        legend=false,
        ylim=(0,1),
        size=(600,400))

    savefig("output/accuracy_plot.png")
end
