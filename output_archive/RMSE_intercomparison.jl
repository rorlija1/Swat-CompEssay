# Function to extract actual and predicted values from a file
using CairoMakie
using Glob  # For searching files in a directory

# Function to extract RMSE values from a file
function read_rmse_values(fstring)
    filename = filename = joinpath(pwd(), "neighbors", fstring, "Actual_and_predicted_values_val")

    rmse_values = Float64[]
    open(filename, "r") do file
        for line in eachline(file)
            if startswith(line, "RMSE is ")
                value = parse(Float64, split(line)[end])  # Extract numeric value
                push!(rmse_values, value)
            end
        end
    end
    return rmse_values
end

# Load RMSE values

fstrings = ["MPL=7_edge_thresh=5_lr=1e-4_500epoch", "Exponential+MultiStepLR_betterparams_500ep", "Exponential+MultiStepLR_500ep", "CosineAnnealing+ReduceOnPlateauLR_500ep"]
titles = ["No Scheduler", "Exponential + MultiStep + Plateau (2nd)", "Exponential + MultiStep + Plateau (1st)", "Cosine Anneal + Plateau"]
rmses = []
epochs = [100, 200, 300, 400, 500]
for string in fstrings
    rmse_values_val = read_rmse_values(string)
    best_rmses = []
    for j in epochs
        push!(best_rmses, minimum(rmse_values_val[1:j]))
    end
    push!(rmses, best_rmses)
end
println(rmses)

# Plot RMSE vs. Epoch for Validation
fig = Figure(size=(1200, 600))
ax1 = Axis(fig[1, 1], xlabel="Epoch", ylabel="Best RMSE", title="Best RMSEs (neighbors) for Scheduler Choices")
for i in 1:4
    lines!(ax1, epochs, rmses[i], linewidth=4, label=titles[i])
    scatter!(ax1, epochs, rmses[i], color=:black, markersize=10)
end

# lines!(ax1, epochs, rmse_values_train, linewidth=2, color=:cyan, label="Training")
# scatter!(ax1, epochs, rmse_values_train, color=:black, markersize=5)
# lines!(ax1, epochs, fill(best_rmse, length(epochs)), color=:mediumseagreen, linestyle=:dash, label="Best")
axislegend(ax1)
save("BestRMSE_Schedulers.png", fig)