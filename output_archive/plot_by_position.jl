using CairoMakie
using Glob  # For searching files in a directory

# function that takes LAMMPS file and returns particle IDs of particles
# within a certain range of the simulation box area

# NOTE -- must have snapshot file in current working directory

function find_particles(
    filename::String,
    training_flag::Int;
    x_percentile::Tuple{Float64, Float64} = (0.0, 1.0),
    y_percentile::Tuple{Float64, Float64} = (0.0, 1.0),
    skip_header::Int = 1)
    
    println("CAUTION: The Snapshot file used in training must be in this directory!")
    ids = Int[]
    x_positions = Float64[]
    y_positions = Float64[]
    
    # Read Snapshot for x, y positions (validation only)
    id = 1
    open(filename, "r") do io
        for _ in 1:skip_header
            readline(io)
        end
        for line in eachline(io)
            try
                fields = split(line)
                # validation flag = 0
                # training   flag = 1
                if fields[6] == "$training_flag"
                    push!(ids, id)
                    push!(x_positions, parse(Float64, fields[2]))
                    push!(y_positions, parse(Float64, fields[3]))
                    id += 1
                end
            catch e
                println("Skipping invalid line in $filename: $line")
            end
        end
    end

    # Get bounds for x and y
    xmin, xmax = minimum(x_positions), maximum(x_positions)
    ymin, ymax = minimum(y_positions), maximum(y_positions)

    x_lower = xmin + x_percentile[1] * (xmax - xmin)
    x_upper = xmin + x_percentile[2] * (xmax - xmin)

    y_lower = ymin + y_percentile[1] * (ymax - ymin)
    y_upper = ymin + y_percentile[2] * (ymax - ymin)

    # Find indices within the percentile range
    selected_ids = [ids[i] for i in 1:length(ids) if
        x_positions[i] ≥ x_lower && x_positions[i] ≤ x_upper &&
        y_positions[i] ≥ y_lower && y_positions[i] ≤ y_upper]

    return selected_ids
end

# Function to extract actual and predicted values for particles with IDs
function read_actual_predicted(filename, id)
    
    actual_values_single = Float64[]
    predicted_values_single = Float64[]

    # count total number of particles
    n = 0
    open(filename, "r") do file
        for line in eachline(file)
            parts = split(line)
            if length(parts) == 2  # Ensure correct format
                n += 1
            elseif parts[1] == "RMSE"
                break
            end
        end
    end

    # accumulate vals for each particle
    if id > n
        throw("particle ID greater than num of particles")
    end

    counter = 1 # count particle number
    epoch = 0 # count epochs
    open(filename, "r") do file
        for line in eachline(file)
            parts = split(line)
            if length(parts) == 2  # Ensure correct format
                if counter == id # accumulate for one particle
                    try
                        actual = parse(Float64, parts[1]) # first column: actual
                        predicted = parse(Float64, parts[2])  # Second column: predicted values
                        push!(actual_values_single, actual)
                        push!(predicted_values_single, predicted)
                    catch e
                        println("Skipping invalid line in $filename: $line")
                    end
                end
                counter += 1
            elseif parts[1] == "RMSE"
                counter = 1 # reset for new epoch
                epoch += 1
            end
        end
    end


    return (;actual = actual_values_single, pred = predicted_values_single, n_epochs = epoch)
end

function plot_predictions(lower_bound_y, upper_bound_y; training_flag::Int, snapshotfile = "Snapshot_1_displacement.graphdata", task = "Displacement", morefname = "")
    output_file = ""
    if training_flag == 0
        output_file = "Actual_and_predicted_values_val"
        title = "Validation"
    elseif training_flag == 1
        output_file = "Actual_and_predicted_values_train"
        title = "Training"
    end
    savefig = "$(lower_bound_y)_$(upper_bound_y)_$(morefname)"

    # get indices of validation particles in percentile of simulation box
    IDs = find_particles(snapshotfile, training_flag, y_percentile = (lower_bound_y, upper_bound_y))
    all_actual = []
    all_pred = []

    n_epochs = 0
    for id in IDs
        result = read_actual_predicted(output_file, id)
        push!(all_actual, result.actual)
        push!(all_pred, result.pred)
        n_epochs = result.n_epochs
    end

    actual_avg = []
    pred_avg = []

    for i in 1:n_epochs
        tot_a = 0
        tot_p = 0
        for j in 1:length(IDs)
            tot_a += all_actual[j][i]
            tot_p += all_pred[j][i]
        end
        tot_a /= length(IDs)
        tot_p /= length(IDs)

        push!(actual_avg, tot_a)
        push!(pred_avg, tot_p)
    end    



    #  plot particle actual and predicted
    fig = Figure(size=(1200, 600))
    epochs = range(1, n_epochs)
    ax1 = Axis(fig[1, 1], xlabel="Epoch", ylabel="$task (normalized)", title="$title predictions for particles with y-coords in ($(lower_bound_y), $(upper_bound_y)) percentile")
    for i in 1:length(IDs)
        lines!(ax1, epochs, all_actual[i], linewidth=1.5, color=(:lightcoral, 0.3), linestyle = :dash) # plot individual targets
        lines!(ax1, epochs, all_pred[i], linewidth=1.5, color=(:cyan, 0.3))
    end
    lines!(ax1, epochs, pred_avg, linewidth=4, color=:blue, label="Prediction Mean")
    lines!(ax1, epochs, actual_avg, linewidth=4, color=:red, linestyle=:dash, label="Target Mean")
    axislegend(ax1)
    save("$(title)_predictions_byPosition_$(task)_$(savefig).png", fig)
end

# CAUTION: wall particles are 0 to 8.4 units
# wall particles live in bottom and top 5% of y

plot_predictions(0.48, 0.52, training_flag = 1, morefname = "5e-5")
plot_predictions(0.925, 0.95, training_flag = 1, morefname = "5e-5")
plot_predictions(0.05, 0.0725, training_flag = 1, morefname = "5e-5")
