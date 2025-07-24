# Function to extract actual and predicted values from a file
using CairoMakie
using Glob  # For searching files in a directory
using FileIO
using Statistics

# Function to read normalization parameters
function read_normalization_params(filename="normalization_params.txt"; output_name = '_')
    std_path = joinpath("output_$(output_name)", filename)
    required_files = [std_path]

    for file in required_files
        isfile(file) || error("Missing required file: $file")
    end
    target_mean = 0.0
    target_std = 1.0
    
    if isfile(std_path)
        open(std_path, "r") do file
            for line in eachline(file)
                parts = split(line)
                if length(parts) == 2
                    if parts[1] == "target_mean"
                        target_mean = parse(Float64, parts[2])
                    elseif parts[1] == "target_std"
                        target_std = parse(Float64, parts[2])
                    end
                end
            end
        end
        println("Loaded normalization params: mean=$target_mean, std=$target_std")
    else
        println("Warning: normalization_params.txt not found. Using default values.")
    end
    
    return target_mean, target_std
end

# Function to extract RMSE values from a file
function read_rmse_values(filename)
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

# Function to extract actual and predicted values from a file
# also extracts actual and predicted for one particle across all epochs
# (if particle_plot = true)
function read_actual_predicted(filename, rmse_values, output_name; particle_plot = false, task = "", savefig = "", units = "standardized", best_epoch = 0)
    actual_values = Float64[]
    predicted_values = Float64[]
    actual_values_parity = Float64[]
    predicted_values_parity = Float64[]
    n_epochs = length(rmse_values)

    MLfiles = "output_$(output_name)"
    outdir = joinpath(MLfiles, "plots_$(task)_$(savefig)")
    isdir(outdir) || mkpath(outdir)

    if particle_plot
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

        # pick random particle and accumulate vals
        choice = rand(range(1, n))

        counter = 1 # count particle number
        open(filename, "r") do file
            for line in eachline(file)
                parts = split(line)
                if length(parts) == 2  # Ensure correct format
                    if counter == choice # accumulate for one particle
                        try
                            actual = parse(Float64, parts[1])  # First column: actual values
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
                end
            end
        end

        #  plot particle actual and predicted
        fig = Figure(size=(1200, 600))
        epochs = range(1, n_epochs)
        ylabel_text = units == "standardized" ? "Target (normalized)" : "Target (original units)"
        ax1 = Axis(fig[1, 1], xlabel="Epoch", ylabel=ylabel_text, 
                   title="Validation predictions; particle #$(choice) ($units)")
        lines!(ax1, epochs, predicted_values_single, linewidth=2, color=:darkcyan, label="Prediction")
        scatter!(ax1, epochs, predicted_values_single, color=:black, markersize=5)
        lines!(ax1, epochs, actual_values_single, linewidth=2, color=:cyan, linestyle=:dash, label="Target")
        axislegend(ax1, position=:lt)
        fname = "1Particle_$(task)_$(units)_$(savefig).png"
        save(joinpath(outdir, fname), fig)
    end

    epoch = 1 # counter
    if best_epoch == 0
        best_rmse, best_epoch = findmin(rmse_values)
    elseif best_epoch != 0
        best_rmse = rmse_values[best_epoch-1]
    end
    open(filename, "r") do file
        for line in eachline(file)
            parts = split(line)
            if epoch == best_epoch
                # save output from best epoch to use for parity plot
                if length(parts) == 2  # Ensure correct format
                    try
                        actual = parse(Float64, parts[1])  # First column: actual values
                        predicted = parse(Float64, parts[2])  # Second column: predicted values
                        push!(actual_values_parity, actual)
                        push!(predicted_values_parity, predicted)
                    catch e
                        println("Skipping invalid line in $filename: $line")
                    end
                end
            end
            if length(parts) == 2  # Ensure correct format
                try
                    actual = parse(Float64, parts[1])  # First column: actual values
                    predicted = parse(Float64, parts[2])  # Second column: predicted values
                    push!(actual_values, actual)
                    push!(predicted_values, predicted)
                catch e
                    println("Skipping invalid line in $filename: $line")
                end
            elseif parts[1] == "RMSE"
                epoch += 1
            end
        end
    end
    return actual_values, predicted_values, actual_values_parity, predicted_values_parity
end

function read_lr(filename = "Learning_rates"; output_name = '_')
    lr_path = joinpath("output_$(output_name)", filename)
    required_files = [lr_path]

    for file in required_files
        isfile(file) || error("Missing required file: $file")
    end

    lrs = Float64[]
    open(lr_path, "r") do file
        for line in eachline(file)
            value = parse(Float64, split(line)[end])  # Extract numeric value
            push!(lrs, value)
        end
    end
    return lrs
end

function safe_lines_and_scatter!(ax, x, y; scatter_only=false, origin="", label="", kwargs...)
    # Split kwargs manually
    linekwargs = get(kwargs, :linekwargs, NamedTuple())
    scatterkwargs = get(kwargs, :scatterkwargs, NamedTuple())

    if length(x) != length(y)
        error_msg = """
        🚫 Dimension mismatch in plotting at: $origin \n → x has length $(length(x)), y has length $(length(y)) \n → size(x) = $(size(x)), size(y) = $(size(y))
        """
        throw(DimensionMismatch(error_msg))
    end
    !scatter_only ? lines!(ax, x, y; label=label, linekwargs...) :
    scatter!(ax, x, y; scatterkwargs...)
end

function compute_r2(y_true::Vector{Float64}, y_pred::Vector{Float64})
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return 1.0 - ss_res / ss_tot
end


function make_plots(task, savefig; display_only=false, output_name='_')

    # Output directory
    MLfiles = "output_$(output_name)"
    outdir = joinpath(MLfiles, "plots_$(task)_$(savefig)")
    isdir(outdir) || mkpath(outdir)

    function handle_output(fig, name)
        if display_only
            display(fig)
        else
            save(joinpath(outdir, name), fig)
        end
    end

    # File checks
    val_file_std = joinpath(MLfiles, "Actual_and_predicted_values_val_standardized")
    train_file_std = joinpath(MLfiles, "Actual_and_predicted_values_train_standardized")
    val_file_orig = joinpath(MLfiles, "Actual_and_predicted_values_val_original")
    train_file_orig = joinpath(MLfiles, "Actual_and_predicted_values_train_original")
    required_files = [val_file_std, train_file_std, val_file_orig, train_file_orig]

    for file in required_files
        isfile(file) || error("Missing required file: $file")
    end

    target_mean, target_std = read_normalization_params(output_name = output_name)
    rmse_values_val = read_rmse_values(val_file_std)
    rmse_values_train = read_rmse_values(train_file_std)
    best_rmse, best_epoch = findmin(rmse_values_val)
    best_rmse_train = rmse_values_train[best_epoch-1]
    epochs = 1:length(rmse_values_val)

    actual_val_std, predicted_val_std, actual_val_parity_std, predicted_val_parity_std = read_actual_predicted(
        val_file_std, rmse_values_val, output_name, particle_plot=true, task=task, savefig=savefig, units="standardized")
    actual_train_std, predicted_train_std, actual_train_parity_std, predicted_train_parity_std = read_actual_predicted(
        train_file_std, rmse_values_val, output_name, task=task, savefig=savefig, units="standardized", best_epoch=best_epoch)

    rmse_values_val_orig = read_rmse_values(val_file_orig)
    rmse_values_train_orig = read_rmse_values(train_file_orig)
    best_rmse_orig, best_epoch_orig = findmin(rmse_values_val_orig)
    best_rmse_train_orig = rmse_values_train_orig[best_epoch_orig-1]

    actual_val_orig, predicted_val_orig, actual_val_parity_orig, predicted_val_parity_orig = read_actual_predicted(
        val_file_orig, rmse_values_val_orig, output_name, particle_plot=true, task=task, savefig=savefig, units="original")
    actual_train_orig, predicted_train_orig, actual_train_parity_orig, predicted_train_parity_orig = read_actual_predicted(
        train_file_orig, rmse_values_train_orig, output_name, task=task, savefig=savefig, units="original", best_epoch=best_epoch_orig)

    # === RMSE Plot ===
    try
        fig = Figure(size=(1200, 600))
        ax1 = Axis(fig[1, 1], xlabel="Epoch", ylabel="RMSE (normalized)", title="RMSE ($task)")
        safe_lines_and_scatter!(ax1, epochs, rmse_values_val;
            origin="RMSE (Validation)",
            label="Validation",
            linekwargs=(linewidth=2, color=:darkcyan),
            scatterkwargs=(color=:black, markersize=5)
        )
        safe_lines_and_scatter!(ax1, epochs, rmse_values_train;
            origin="RMSE (Training)",
            label="Training",
            linekwargs=(linewidth=2, color=:cyan),
            scatterkwargs=(color=:black, markersize=5)
        )
        lines!(ax1, epochs, fill(best_rmse, length(epochs)), color=:mediumseagreen, linestyle=:dash, label="Best: $best_rmse (epoch $(best_epoch))")
        axislegend(ax1, position=:lb)
        handle_output(fig, "RMSE.png")
    catch e
        @warn "⚠️ Failed to plot RMSE: $e"
    end

    # === Log-log RMSE ===
    try
        fig_log = Figure(size=(1200, 600))
        ax_log = Axis(fig_log[1, 1], xlabel="log(Epoch)", ylabel="log(RMSE)", title="RMSE log-log ($task)", yscale=log10, xscale=log10)
        safe_lines_and_scatter!(ax_log, epochs, rmse_values_val;
            origin="LogRMSE (Validation)",
            label="Validation",
            linekwargs=(linewidth=2, color=:red, alpha=0.5),
            scatterkwargs=(color=:red, markersize=5)
        )
        safe_lines_and_scatter!(ax_log, epochs, rmse_values_train;
            origin="LogRMSE (Training)",
            label="Training",
            linekwargs=(linewidth=2, color=:blue, alpha=0.5),
            scatterkwargs=(color=:blue, markersize=5)
        )
        lines!(ax_log, epochs, fill(best_rmse, length(epochs)), color=:mediumseagreen, linestyle=:dash, label="Best: $best_rmse (epoch $(best_epoch))")
        axislegend(ax_log, position=:lb)
        handle_output(fig_log, "LogRMSE.png")
    catch e
        @warn "⚠️ Failed to plot Log-log RMSE: $e"
    end

    # === Parity Plots: Standardized ===
    try
        fig_std = Figure(size=(1600, 600))
        ax_val_std = Axis(fig_std[1, 1], xlabel="Actual (standardized)", ylabel="Predicted", title="$(task), $(savefig) Validation (Standardized) - Epoch $(best_epoch)")
        ax_train_std = Axis(fig_std[1, 2], xlabel="Actual (standardized)", ylabel="Predicted", title="$(task), $(savefig) Training (Standardized) - Epoch $(best_epoch)")

        safe_lines_and_scatter!(ax_val_std, actual_val_parity_std, predicted_val_parity_std;
            origin="Parity (Validation, Standardized)",
            linekwargs=(linestyle=:dash, color=:black),
            scatterkwargs=(marker=:circle, color=(:darkcyan, 0.2), strokecolor=:darkcyan, strokewidth=0.8, markersize=10),
            scatter_only = true
        )

        # Compute shared axis limits
        all_vals_std = vcat(actual_val_parity_std, predicted_val_parity_std, actual_train_parity_std, predicted_train_parity_std)
        shared_min_std = minimum(all_vals_std)
        shared_max_std = maximum(all_vals_std)

        # R² and RMSE
        r2 = compute_r2(actual_val_parity_std, predicted_val_parity_std)
        text!(ax_val_std, 0, 1,
            text= "R² = $(round(r2, digits=4)), RMSE = $(best_rmse)",
            position=:lt, align=(:left, :top), fontsize=16, space=:relative, offset = (4, -2))
    
        safe_lines_and_scatter!(ax_train_std, actual_train_parity_std, predicted_train_parity_std;
            origin="Parity (Training, Standardized)",
            linekwargs=(linestyle=:dash, color=:black),
            scatterkwargs=(marker=:circle, color=(:darkcyan, 0.2), strokecolor=:darkcyan, strokewidth=0.8, markersize=10),
            scatter_only = true
        )
        # R² and RMSE
        r2 = compute_r2(actual_train_parity_std, predicted_train_parity_std)
        text!(ax_train_std, 0, 1,
            text= "R² = $(round(r2, digits=4)), RMSE = $(best_rmse_train)",
            position=:lt, align=(:left, :top), fontsize=16, space=:relative, offset = (4, -2))

        # Apply to both plots
        xlims!(ax_val_std, shared_min_std, shared_max_std)
        ylims!(ax_val_std, shared_min_std, shared_max_std)
        xlims!(ax_train_std, shared_min_std, shared_max_std)
        ylims!(ax_train_std, shared_min_std, shared_max_std)


        # Reference line y = x
        lines!(ax_val_std, [shared_min_std, shared_max_std], [shared_min_std, shared_max_std], color=:black, linestyle=:dot)
        lines!(ax_train_std, [shared_min_std, shared_max_std], [shared_min_std, shared_max_std], color=:black, linestyle=:dot)

        handle_output(fig_std, "Parity_standardized.png")
    catch e
        @warn "⚠️ Failed to plot standardized parity plots: $e"
    end

    # === Parity Plots: Original Units ===
    try
        fig_orig = Figure(size=(1600, 600))
        ax_val_orig = Axis(fig_orig[1, 1], xlabel="Actual", ylabel="Predicted", title="$(task), $(savefig) Validation (Original Units) - Epoch $(best_epoch_orig)")
        ax_train_orig = Axis(fig_orig[1, 2], xlabel="Actual", ylabel="Predicted", title="$(task), $(savefig) Training (Original Units) - Epoch $(best_epoch_orig)")

        safe_lines_and_scatter!(ax_val_orig, actual_val_parity_orig, predicted_val_parity_orig;
            origin="Parity (Validation, Original)",
            linekwargs=(linestyle=:dash, color=:black),
            scatterkwargs=(marker=:circle, color=(:darkcyan, 0.2), strokecolor=:darkcyan, strokewidth=0.8, markersize=10),
            scatter_only = true
        )

        # R² and RMSE
        r2 = compute_r2(actual_val_parity_orig, predicted_val_parity_orig)
        text!(ax_val_orig, 0, 1,
            text= "R² = $(round(r2, digits=4)), RMSE = $(best_rmse_orig)",
            position=:lt, align=(:left, :top), fontsize=16, space=:relative, offset = (4, -2))


        safe_lines_and_scatter!(ax_train_orig, actual_train_parity_orig, predicted_train_parity_orig;
            origin="Parity (Training, Original)",
            linekwargs=(linestyle=:dash, color=:black),
            scatterkwargs=(marker=:circle, color=(:darkcyan, 0.2), strokecolor=:darkcyan, strokewidth=0.8, markersize=10),
            scatter_only = true
        )

        # R² and RMSE
        r2 = compute_r2(actual_train_parity_orig, predicted_train_parity_orig)
        text!(ax_train_orig, 0, 1,
            text= "R² = $(round(r2, digits=4)), RMSE = $(best_rmse_train_orig)",
            position=:lt, align=(:left, :top), fontsize=16, space=:relative, offset = (4, -2))

        # Compute shared axis limits
        all_vals_orig = vcat(actual_val_parity_orig, predicted_val_parity_orig, actual_train_parity_orig, predicted_train_parity_orig)
        shared_min_orig = minimum(all_vals_orig)
        shared_max_orig = maximum(all_vals_orig)

        # Apply to both plots
        xlims!(ax_val_orig, shared_min_orig, shared_max_orig)
        ylims!(ax_val_orig, shared_min_orig, shared_max_orig)
        xlims!(ax_train_orig, shared_min_orig, shared_max_orig)
        ylims!(ax_train_orig, shared_min_orig, shared_max_orig)


        # Reference line y = x
        lines!(ax_val_orig, [shared_min_orig, shared_max_orig], [shared_min_orig, shared_max_orig], color=:black, linestyle=:dot)
        lines!(ax_train_orig, [shared_min_orig, shared_max_orig], [shared_min_orig, shared_max_orig], color=:black, linestyle=:dot)


        handle_output(fig_orig, "Parity_original.png")
    catch e
        @warn "⚠️ Failed to plot original unit parity plots: $e"
    end

    # === Learning Rate Plot ===
    try
        learning_rates = read_lr(output_name = output_name)
        fig_lr = Figure(size=(1200, 600))
        ax_lr = Axis(fig_lr[1, 1], xlabel="Epoch", ylabel="Learning Rate", title="Learning Rate ($task, $savefig)")
        safe_lines_and_scatter!(ax_lr, epochs, learning_rates;
            origin="Learning Rate",
            linekwargs=(linewidth=2, color=:darkcyan),
            scatterkwargs=(color=:black, markersize=6)
        )
        handle_output(fig_lr, "LearningRate.png")
    catch e
        @warn "⚠️ Failed to plot learning rate: $e"
    end

    println("✅ Finished. Plots stored in: $outdir")
end


make_plots("Displacement", "MD50000 (t₀=0)", output_name = "final_choices_3")