import numpy as np

def read_wall_file(wall_file):
    wall_particles = []
    with open(wall_file, 'r') as f:
        # Skip the first 9 lines
        for _ in range(9):
            next(f)
        # Read the rest of the lines
        for line in f:
            # Assuming the first column contains the ID of the wall particle
            wall_particles.append(int(line.strip().split()[0]))
    return wall_particles

def calculate_neighbors(atom_positions, cutoff_distance, box_length_x):
    n_atoms = len(atom_positions)
    num_neighbors = np.zeros(n_atoms, dtype=int)
    
    for i in range(n_atoms):
        x_i, y_i = atom_positions[i]

        for j in range(n_atoms):
            if i != j:
                x_j, y_j = atom_positions[j]

                # Minimum image convention in x-direction (periodic)
                dx = abs(x_i - x_j)
                if dx > 0.5 * box_length_x:
                    dx = box_length_x - dx

                dy = abs(y_i - y_j)

                distance = np.sqrt(dx**2 + dy**2)
                if distance <= cutoff_distance:
                    num_neighbors[i] += 1
    
    return num_neighbors

def process_lammps_dump(lammps_dump_file, wall_file, cutoff_distance, train_fraction=0.8):
    # Read wall particles IDs
    wall_particles = read_wall_file(wall_file)
    
    # Read LAMMPS dump file
    atom_data = []
    with open(lammps_dump_file, 'r') as f:
        lines = f.readlines()

    # Extract box_length_x from line 6
    xlow, xhigh = map(float, lines[5].split())
    box_length_x = xhigh - xlow

    ylow, yhigh = map(float, lines[6].split())
    box_length_y = yhigh - ylow

    # Find the line where atoms data starts
    for i, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            start_idx = i + 1
            break

    # Read atom data
    for line in lines[start_idx:]:
        if line.strip():  # skip empty lines
            data = line.split()
            atom_id = int(data[0])  # atom ID
            atom_type = int(data[1])  # atom type
            x_coord = float(data[2])  # x-coordinate
            y_coord = float(data[3])  # y-coordinate

            # Determine if the atom is a wall particle
            wall_flag = 1 if atom_id in wall_particles else 0

            # Initialize properties: prediction_property and training_flag
            prediction_property = 0  # To be updated later
            training_flag = 0  # To be updated later

            atom_data.append((atom_id, atom_type, x_coord, y_coord, wall_flag, prediction_property, training_flag))

    # Calculate number of neighbors within cutoff distance
    atom_positions = np.array([(data[2], data[3]) for data in atom_data])  # extract x, y positions
    num_neighbors = calculate_neighbors(atom_positions, cutoff_distance, box_length_x)

    # Update prediction property based on number of neighbors
    for i in range(len(atom_data)):
        atom_data[i] = atom_data[i][:5] + (num_neighbors[i],) + (atom_data[i][6],)  # update prediction_property

    # Assign training and validation flags
    n_atoms = len(atom_data)
    n_training = int(train_fraction * n_atoms)
    training_indices = np.random.choice(n_atoms, n_training, replace=False)

    for i in range(n_atoms):
        training_flag = 1 if i in training_indices else 0
        atom_data[i] = atom_data[i][:6] + (training_flag,)  # update training_flag

    # Write output to text file
    output_file = "Snapshot_1.graphdata"
    with open(output_file, 'w') as f:
        f.write(f"{box_length_x} {box_length_y} 0.0 0.0 0.0 0.0\n") # the added zeros are just placeholders to have equal-size columns
        for data in atom_data:
            f.write(f"{data[1]} {data[2]} {data[3]} {data[4]} {data[5]} {data[6]}\n")

    print(f"Output written to {output_file}")

# Example usage:
# process_lammps_dump("MD_Data/confdumpallelasticMD0.data", "MD_Data/wallpartidtype", cutoff_distance=5.0, train_fraction=0.9)
