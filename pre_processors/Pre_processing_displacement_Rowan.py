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

def calculate_dx(tMD, t_0):

  # Filenames for the data of the entire simulation and the wall region
  # change "elastic" to "eq" or "inf" depending on timestep
  filenameconfwall = "MD_Data/wallpartidtype"
  filenameconfall = "MD_Data/confdumpallelasticMD"+str(tMD)+".data"
  filenameconfall_init = "MD_Data/confdumpallelasticMD"+str(t_0)+".data"

  # Reading the number of particles from the full configuration file
  with open(filenameconfall, mode='r') as fin:
      countline = 0
      for line in fin.readlines():
        countline += 1
        fields = line.split()
        if countline == 4:
            Nall = int(fields[0])
        elif countline == 6:
            Lx = float(fields[1])
        elif countline == 7:
            Ly = float(fields[1])
            break

  with open(filenameconfall_init, mode='r') as fin:
      countline = 0
      for line in fin.readlines():
        countline += 1
        fields = line.split()
        if countline == 4:
            Nall = int(fields[0])
        elif countline == 6:
            Lx = float(fields[1])
        elif countline == 7:
            Ly = float(fields[1])
            break

  # Reading the number of particles in the wall region from the wall region file
  with open(filenameconfwall, mode='r') as fin:
      countline = 0
      for line in fin.readlines():
        countline += 1
        fields1 = line.split()
        if countline == 4:
            Nwall = int(fields1[0])
            break

  # Calculate the number of particles in the mid region (excluding the wall region)
  Nmid = Nall - Nwall

  # Load data from the entire simulation file.
  header = np.loadtxt(filenameconfall, skiprows=8, max_rows=1, dtype=np.dtype(str))
  header[4] = 'dx'
  header[5] = 'dy'
  data = np.loadtxt(filenameconfall, skiprows=9, dtype='float', usecols=(0, 1,
  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), unpack=True)
  data_init = np.loadtxt(filenameconfall_init, skiprows=9, dtype='float', usecols=(0, 1,
  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), unpack=True)
  idarrayall = data[0]
  typearrayall = data[1]
  xarrayall = data[2]
  yarrayall = data[3]
  vxarrayall = data[4]
  vyarrayall = data[5]
  fxarrayall = data[6]
  fyarrayall = data[7]
  sigxyarrayall = data[8]
  sigxyarrayall1 = data[9]
  sigxyarraywall2 = data[10]
  v_coordA = data[11]
  v_coordB = data[12]
  v_coordA_wall = data[13]
  v_coordB_wall = data[14]
  v_coordA_pin = data[15]
  v_coordB_pin = data[16]
  # initial x and y positions:
  xarrayall_init = data_init[2]
  yarrayall_init = data_init[3]
  # initial velocities:
  vxarrayall_init = data_init[4]
  vyarrayall_init = data_init[5]
  # displacements:
  dxarrayall = xarrayall - data_init[2]
  dyarrayall = yarrayall - data_init[3]
  # Load data from the wall region file.
  data = np.loadtxt(filenameconfwall, skiprows=9, dtype='float', usecols=(0,
  1), unpack=True)
  idarraywall = data[0]
  typearraywall = data[1]

  # Create arrays to store mid region particle data.
  # First, use idarraywall to create array of wall particle coordinates
  xarraywall = []
  yarraywall = []
  xarraywall_init = []
  yarraywall_init = []
  dxarraywall = []
  dyarraywall = []
  for id in idarrayall:
    if id in idarraywall:
      idint = int(id-1) #Do identifiers start at 0 or 1? The “-1” fixes issue.
      #print(True, " ", id, " ", yarrayall[idint])
      xarraywall = np.append(xarraywall, xarrayall[idint])
      yarraywall = np.append(yarraywall, yarrayall[idint])
      xarraywall_init = np.append(xarraywall_init, xarrayall_init[idint])
      yarraywall_init = np.append(yarraywall_init, yarrayall_init[idint])
      dxarraywall = np.append(dxarraywall, dxarrayall[idint])

  # Getting the info on the mid region by removing the wall particles from the
  # entire simulation data.
  _, wallpart_index, _ = np.intersect1d(idarrayall, idarraywall,
  return_indices=True)
  idarraymid = np.delete(idarrayall, wallpart_index)
  typearraymid = np.delete(typearrayall, wallpart_index)
  xarraymid = np.delete(xarrayall, wallpart_index)
  yarraymid = np.delete(yarrayall, wallpart_index)
  xarraymid_init = np.delete(xarrayall_init, wallpart_index)
  yarraymid_init = np.delete(yarrayall_init, wallpart_index)
  dxarraymid = np.delete(dxarrayall, wallpart_index)
  dyarraymid = np.delete(dyarrayall, wallpart_index)
  vxarraymid = np.delete(vxarrayall, wallpart_index)
  vyarraymid = np.delete(vyarrayall, wallpart_index)
  fxarraymid = np.delete(fxarrayall, wallpart_index)
  fyarraymid = np.delete(fyarrayall, wallpart_index)
  sigxyarraymid = np.delete(sigxyarrayall, wallpart_index)
  # Convert mid region arrays to integer and float data types.
  idarraymid = idarraymid.astype(int)
  typearraymid = typearraymid.astype(int)
  # Calculate the number of particles of each type in the mid region.
  Namid = (typearraymid == 1).sum()
  Nbmid = (typearraymid == 2).sum()
  Npinmid = (typearraymid == 3).sum()
  # Convert wall region arrays to integer data type.
  idarraywall = idarraywall.astype(int)
  typearraywall = typearraywall.astype(int)

  # Create arrays to store mid region particle data.
  # First, use idarraywall to create array of wall particle coordinates
  xarraywall = []
  yarraywall = []
  xarraywall_init = []
  yarraywall_init = []
  dxarraywall = []
  dyarraywall = []
  for id in idarrayall:
    if id in idarraywall:
      idint = int(id-1) #Do identifiers start at 0 or 1? The “-1” fixes issue.
      #print(True, " ", id, " ", yarrayall[idint])
      xarraywall = np.append(xarraywall, xarrayall[idint])
      yarraywall = np.append(yarraywall, yarrayall[idint])
      xarraywall_init = np.append(xarraywall_init, xarrayall_init[idint])
      yarraywall_init = np.append(yarraywall_init, yarrayall_init[idint])
      dxarraywall = np.append(dxarraywall, dxarrayall[idint])

  # Getting the info on the mid region by removing the wall particles from the
  # entire simulation data.
  _, wallpart_index, _ = np.intersect1d(idarrayall, idarraywall,
  return_indices=True)
  idarraymid = np.delete(idarrayall, wallpart_index)
  typearraymid = np.delete(typearrayall, wallpart_index)
  xarraymid = np.delete(xarrayall, wallpart_index)
  yarraymid = np.delete(yarrayall, wallpart_index)
  xarraymid_init = np.delete(xarrayall_init, wallpart_index)
  yarraymid_init = np.delete(yarrayall_init, wallpart_index)
  dxarraymid = np.delete(dxarrayall, wallpart_index)
  dyarraymid = np.delete(dyarrayall, wallpart_index)
  vxarraymid = np.delete(vxarrayall, wallpart_index)
  vyarraymid = np.delete(vyarrayall, wallpart_index)
  fxarraymid = np.delete(fxarrayall, wallpart_index)
  fyarraymid = np.delete(fyarrayall, wallpart_index)
  sigxyarraymid = np.delete(sigxyarrayall, wallpart_index)
  # Convert mid region arrays to integer and float data types.
  idarraymid = idarraymid.astype(int)
  typearraymid = typearraymid.astype(int)
  # Calculate the number of particles of each type in the mid region.
  Namid = (typearraymid == 1).sum()
  Nbmid = (typearraymid == 2).sum()
  Npinmid = (typearraymid == 3).sum()
  # Convert wall region arrays to integer data type.
  idarraywall = idarraywall.astype(int)
  typearraywall = typearraywall.astype(int)

  # calculate displacements for mid region particles
  # don't do it for wall particles because we know they're displacing at shear rate
  # so wouldn't be very interesting to visualize
  dxarraymid = xarraymid - xarraymid_init
  dyarraymid = yarraymid - yarraymid_init
  return dxarrayall

def process_lammps_dump_displacement(lammps_dump_file, wall_file, tMD, t0, cutoff_distance, train_fraction=0.8):
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

    # Calculate x-displacements for mid-region atoms
    x_displacements = calculate_dx(tMD, t0)
    
    # Update prediction property
    displacement_idx = 0  # index for x_displacements
    wall_counter = 0
    for i in range(len(atom_data)):
        displacement_value = x_displacements[displacement_idx]
        displacement_idx += 1
    
        atom_data[i] = atom_data[i][:5] + (displacement_value,) + (atom_data[i][6],)


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
# process_lammps_dump_displacement("MD_Data/confdumpallelasticMD0.data", "MD_Data/wallpartidtype", 10100000, 10000000, cutoff_distance=5.0, train_fraction=0.9)
