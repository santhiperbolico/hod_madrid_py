"""
Input/output and information functions for HOD simulation
"""

import os
from pathlib import Path
from typing import Tuple, NamedTuple
import numpy as np
import src.hod_const as c

# Define a parameters structure
class HODParams(NamedTuple):
    """Container for all HOD parameters"""
    infile: str
    outfile: str
    ftype: str
    seed: int
    zsnap: float
    Lbox: float
    omega_M: float
    analytical_shape: bool
    HODfit2sim: bool
    hod_shape_file: str
    conformity: bool
    conformity_file: str
    hodshape: int
    mu: float
    Ac: float
    As: float
    M0: float
    M1: float
    alpha: float
    sig: float
    gamma: float
    beta: float
    analytical_rp: bool
    K: float
    extended_NFW: bool
    hod_rp_file: str
    N0: float
    r0: float
    alpha_r: float
    beta_r: float
    kappa_r: float
    analytical_vp: bool
    hod_vp_file: str
    extended_vp: bool
    vfact: float
    vt: float
    vtdisp: float
    vr1: float
    vr2: float
    vr3: float
    mu1: float
    mu2: float
    mu3: float
    sigma1: float
    sigma2: float
    sigma3: float
    v0_tan: float
    epsilon_tan: float
    omega_tan: float
    delta_tan: float


def create_hod_params(infile=None, outdir=None, ftype='txt', seed=50,
                      zsnap=None, Lbox=None, omega_M=None,
                      analytical_shape=True, HODfit2sim=False, hod_shape_file=None,
                      conformity=False, conformity_file=None,
                      hodshape=None, mu=12, Ac=1.0, As=0.5,
                      M0=10**11.95, M1=10**12.35, alpha=0.9, sig=0.08, gamma=-1.4,
                      beta=0,
                      analytical_rp=True, K=1, extended_NFW=True,  hod_rp_file=None,
                      N0=None, r0=None, alpha_r=None, beta_r=None, kappa_r=None,
                      analytical_vp=True, hod_vp_file=None, extended_vp=True,
                      vfact=1, vt=0, vtdisp=0,
                      vr1=None, vr2=None, vr3=None,
                      mu1=None, mu2=None, mu3=None,
                      sigma1=None, sigma2=None, sigma3=None,
                      v0_tan=None, epsilon_tan=None, omega_tan=None, delta_tan=None,
                      ):
    """Create HOD parameters structure"""
    if gamma is None:
        gamma =c.default_gamma

    # Check input file
    if not check_input_file(infile):
        print("STOP (hod_io): Please check your input file.")
        return None

    # Check output file
    outnom = (f"galaxies_{Lbox:.0f}Mpc_NFW_"
              f"mu{mu:.3f}_Ac{Ac:.4f}_As{As:.5f}_"
              f"vfact{vfact:.2f}_beta{beta:.3f}_K{K:.2f}_"
              f"vt{vt:.0f}pm{vtdisp:.0f}"
              f".dat")
    outfile = outdir / outnom
    if not check_output_file(outfile):
        print("STOP (hod_io): Please check your output path.")
        return None

    # Check files if not analytical expressions
    if not analytical_shape and (hod_shape_file is None):
        print("STOP (hod_io): For a not analytical HOD shape, a file,")
        print ("    hod_shape_file, should be provided")
        return None
    if not analytical_rp and (hod_rp_file is None):
        print("STOP (hod_io): For a not analytical radial profile, a file,")
        print("     hod_rp_file, should be provided")
        return None
    if not analytical_vp and (hod_vp_file is None):
        print("STOP (hod_io): For a not analytical velocity profile, a file,")
        print("     hod_vp_file, should be provided")
        return None
        
        
    return HODParams(infile=str(infile), outfile=str(outfile), ftype=ftype, seed=seed,
                     zsnap=zsnap, Lbox=Lbox, omega_M=omega_M,
                     analytical_shape=analytical_shape, HODfit2sim=HODfit2sim, hod_shape_file=hod_shape_file,
                     conformity=conformity, conformity_file=conformity_file,
                     hodshape=hodshape, mu=mu, Ac=Ac, As=As,
                     M0=M0, M1=M1, alpha=alpha, sig=sig, gamma=gamma,
                     beta=beta,
                     analytical_rp=analytical_rp,  K=K, extended_NFW=extended_NFW, hod_rp_file=hod_rp_file,
                     N0=N0, r0=r0, alpha_r=alpha_r, beta_r=beta_r, kappa_r=kappa_r,
                     analytical_vp=analytical_vp, hod_vp_file=hod_vp_file, extended_vp=extended_vp,
                     vfact=vfact, vt=vt, vtdisp=vtdisp,
                     vr1=vr1, vr2=vr2, vr3=vr3,
                     mu1=mu1, mu2=mu2, mu3=mu3,
                     sigma1=sigma1, sigma2=sigma2, sigma3=sigma3,
                     v0_tan=v0_tan, epsilon_tan=epsilon_tan, omega_tan=omega_tan, delta_tan=delta_tan
                    )

    
    
def create_output_directory(output_file):
    """Create output directory if it doesn't exist"""
    output_dir = Path(output_file).parent
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"ERROR: Could not create output directory {output_dir}: {e}")
        return False

    
def check_input_file(filename):
    """Check if input file exists and is readable"""
    if not os.path.exists(filename):
        print(f"ERROR: Input file does not exist: {filename}")
        return False
    
    if not os.path.isfile(filename):
        print(f"ERROR: Input path is not a file: {filename}")
        return False
    
    if not os.access(filename, os.R_OK):
        print(f"ERROR: Input file is not readable: {filename}")
        return False
    
    return True

    
def check_output_file(filename):
    """Check if output file can be created"""
    output_path = Path(filename)
    
    # Check if parent directory exists or can be created
    if not create_output_directory(filename):
        return False
    
    # Check if file already exists and warn user
    if output_path.exists():
        print(f"WARNING: Output file already exists and will be overwritten: {filename}")
    
    # Check if directory is writable
    if not os.access(output_path.parent, os.W_OK):
        print(f"ERROR: Output directory is not writable: {output_path.parent}")
        return False
    
    return True

    

def validate_parameters(params):
    """Validate parameter ranges and combinations"""
    warnings = []
    errors = []
    
    # Check that all parameters that should be positive are positive
    positive_params = ['mu', 'Ac', 'As', 'vfact', 'K', 'M0', 'M1',
                       'alpha', 'sig', 'Lbox', 'zsnap']
    for param_name in positive_params:
        param_value = getattr(params, param_name)
        if param_value <= 0.0:
            errors.append(f"{param_name} = {param_value:.6g} must be positive")
    
    # Check omega_M range (0 <= omega_M <= 1)
    if params.omega_M < 0.0 or params.omega_M > 1.0:
        errors.append(f"omega_M = {params.omega_M:.4f} must be between 0 and 1 (inclusive)")

   # Check hodshape is a valid integer
    if not isinstance(params.hodshape, int) or params.hodshape < 0:
        errors.append(f"hodshape = {params.hodshape} must be a non-negative integer")
        
    # Print warnings and errors
    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    
    if errors:
        print("ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print()
        return False
    
    return True


def line_separator():
    # width = min(os.get_terminal_size().columns, 80)
    separator = "=" * 60
    return separator


def read_halo_data_chunked(filename, ftype='txt', chunk_size=c.chunk_size):
    """
    Read halo data in chunks for memory-efficient processing
    
    Parameters:
    -----------
    filename : str
        Path to input file
    ftype : str
        File type: 'txt' or 'hdf5'
    chunk_size : int
        Number of halos to read per chunk
        
    Yields:
    -------
    numpy.ndarray: chunk of halo data with shape (n_halos, 8)
        columns: x, y, z, vx, vy, vz, logM, halo_id
    """
    if ftype.lower() == 'hdf5':
        yield from read_hdf5_chunked(filename, chunk_size)
    else:
        yield from read_txt_chunked(filename, chunk_size)


def read_txt_chunked(filename, chunk_size=c.chunk_size):
    """Read text file in chunks"""
    try:
        with open(filename, 'r') as f:
            # Skip header if present
            first_line = f.readline().strip()
            if first_line.startswith('#'):
                first_line = f.readline().strip()
            
            chunk_data = []
            
            # Process first line if it exists
            if first_line and not first_line.startswith('#'):
                parts = first_line.split()
                if len(parts) >= 8:
                    chunk_data.append([float(p) for p in parts[:7]] + [int(float(parts[7]))])
            
            # Process remaining lines
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 8:
                        chunk_data.append([float(p) for p in parts[:7]] + [int(float(parts[7]))])

                        # Yield chunk when it reaches chunk_size
                        if len(chunk_data) >= chunk_size:
                            yield np.array(chunk_data)
                            chunk_data = []
            
            # Yield remaining chunk if not empty
            if chunk_data:
                yield np.array(chunk_data)
                
    except FileNotFoundError:
        print(f"ERROR: Could not open input file {filename}")
        return
    except Exception as e:
        print(f"ERROR: Error reading file {filename}: {e}")
        return


def read_hdf5_chunked(filename, chunk_size=10000):
    """Read HDF5 file in chunks"""
    try:
        import h5py
        
        with h5py.File(filename, 'r') as f:
            # Try common dataset names for halo catalogs
            possible_names = ['halos', 'Halo', 'data', 'catalog']
            dataset = None
            
            for name in possible_names:
                if name in f:
                    dataset = f[name]
                    break
            
            if dataset is None:
                # List available datasets
                print(f"Available datasets in {filename}:")
                f.visit(print)
                raise ValueError("Could not find halo data. Please specify the correct dataset name.")
            
            # Try to read standard columns
            # Common column names in halo catalogs
            col_mappings = {
                'x': ['x', 'X', 'pos_x', 'position_x'],
                'y': ['y', 'Y', 'pos_y', 'position_y'], 
                'z': ['z', 'Z', 'pos_z', 'position_z'],
                'vx': ['vx', 'VX', 'vel_x', 'velocity_x'],
                'vy': ['vy', 'VY', 'vel_y', 'velocity_y'],
                'vz': ['vz', 'VZ', 'vel_z', 'velocity_z'],
                'logM': ['logM', 'log_mass', 'mass', 'M', 'log10_mass'],
                'id': ['id', 'ID', 'halo_id', 'haloid'],
            }
            
            # Find actual column names
            if hasattr(dataset, 'dtype') and dataset.dtype.names:
                # Structured array
                columns = {}
                for standard_name, possible_names in col_mappings.items():
                    for possible_name in possible_names:
                        if possible_name in dataset.dtype.names:
                            columns[standard_name] = possible_name
                            break
                    if standard_name not in columns:
                        raise ValueError(f"Could not find column for {standard_name}")
                
                n_halos = len(dataset)
                
                # Read in chunks
                for start_idx in range(0, n_halos, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_halos)
                    chunk = dataset[start_idx:end_idx]
                    
                    # Extract columns in correct order
                    chunk_data = np.column_stack([
                        chunk[columns['x']],
                        chunk[columns['y']],
                        chunk[columns['z']],
                        chunk[columns['vx']],
                        chunk[columns['vy']],
                        chunk[columns['vz']],
                        chunk[columns['logM']],
                        chunk[columns['id']]
                    ])
                    
                    yield chunk_data
                    
            else:
                # Try as regular array (assume columns are in standard order)
                if len(dataset.shape) != 2 or dataset.shape[1] < 8:
                    raise ValueError(f"Expected 2D array with at least 8 columns, got shape {dataset.shape}")
                
                n_halos = dataset.shape[0]
                
                for start_idx in range(0, n_halos, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_halos)
                    chunk_data = dataset[start_idx:end_idx, :8]  # Take first 8 columns
                    yield chunk_data
                    
    except ImportError:
        print("ERROR: h5py is required to read HDF5 files. Install with: pip install h5py")
        return
    except Exception as e:
        print(f"ERROR: Error reading HDF5 file {filename}: {e}")
        return



def print_parameter_info():
    """Print detailed information about the parameters that can be modified"""
    print("""
    HOD Mock Generation Script
    ==========================
    Usage:
        python produce_hod_mock.py
        
    Configuration:
    - Edit parameters in the "PARAMETER DEFINITION SECTION"
    - Adjust performance thresholds in "PERFORMANCE CONFIGURATION"
    - Modify file paths in "FILE PATH CONFIGURATION"
    
    Input file format:
    - Space-separated columns: x y z vx vy vz logM halo_id
    - Lines starting with '#' are treated as comments
    - Units: positions in Mpc/h, velocities in km/s, masses in M_sun/h
    
    Output file format: x y z vx vy vz M Nsat Dvx Dvy Dvz Dx Dy Dz halo_id is_central
    
    PARAMETER MODIFICATION GUIDE:
    ============================
    - mu: Log10 of characteristic halo mass (affects both centrals and satellites)

    - Ac: Amplitude for central galaxies (0-1, typically ~1)
          Controls the maximum fraction of halos that host central galaxies
          Ac = 1.0 means all halos above threshold can host centrals
    
    - As: Amplitude for satellite galaxies (affects satellite abundance)
          Typical range: 0.1 - 2.0
          Higher values = more satellites per halo
    
    - alpha: Power law slope for satellite HOD (typically 0.8-1.2)
             Controls how satellite number scales with halo mass
             Higher values = steeper scaling with mass
    
    - sig: Scatter in central galaxy occupation (log-normal width)
           Typical range: 0.05 - 0.2
           Controls the smoothness of the central galaxy transition
    
    - gamma: Power law slope for high-mass centrals 
             Typical range: -2.0 to -1.0
             Controls central galaxy behavior in massive halos

    - beta: PDF parameter
            beta = 0: Poisson distribution (no extra scatter)
            beta > 0: Negative binomial (super-Poissonian scatter)
            beta < 0: Sub-Poissonian scatter (more regular than Poisson)
    
    - K: NFW truncation parameter (typically ~1)
         Controls where satellite orbits are truncated
         K = 1.0 means truncation at virial radius

    
    Velocity Parameters:
    -------------------
    - vfact: Velocity scaling factor for satellites (typically 0.5-1.5)
             Controls how much satellites follow halo internal motions
             vfact = 1.0 means satellites follow halo velocities exactly
    
    - vt: Systematic tangential velocity (km/s, often 0 or ~500)
          Additional radial motion component for satellites
    
    - vtdisp: Random tangential velocity dispersion (km/s, often 0 or ~200)
              Random scatter in the tangential velocity component
    
    Cosmological Parameters:
    -----------------------
    - zsnap: Redshift of the snapshot
             Affects concentration-mass relations and virial properties
    
    - omega_M: Matter density parameter
               Standard cosmological parameter (typically ~0.3)
    
    - Lbox: Simulation box size (Mpc/h)
            Used for periodic boundary conditions
    
    """)
    print(line_separator())
    return


def print_run_summary(params):
    """Print summary of run parameters"""
    print(f"HOD Parameters:")
    print(f"  hodshape = {params.hodshape}")
    print(f"  mu = {params.mu:.3f}")
    print(f"  Ac = {params.Ac:.4f}")
    print(f"  As = {params.As:.5f}")
    print(f"  vfact = {params.vfact:.2f}")
    print(f"  beta = {params.beta:.3f}")
    print(f"  K = {params.K:.2f}")
    print(f"  vt = {params.vt:.0f}")
    print(f"  vtdisp = {params.vtdisp:.0f}")
    print(f"  alpha = {params.alpha:.1f}")
    print(f"  sig = {params.sig:.2f}")
    print(f"  gamma = {params.gamma:.1f}")
    
    print(f"Derived parameters:")
    print(f"  M0 = {params.M0:.6e}")
    print(f"  M1 = {params.M1:.6e}")
    
    print(f"Cosmology:")
    print(f"  zsnap = {params.zsnap:.4f}")
    print(f"  omega_M = {params.omega_M:.4f}")
    print(f"  Lbox = {params.Lbox:.1f} Mpc/h")
    
    print(f"Files:")
    print(f"  Input: {params.infile}")
    print(f"  Output: {params.outfile}")
    print(line_separator())
    return


        
def print_file_info(input_file, nhalos):
    """Print information about input and output files"""
    print(f"Input file: {input_file}")
    print(f"Found {nhalos} halos in the catalogue")
    
    # Try to determine file size
    try:
        file_size = os.path.getsize(input_file)
        if file_size > 1024**3:  # GB
            size_str = f"{file_size / 1024**3:.1f} GB"
        elif file_size > 1024**2:  # MB
            size_str = f"{file_size / 1024**2:.1f} MB"
        elif file_size > 1024:  # KB
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size} bytes"
        
        print(f"File size: {size_str}")
    except:
        pass  # Don't fail if we can't get file size


def write_galaxies_to_file(galaxies, f_out, more_info=False):
    """
    Write galaxy data to output file
    
    Parameters:
    -----------
    galaxies : ndarray
        Array of galaxy properties with shape (n_galaxies, 16)
        Columns: x, y, z, vx, vy, vz, M, Nsat, Dvx, Dvy, Dvz, Dx, Dy, Dz, halo_id, is_central
    f_out : file handle
        Open file handle for writing
    more_info : bool
        If True, write all columns (MORE format from C code)
        If False, write only essential columns
    """
    for i in range(len(galaxies)):
        galaxy = galaxies[i, :]
        
        if more_info:
            # Write all columns (matching C code #ifdef MORE format)
            # Format: x y z vx vy vz M Nsat Dvx Dvy Dvz Dx Dy Dz halo_id is_central
            f_out.write(f"{galaxy[0]:.5f} {galaxy[1]:.5f} {galaxy[2]:.5f} "
                       f"{galaxy[3]:.5f} {galaxy[4]:.5f} {galaxy[5]:.5f} "
                       f"{galaxy[6]:.6e} {int(galaxy[7])} "
                       f"{galaxy[8]:.4f} {galaxy[9]:.4f} {galaxy[10]:.4f} "
                       f"{galaxy[11]:.4f} {galaxy[12]:.4f} {galaxy[13]:.4f} "
                       f"{int(galaxy[14])} {int(galaxy[15])}\n")
        else:
            # Write only essential columns (matching C code #else format)
            # Format: x y z vx vy vz M Nsat
            f_out.write(f"{galaxy[0]:.5f} {galaxy[1]:.5f} {galaxy[2]:.5f} "
                       f"{galaxy[3]:.5f} {galaxy[4]:.5f} {galaxy[5]:.5f} "
                       f"{galaxy[6]:.6e} {int(galaxy[7])}\n")

import h5py

def read_occupation_from_h5(h5file):
    """
    Reads the mass bins and mean occupation numbers from the HDF5 file.

    Parameters:
    -----------
    h5file : str
        Path to your h2s_output.h5 file.

    Returns:
    --------
    M_min : ndarray
        Lower limits of mass bins.
    M_max : ndarray
        Upper limits of mass bins.
    Ncen_mean : ndarray
        Mean number of centrals per halo per bin.
    Nsat_mean : ndarray
        Mean number of satellites per halo per bin.
    """
    with h5py.File(h5file, 'r') as f:
        data = f['data']
        M_min = np.array(data['M_min'])
        M_max = np.array(data['M_max'])
        Ncen = np.array(data['Ncen'])
        Nsat = np.array(data['Nsat'])
        N_halo = np.array(data['N_halo'])  

    # Calcula las medias, evitando divisiones por cero
    Ncen_mean = np.zeros_like(Ncen, dtype=float)
    Nsat_mean = np.zeros_like(Nsat, dtype=float)
    Ncen_mean = Ncen / N_halo
    Nsat_mean = Nsat / N_halo

    return M_min, M_max, Ncen_mean, Nsat_mean

def read_global_conformity_factors(h5file):
    """
    Reads K1_global y K2_global from the group'header' in  h5file.
    Returns: (K1_global, K2_global) as floats
    """
    with h5py.File(h5file, 'r') as f:
        header = f['header']
        K1_global = float(header.attrs['K1_global'])
        K2_global = float(header.attrs['K2_global'])
    return K1_global, K2_global
