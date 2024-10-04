#%%

# Importing the required libraries

import sionna as sn

import numpy as np
import tensorflow as tf
# For the implementation of the Keras models
from tensorflow.keras import Model



# also try %matplotlib widget
import matplotlib.pyplot as plt

# for performance measurements
import time

# Importing the required classes from the sionna library
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.utils import BinarySource, ebnodb2no
from sionna.channel import AWGN
from sionna.utils.metrics import compute_ber
from sionna.signal import Upsampling, Downsampling, RootRaisedCosineFilter, empirical_psd, empirical_aclr

# Allow memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
    
print("The required libraries are imported successfully!")
print('sionna version:', sn.__version__)


#%%
def generate_ue_resource_grid(simulation_params, batch_size):
    """
    Create the UE resource grids to be transmitted and the indices of the replicas for one UE.

    Args:
        simulation_params: Dictionary containing all simulation parameters.
        batch_size: The batch size for the simulation.

    Returns:
        symbols_rg: The UE resource grids to be transmitted.
        bits: The transmitted bits.
    """
    # Params extraction
    # Extract necessary parameters from the simulation_params dictionary
    carrier_params = simulation_params["Carrier parameters"]
    num_resource_blocks = carrier_params['num_resource_blocks']
    numerology = carrier_params['numerology']
    pilot_indices = carrier_params['pilot_indices']
    num_ofdm_symbols = carrier_params['num_ofdm_symbols']

    transport_block_params = simulation_params["Transport block parameters"]
    num_bits_per_symbol = transport_block_params['num_bits_per_symbol']
    coderate = transport_block_params['coderate']

    # Object creations
    # Create the needed objects for transmission
    
    # Create the resource grid object
    resource_grid_config = sn.ofdm.ResourceGrid(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=12 * num_resource_blocks,
        subcarrier_spacing=1e3 * (15 * 2 ** numerology),
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=pilot_indices
    )

    # Create other needed objects
    
    # Binary source: Generates random binary sequences
    binary_source = sn.utils.BinarySource()
    
    # Calculate the number of coded bits (n) and information bits (k) in a resource grid
    n = int(resource_grid_config.num_data_symbols * num_bits_per_symbol)
    k = int(n * coderate)
    
    # LDPC Encoder: Encodes the information bits into coded bits
    encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)
    
    # QAM Mapper: Maps blocks of information bits to constellation symbols
    mapper = sn.mapping.Mapper("qam", num_bits_per_symbol)
    
    # Resource Grid Mapper: Maps symbols onto an OFDM resource grid
    resource_grid_mapper = sn.ofdm.ResourceGridMapper(resource_grid_config)
    
    # Transmission
    # Generate random binary bits for the UE
    bits = binary_source([batch_size, k])
    
    # Encode the bits using LDPC encoder
    codewords = encoder(bits)
    
    # Map the encoded bits to QAM symbols
    symbols = mapper(codewords)
    
    # Map the QAM symbols onto the OFDM resource grid
    resource_grid = resource_grid_mapper(tf.expand_dims(tf.expand_dims(symbols, axis=1), axis=1))
    
    # Remove the extra dimensions added during mapping
    resource_grid = tf.squeeze(tf.squeeze(resource_grid, axis=1), axis=1)
    
    return resource_grid, bits

def generate_resource_grid(simulation_params, batch_size, num_ues):
    """
    Generate resource grids for multiple UEs with interference and optional phase shift.

    Args:
        simulation_params: Dictionary containing all simulation parameters.
        batch_size: The batch size for the simulation.
        num_ues: The number of UEs in the simulation.

    Returns:
        interfered_resource_grids: The interfered resource grids.
        bits_list: List of transmitted bits for each UE.
        resource_grids_list: List of original resource grids for each UE.
        channels: List of dictionaries containing channel angles and coefficients for each UE.
    """
    # Extract necessary parameters from the simulation_params dictionary
    channel_params = simulation_params["Channel parameters"]
    is_phase_shift_applied = channel_params['is_phase_shift_applied']

    # Initialize lists to store resource grids, bits, and channels for each UE
    faded_resource_grid_list = []
    resource_grid_list = []
    bits_list = []
    channels = []

    # Generate resource grids, bits, and channels for each UE
    for _ in range(num_ues):
        resource_grid, bits = generate_ue_resource_grid(simulation_params, batch_size)
        
        # Apply phase shift if specified
        if is_phase_shift_applied:
            # Step 1: Create the angle
            angles = 2 * np.pi * tf.random.uniform(shape=[batch_size], minval=0, maxval=1)
            # Expand the angles to match the resource grid shape
            angles = tf.expand_dims(tf.expand_dims(angles, axis=1), axis=1) 
            # Step 2: Calculate the channel coefficients
            channel_coeff = tf.complex(tf.cos(angles), tf.sin(angles))
            resource_grid = resource_grid * channel_coeff
            
        else:
            angles = tf.zeros_like(resource_grid)
            channel_coeff = tf.ones_like(resource_grid)
        
        channels.append({'angle': angles, 'coeff': channel_coeff})

        # Append the resource grid and bits to the lists
        resource_grid_list.append(resource_grid)
        bits_list.append(bits)
        
        # Apply the channel coeffient
        faded_resource_grid_list.append(resource_grid * channel_coeff)
        

    # Sum the resource grids to create the interfered resource grid
    resource_grids = tf.reduce_sum(tf.stack(faded_resource_grid_list), axis=0)

    return resource_grids, bits_list, resource_grid_list, channels


def pass_through_awgn(resource_grids, ebno_db, simulation_params):
    """
    Pass the interfered resource grids through an AWGN channel.

    Args:
        interfered_resource_grids: The interfered resource grids to be transmitted.
        ebno_db: The Eb/No value in dB.
        simulation_params: Dictionary containing all simulation parameters.

    Returns:
        y_resource_grids: The received signal after passing through the AWGN channel.
        no: The noise variance.
    """
    # Extract necessary parameters from the simulation_params dictionary
    transport_block_params = simulation_params["Transport block parameters"]
    num_bits_per_symbol = transport_block_params['num_bits_per_symbol']
    coderate = transport_block_params['coderate']

    # Create an instance of the AWGN channel
    awgn_channel = AWGN()

    # Calculate the noise variance
    no = ebnodb2no(ebno_db, num_bits_per_symbol=num_bits_per_symbol, coderate=coderate)

    # Pass the interfered resource grids through the AWGN channel
    y_resource_grids = awgn_channel([resource_grids, no])

    return y_resource_grids, no


def remove_interference(y_resource_grids, resource_grid_list, channels, is_perfect_CSI):
    """
    Remove interference from the received resource grids using estimated or perfect channel coefficients.

    Args:
        y_resource_grids: The received resource grids with interference.
        resource_grid_list: List of original resource grids for each UE.
        channels: List of dictionaries containing channel angles and coefficients for each UE.
        is_perfect_CSI: Boolean indicating whether to use perfect CSI (True) or estimated CSI (False).

    Returns:
        y_resource_grids_cleaned: The received resource grids with interference removed.
    """
    # Initialize the cleaned resource grids with the received resource grids
    y_resource_grids_cleaned = y_resource_grids

    # Iterate over all UEs except the first one
    for i in range(1, len(resource_grid_list)):
        # Extract the resource grid for the current UE
        resource_grid = resource_grid_list[i]

        if is_perfect_CSI:
            # Use perfect knowledge of the channel coefficients
            h_hat = channels[i]['coeff']
        else:
            # Estimate the phase shift (phi_hat) using a matched filter
            correlation = tf.reduce_sum(tf.math.conj(resource_grid) * y_resource_grids_cleaned, axis=[1, 2])
            phi_hat = tf.math.angle(correlation)
            h_hat = tf.complex(tf.cos(phi_hat), tf.sin(phi_hat))
            # Expand the channel coefficients to match the resource grid shape
            h_hat = tf.expand_dims(tf.expand_dims(h_hat, axis=1), axis=1)

        # Remove the interference caused by the current UE
        y_resource_grids_cleaned -= resource_grid * h_hat

    return y_resource_grids_cleaned


def decode_slot(received_rg, no, simulation_params, batch_size):
    """
    Decode a single slot and output the estimated bits and channel coefficients.

    Args:
        received_rg: The received resource grid for the slot.
        no: The noise variance.
        simulation_params: Dictionary containing all simulation parameters.

    Returns:
        bits_hat: The estimated bits.
        h_hat: The estimated channel coefficients.
    """
    # Extract necessary parameters from the simulation_params dictionary
    carrier_params = simulation_params["Carrier parameters"]
    numerology = carrier_params['numerology']
    num_resource_blocks = carrier_params['num_resource_blocks']
    num_ofdm_symbols = carrier_params['num_ofdm_symbols']
    pilot_indices = carrier_params['pilot_indices']
    data_indices = np.setdiff1d(np.arange(num_ofdm_symbols), pilot_indices)

    transport_block_params = simulation_params["Transport block parameters"]
    num_bits_per_symbol = transport_block_params['num_bits_per_symbol']
    coderate = transport_block_params['coderate']

    # Create instances of needed objects
    resource_grid_config = sn.ofdm.ResourceGrid(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=12 * num_resource_blocks,
        subcarrier_spacing=1e3 * (15 * 2 ** numerology),
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=pilot_indices
    )
    demapper = sn.mapping.Demapper("app", "qam", num_bits_per_symbol)
    
    # Calculate the number of coded bits (n) and information bits (k) in a resource grid
    n = int(resource_grid_config.num_data_symbols * num_bits_per_symbol)
    k = int(n * coderate)
    
    # LDPC Encoder: Encodes the information bits into coded bits
    encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)
    
    # LDPC Decoder: Decodes the coded bits back into information bits
    decoder = sn.fec.ldpc.LDPC5GDecoder(encoder, hard_out=True)

    # Least Squares Channel Estimator: Estimates the channel coefficients
    ls_est = sn.ofdm.LSChannelEstimator(resource_grid_config, interpolation_type="nn")
    
    # Start the decoding process
    
    # Estimate the channel coefficients (h_hat) and error variance (err_var)
    h_hat, err_var = ls_est([tf.expand_dims(tf.expand_dims(received_rg, axis=1), axis=1), no]) # Add two dimensions to match the input shape of the LSChannelEstimator
    h_hat = tf.squeeze(h_hat, axis=[1,2,3,4])  # Remove the added dimensions
    
    # Equalize the received resource grid using the estimated channel coefficients
    received_rg_equalized = tf.math.divide_no_nan(received_rg, h_hat)
    
    # Extract the received symbols from the equalized resource grid
    received_symbols = tf.gather(received_rg_equalized, data_indices, axis=1)
    received_symbols= tf.reshape(received_symbols, (batch_size, -1))
    
    # Demap the received symbols to log-likelihood ratios (LLRs)
    llr = demapper([received_symbols, no])
    
    # Decode the LLRs to estimate the transmitted bits
    bits_hat = decoder(llr)
    
    return bits_hat, h_hat

def run_simulation(simulation_params, ebno_db, batch_size, num_ues):
    """
    Run the simulation with the given parameters and Eb/No value.

    Args:
        simulation_params: Dictionary containing all simulation parameters.
        ebno_db: The Eb/No value in dB.
        batch_size: The batch size for the simulation.
        num_ues: The number of UEs in the simulation.

    Returns:
        ber: Bit Error Rate for the first UE.
    """
    # Extract the is_perfect_CSI parameter
    is_perfect_CSI = simulation_params["SIC"]["is_perfect_CSI"]
    
    # Generate the resource grids for multiple UEs
    resource_grids, bits_list, resource_grid_list, channels = generate_resource_grid(simulation_params, batch_size, num_ues)
    
    # Pass the resource grids through the AWGN channel
    received_rg, no = pass_through_awgn(resource_grids, ebno_db, simulation_params)
    
    # Remove interference from the received resource grids
    cleaned_rg = remove_interference(received_rg, resource_grid_list, channels, is_perfect_CSI)
    
    # Decode the cleaned resource grid for the first UE
    bits_hat, _ = decode_slot(cleaned_rg, no, simulation_params, batch_size)
    
    # Compute the Bit Error Rate (BER) for the first UE
    ber = compute_ber(bits_list[0], bits_hat)
    
    return ber

#%%

# Example usage
simulation_params = {
    "Carrier parameters": {
        "num_resource_blocks": 6,
        "numerology": 0,
        "pilot_indices": [0, 1, 2],
        "num_ofdm_symbols": 14
    },
    "Transport block parameters": {
        "num_bits_per_symbol": 2,
        "coderate": 0.5
    },
    "SIC": {
        "is_perfect_CSI": False
    },
    "Channel parameters": {
        "is_phase_shift_applied": True
    }
}

ebno_db = 100
batch_size = 1000
num_ues = 10
ber = run_simulation(simulation_params, ebno_db, batch_size, num_ues)

# Print the BER
print("\n\n")
print("=====================================================")
print("Simulation Report:")
print(f"Eb/No (dB): {ebno_db}")
print(f"Bit Error Rate (BER) for the first UE: {ber}")
print("=====================================================")

# %%
# Define a range of Eb/No values in dB
# ebno_db_values = np.arange(-5, 21, 2)
ebno_db_values = np.arange(100, 100, 1)
# Define different numbers of UEs to simulate
num_ues_values = [1, 2, 4, 6]

# Initialize a dictionary to store the BER values for each number of UEs
ber_values_dict = {}

# Run the simulation for each number of UEs and each Eb/No value
for num_ues in num_ues_values:
    ber_values = []
    for ebno_db in ebno_db_values:
        ber = run_simulation(simulation_params, ebno_db, batch_size, num_ues)
        ber_values.append(ber)
        print(f"Num UEs: {num_ues}, Eb/No: {ebno_db} dB, BER: {ber}")
    ber_values_dict[num_ues] = ber_values

# Plot the BER vs Eb/No for different numbers of UEs
plt.figure()
for num_ues, ber_values in ber_values_dict.items():
    plt.semilogy(ebno_db_values, ber_values, marker='o', label=f'{num_ues} UEs')
plt.xlabel('Eb/No (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs Eb/No for Different Numbers of UEs')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()