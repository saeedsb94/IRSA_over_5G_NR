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
from sionna.utils.metrics import compute_ber, compute_bler
from sionna.signal import Upsampling, Downsampling, RootRaisedCosineFilter, empirical_psd, empirical_aclr
import os

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
        simulation_params (dict): Dictionary containing all simulation parameters.
        batch_size (int): The batch size for the simulation.

    Returns:
        tf.Tensor: The UE resource grids to be transmitted.
        tf.Tensor: The transmitted bits.
    """
    # Extract necessary parameters from the simulation_params dictionary
    carrier_params = simulation_params["Carrier parameters"]
    num_resource_blocks = carrier_params['num_resource_blocks']
    numerology = carrier_params['numerology']
    pilot_indices = carrier_params['pilot_indices']
    num_ofdm_symbols = carrier_params['num_ofdm_symbols']

    transport_block_params = simulation_params["Transport block parameters"]
    num_bits_per_symbol = transport_block_params['num_bits_per_symbol']
    coderate = transport_block_params['coderate']

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
        simulation_params (dict): Dictionary containing all simulation parameters.
        batch_size (int): The batch size for the simulation.
        num_ues (int): The number of UEs in the simulation.

    Returns:
        tf.Tensor: The interfered resource grids.
        list: List of transmitted bits for each UE.
        list: List of original resource grids for each UE.
        list: List of dictionaries containing channel angles and coefficients for each UE.
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
            
            
        else:
            angles = tf.zeros_like(resource_grid)
            channel_coeff = tf.ones_like(resource_grid)
        
        channels.append({'angle': angles, 'coeff': channel_coeff})
        

        # Append the resource grid and bits to the lists
        resource_grid_list.append(resource_grid)
        bits_list.append(bits)
        
        # Apply the channel coefficient
        faded_resource_grid_list.append(resource_grid * channel_coeff)
        

    # Sum the resource grids to create the interfered resource grid
    resource_grids = tf.reduce_sum(tf.stack(faded_resource_grid_list), axis=0)

    return resource_grids, bits_list, resource_grid_list, channels


def pass_through_awgn(resource_grids, ebno_db, simulation_params):
    """
    Pass the interfered resource grids through an AWGN channel.

    Args:
        resource_grids (tf.Tensor): The interfered resource grids to be transmitted.
        ebno_db (float): The Eb/No value in dB.
        simulation_params (dict): Dictionary containing all simulation parameters.

    Returns:
        tf.Tensor: The received signal after passing through the AWGN channel.
        float: The noise variance.
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
        y_resource_grids (tf.Tensor): The received resource grids with interference.
        resource_grid_list (list): List of original resource grids for each UE.
        channels (list): List of dictionaries containing channel angles and coefficients for each UE.
        is_perfect_CSI (bool): Boolean indicating whether to use perfect CSI (True) or estimated CSI (False).

    Returns:
        tf.Tensor: The received resource grids with interference removed.
        tf.Tensor: The energy of the interference noise after removal.
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
    
    # Extract the resource grid for the first UE
    resource_grid_ue1 = resource_grid_list[0]
    # Get the perfect channel of the first UE
    h_ue1 = channels[0]['coeff']
    
    y_ue1 = resource_grid_ue1 * h_ue1
    
    # Calculate the size of the resource grid (excluding the batch dimension)
    resource_grid_size = resource_grid_ue1.shape[1]*resource_grid_ue1.shape[2]
    
    interference_noise_energy = tf.reduce_sum(tf.abs(y_resource_grids_cleaned - y_ue1) ** 2, axis=[1, 2])/resource_grid_size

    return y_resource_grids_cleaned, interference_noise_energy

def decode_slot(received_rg, no, simulation_params, batch_size):
    """
    Decode a single slot and output the estimated bits and channel coefficients.

    Args:
        received_rg (tf.Tensor): The received resource grid for the slot.
        no (float): The noise variance.
        simulation_params (dict): Dictionary containing all simulation parameters.

    Returns:
        tf.Tensor: The estimated bits.
        tf.Tensor: The estimated channel coefficients.
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

# @tf.function() # Enable graph execution to speed things up
def run_simulation(simulation_params, ebno_db, batch_size, num_ues):
    """
    Run the simulation with the given parameters and Eb/No value.

    Args:
        simulation_params (dict): Dictionary containing all simulation parameters.
        ebno_db (float): The Eb/No value in dB.
        batch_size (int): The batch size for the simulation.
        num_ues (int): The number of UEs in the simulation.

    Returns:
        dict: Dictionary containing BER, BLER, and average residual interference energy.
    """
    # Extract the is_perfect_CSI parameter
    is_perfect_CSI = simulation_params["SIC"]["is_perfect_CSI"]
    
    # Generate the resource grids for multiple UEs
    resource_grids, bits_list, resource_grid_list, channels = generate_resource_grid(simulation_params, batch_size, num_ues)
    
    # Pass the resource grids through the AWGN channel
    received_rg, no = pass_through_awgn(resource_grids, ebno_db, simulation_params)
    
    # Remove interference from the received resource grids
    cleaned_rg, residual_interference = remove_interference(received_rg, resource_grid_list, channels, is_perfect_CSI)
    
    # Decode the cleaned resource grid for the first UE
    bits_hat, _ = decode_slot(cleaned_rg, no, simulation_params, batch_size)
    
    # Compute the average residual interference energy
    avg_residual_interference = tf.reduce_mean(residual_interference)
    
    # Compute the Bit Error Rate (BER) for the first UE
    ber = compute_ber(bits_list[0], bits_hat)
    
    # Compute the Block Error Rate (BLER) for the first UE
    bler = compute_bler(bits_list[0], bits_hat)
    
    return {
        "ber": ber,
        "bler": bler,
        "avg_residual_interference": avg_residual_interference
    }

#%%
# Example usage
simulation_params = {
    "Carrier parameters": {
        "num_resource_blocks": 10,
        "numerology": 0,
        "pilot_indices": [2,8],
        "num_ofdm_symbols": 14
    },
    "Transport block parameters": {
        "num_bits_per_symbol": 4,
        "coderate": 0.5
    },
    "SIC": {
        "is_perfect_CSI": False
    },
    "Channel parameters": {
        "is_phase_shift_applied": True
    }
}

#%%
# Run the simulation for a specific Eb/No value, batch size, and number of UEs
ebno_db = 10
batch_size = 100
num_ues = 10
results = run_simulation(simulation_params, ebno_db, batch_size, num_ues)

# Print the BER, BLER, and average residual interference energy
print("\n\n")
print("=====================================================")
print("Simulation Report:")
print(f"Eb/No (dB): {ebno_db}")
print(f"Bit Error Rate (BER) for the first UE: {results['ber']}")
print(f"Block Error Rate (BLER) for the first UE: {results['bler']}")
print(f"Average Residual Interference Energy: {results['avg_residual_interference']}")
print("=====================================================")

# %%
# Plot the BER, BLER, and residual interference energy vs number of UEs for high Eb/No
high_ebno = 100  # Define a high Eb/No value
batch_size = 10000
num_ues_values = np.arange(1, 15, dtype=int)

ber_values_high_ebno = []
bler_values_high_ebno = []
residual_interference_values_high_ebno = []
success_rate_values_high_ebno = []

# Run the simulation for each number of UEs at the high Eb/No value
for num_ues in num_ues_values:
    results = run_simulation(simulation_params, high_ebno, batch_size, num_ues)
    ber_values_high_ebno.append(results['ber'])
    bler_values_high_ebno.append(results['bler'])
    residual_interference_values_high_ebno.append(results['avg_residual_interference'])
    success_rate = 1 - results['bler']  # Calculate success rate as 1 - BLER
    success_rate_values_high_ebno.append(success_rate)
    print(f"Num UEs: {num_ues}, Eb/No: {high_ebno} dB, BER: {results['ber']}, BLER: {results['bler']}, Residual Interference: {results['avg_residual_interference']}, Success Rate: {success_rate}")

# Plot the BER vs number of UEs
plt.figure()
plt.plot(num_ues_values, ber_values_high_ebno, marker='o')
plt.xlabel('Number of UEs')
plt.ylabel('Bit Error Rate (BER)')
plt.title(f'BER vs Number of UEs at Eb/No = {high_ebno} dB')
plt.xticks(num_ues_values)  # Ensure integer ticks for number of UEs
plt.grid(True)
# Create a directory to save the figures if it doesn't exist
output_dir = 'simulation_results'
os.makedirs(output_dir, exist_ok=True)

# Plot the BER vs number of UEs
plt.figure()
plt.plot(num_ues_values, ber_values_high_ebno, marker='o')
plt.xlabel('Number of UEs')
plt.ylabel('Bit Error Rate (BER)')
plt.title(f'BER vs Number of UEs at Eb/No = {high_ebno} dB')
plt.xticks(num_ues_values)  # Ensure integer ticks for number of UEs
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'ber_vs_num_ues_high_ebno.png'))
plt.show()

# Plot the BLER vs number of UEs
plt.figure()
plt.plot(num_ues_values, bler_values_high_ebno, marker='o')
plt.xlabel('Number of UEs')
plt.ylabel('Block Error Rate (BLER)')
plt.title(f'BLER vs Number of UEs at Eb/No = {high_ebno} dB')
plt.xticks(num_ues_values)  # Ensure integer ticks for number of UEs
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'bler_vs_num_ues_high_ebno.png'))
plt.show()

# Plot the residual interference energy vs number of UEs
plt.figure()
plt.plot(num_ues_values, residual_interference_values_high_ebno, marker='o')
plt.xlabel('Number of UEs')
plt.ylabel('Residual Interference Energy')
plt.title(f'Residual Interference Energy vs Number of UEs at Eb/No = {high_ebno} dB')
plt.xticks(num_ues_values)  # Ensure integer ticks for number of UEs
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'residual_interference_vs_num_ues_high_ebno.png'))
plt.show()

# Plot the success rate vs number of UEs
plt.figure()
plt.plot(num_ues_values, success_rate_values_high_ebno, marker='o')
plt.xlabel('Number of UEs')
plt.ylabel('Success Rate')
plt.title(f'Success Rate vs Number of UEs at Eb/No = {high_ebno} dB')
plt.xticks(num_ues_values)  # Ensure integer ticks for number of UEs
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'success_rate_vs_num_ues_high_ebno.png'))
plt.show()

# %%
