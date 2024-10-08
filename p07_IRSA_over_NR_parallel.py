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

def generate_ues(simulation_params, num_ues_per_frame, num_simulations):
    """
    Create the UE resource grid to be transmitted and the indices of the replicas for one UE.

    Args:
        simulation_params: Dictionary containing all simulation parameters.
        num_ues_per_frame: Number of UEs in the frame.
        num_simulations: Number of simulations to run.

    Returns:
        resource_grids: List of resource grids for each UE.
        bits_list: List of transmitted bits for each UE.
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
    num_ues =num_ues_per_frame* num_simulations
    bits = binary_source([num_ues, k])
    
    # Encode the bits using LDPC encoder
    codewords = encoder(bits)
    
    # Map the encoded bits to QAM symbols
    symbols = mapper(codewords)
    
    # Map the QAM symbols onto the OFDM resource grid
    resource_grid = resource_grid_mapper(tf.expand_dims(tf.expand_dims(symbols, axis=1), axis=1))
    
    # Remove the extra dimensions added during mapping
    resource_grid = tf.squeeze(tf.squeeze(resource_grid, axis=1), axis=1)
    
    return resource_grid, bits

def generate_slot_indices(num_simulations, num_ues_per_frame, frame_size, probabilities):
    """
    Generate slot indices for each UE based on the given probabilities.

    Args:
        simulation_num: Number of simulations to run.
        num_ues_per_frame: Number of UEs per frame.
        frame_size: Total number of slots in the frame.
        probabilities: Probabilities for selecting number of replicas.

    Returns:
        slot_indices_tensor: Tensor of shape (batch_size, ) where batch_size = simulation_num * num_ues_per_frame.
                                Each element is a list of the replicas.
        replica_counts: List of the number of replicas for each UE.
    """

    num_ues = num_simulations * num_ues_per_frame
    
    slot_indices_list = []

    # Step 1: Randomly select the number of replicas for each UE based on the given probabilities
    replica_counts = np.random.choice(np.arange(len(probabilities)), size=num_ues, p=probabilities)

    # Step 2: For each UE, randomly select slot indices based on the selected number of replicas
    for i in range(num_simulations):
        for j in range(num_ues_per_frame):
            num_replicas = replica_counts[i * num_ues_per_frame + j]
            slot_indices = np.random.choice(np.arange(i * frame_size, (i + 1) * frame_size), size=num_replicas, replace=False)
            slot_indices_list.append(slot_indices)
    

    return slot_indices_list, replica_counts

def generate_channel(simulation_params, num_simulations, num_ues_per_frame, num_replicas, is_phase_shift_applied):
    """
    Generate the same channel coefficients for all resource elements in the resource grid for each UE.
    
    Args:
        simulation_params: Dictionary containing all simulation parameters.
        num_ues_per_frame: Number of UEs per frame.
        num_simulations: Number of simulations to run.
        num_replicas: Array of number of replicas for each UE.
        is_phase_shift_applied: Boolean indicating whether phase shift should be applied.
        
    Returns:
        angles: The angles used for phase shift as a tf.RaggedTensor.
        channel_coeff: The channel coefficients as a tf.RaggedTensor, where each element is a 3D tensor.
    """
    carrier_params = simulation_params["Carrier parameters"]
    num_ofdm_symbols = carrier_params['num_ofdm_symbols']  # Number of OFDM symbols in a frame
    num_resource_blocks = carrier_params['num_resource_blocks']  # Number of resource blocks
    fft_size = 12 * num_resource_blocks  # Total number of subcarriers (assuming 12 subcarriers per resource block)

    num_ues = num_ues_per_frame * num_simulations
    
    angles_list = []
    channel_coeff_list = []
    
    for ue_idx in range(num_ues):
        # Number of replicas for the current UE
        ue_num_replicas = num_replicas[ue_idx]
        
        if is_phase_shift_applied:
            # Step 1: Create a set of angles for the current UE
            angles = 2 * np.pi * tf.random.uniform(shape=[ue_num_replicas], minval=0, maxval=1, dtype=tf.float32)
            
            # Step 2: Calculate the channel coefficient for the current UE
            channel_coeff_single = tf.complex(tf.cos(angles), tf.sin(angles))
        else:
            # No phase shift, return zeros for angles and ones for channel coefficients
            angles = tf.zeros([ue_num_replicas], dtype=tf.float32)
            channel_coeff_single = tf.ones([ue_num_replicas], dtype=tf.complex64)

        # Step 3: Replicate the channel coefficient across all resource elements (OFDM symbols × subcarriers)
        channel_coeff = tf.tile(tf.expand_dims(tf.expand_dims(channel_coeff_single, -1), -1), 
                                multiples=[1, num_ofdm_symbols, fft_size])

        # Append the results to the lists
        angles_list.append(angles)
        channel_coeff_list.append(channel_coeff)

    return angles_list, channel_coeff_list


def generate_hyper_irsa_frame(simulation_params, num_simulations, num_ues_per_frame, frame_size, probabilities):
    """
    Generate an IRSA frame based on the given simulation parameters.

    Args:
        simulation_params: Dictionary containing all simulation parameters.
        num_simulations: Number of simulations to run.
        num_ues_per_frame: Number of UEs per frame.
        frame_size: Total number of slots in the frame.
        probabilities: Probabilities for selecting 1, 2, 3, or 4 replicas.

    Returns:
        irsa_frame: The generated IRSA frame.
        resource_grid_list: List of resource grids for each UE.
        h_ues: Channel coefficients for each UE in each slot.
        replicas_indices_list: List of replica indices for each UE.
        original_bits_list: List of original bits for each UE.
    """
    # Extract necessary parameters from the simulation_params dictionary
    carrier_params = simulation_params["Carrier parameters"]
    num_ofdm_symbols = carrier_params['num_ofdm_symbols']
    num_resource_blocks = carrier_params['num_resource_blocks']
    fft_size = 12 * num_resource_blocks

    channel_params = simulation_params["Channel parameters"]
    is_phase_shift_applied = channel_params['is_phase_shift_applied']
    
    # Create the output frame tensor
    batch_size = num_simulations * frame_size
    irsa_hyper_frame = tf.zeros([batch_size, num_ofdm_symbols, fft_size], dtype=tf.complex64)
    
    # Lists to store the resource grids, replica indices, and original bits for each UE
    resource_grids, bits = generate_ues(simulation_params, num_ues_per_frame, num_simulations)
    
    # Generate slot indices based on the given probabilities
    slot_indices, num_replicas = generate_slot_indices(num_simulations, num_ues_per_frame, frame_size, probabilities)
    
    # Generate the channel coefficients for each UE
    angles, channel_coeff = generate_channel(simulation_params, num_simulations, num_ues_per_frame, num_replicas, is_phase_shift_applied)
    

    
    # Create an empty tensor to store the hyper IRSA frame
    irsa_hyper_frame = tf.zeros([batch_size, num_ofdm_symbols, fft_size], dtype=tf.complex64)
    
    
    num_ues=num_ues_per_frame*num_simulations
    
    # Process each UE
    for ue_index in range(num_ues):
        # Create a temporary hyper IRSA frame for the current UE
        ue_hyper_irsa_frame = tf.zeros_like(irsa_hyper_frame)
        
        # Get the resource grid and channel coefficient for the current UE
        resource_grid = resource_grids[ue_index]
        channel_coeff_ue = channel_coeff[ue_index]
        
        # Get the slot indices for the current UE
        slot_indices_ue = slot_indices[ue_index]
        num_replicas_ue = num_replicas[ue_index]
        
        # Allocate the replicas of the current UE in the temporary hyper IRSA frame
        for replica_index in range(num_replicas_ue):
            replica_position = slot_indices_ue[replica_index]
            resource_grid_with_channel = resource_grid * channel_coeff_ue[replica_index]
            ue_hyper_irsa_frame = tf.tensor_scatter_nd_update(ue_hyper_irsa_frame, [[replica_position]], tf.expand_dims(resource_grid_with_channel, axis=0))
        
        # Add the temporary hyper IRSA frame to the output frame
        irsa_hyper_frame += ue_hyper_irsa_frame
        
    return irsa_hyper_frame, resource_grids, channel_coeff, slot_indices, bits
                
def pass_through_awgn(irsa_frame, ebno_db, simulation_params):
    """
    Pass an IRSA frame through an AWGN channel.

    Args:
        irsa_frame: The IRSA frame to be transmitted.
        ebno_db: The Eb/No value in dB.
        simulation_params: Dictionary containing all simulation parameters.

    Returns:
        y_combined: The received signal after passing through the AWGN channel.
        no: The noise variance.
    """
    # Extract necessary parameters from the simulation_params dictionary
    transport_block_params = simulation_params["Transport block parameters"]
    num_bits_per_symbol = transport_block_params['num_bits_per_symbol']
    coderate = transport_block_params['coderate']

    # Create an instance of the AWGN channel
    awgn_channel = sn.channel.AWGN()

    # Calculate the noise variance
    no = sn.utils.ebnodb2no(ebno_db, num_bits_per_symbol=num_bits_per_symbol, coderate=coderate)

    # Pass the IRSA frame through the AWGN channel
    received_frame = awgn_channel([irsa_frame, no])

    return received_frame, no

def decode_frame(received_rg, no, num_simulations, frame_size, simulation_params):
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

    batch_size = num_simulations * frame_size
    
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

    ls_est = sn.ofdm.LSChannelEstimator(resource_grid_config, interpolation_type="nn")
    
    
    # Perform the decoding process
      # Estimate the channel coefficients (h_hat) and error variance (err_var)
    h_hat, err_var = ls_est([tf.expand_dims(tf.expand_dims(received_rg, axis=1), axis=1), no]) # Add two dimensions to match the input shape of the LSChannelEstimator
    h_hat = tf.squeeze(h_hat, axis=[1,2,3,4])  # Remove the added dimensions
    
    # Equalize the received resource grid using the estimated channel coefficients
    received_rg_equalized = tf.math.divide_no_nan(received_rg, h_hat)

    # Extract the data symbols from the equalized resource grid
    received_data_rg = tf.gather(received_rg_equalized, data_indices, axis=1)
    # Flatten the received symbols to match the input shape of the demapper
    received_symbols= tf.reshape(received_data_rg, (batch_size, -1))

    # Demap the received symbols to log-likelihood ratios (LLRs)
    llr = demapper([received_symbols, no])
    
    # Decode the LLRs to estimate the transmitted bits
    bits_hat = decoder(llr)
    
    return bits_hat, h_hat

def search_new_identified_ues(bits_hat, bits, slot_indices, identified_ues):
    """
    Search for the identified UEs inside bits_hat.

    Args:
        bits_hat: The estimated bits.
        bits: List of original bits for each UE.
        slot_indices: List of slot indices for each UE.
        identified_ues: Set of indices of already identified UEs.

    Returns:
        new_identified_ues: Set of indices of newly identified UEs.
        identified_positions: Set of positions where UEs were identified.
        new_ues_found: Boolean flag indicating if new UEs were found.
    """
    num_ues = len(bits)
    new_identified_ues = set()
    new_identified_positions = set()
    new_ues_found = False
    
    for i in range(num_ues):
        if i not in identified_ues:
            # Get the position of the bits in the bits_hat tensor
            ue_slot_indices = slot_indices[i]
            for slot_index in ue_slot_indices:
                is_match = tf.reduce_all(tf.equal(bits_hat[slot_index], bits[i]))
                if is_match:
                    new_identified_ues.add(i)
                    new_identified_positions.add(slot_index)
                    new_ues_found = True
                
    return new_identified_ues, new_identified_positions, new_ues_found

def remove_replicas_of_newlly_identified_ues(y_resource_grids, resourse_grids, slot_indices, new_identified_ues, is_perfect_SIC, channels):
    """
    Remove the replicas of a UE when it is recognized.

    Args:
        received_frame: The received IRSA frame.
        resource_grid_list: List of resource grids for each UE.
        replicas_indices_list: List of replica indices for each UE.
        identified_ues: List of indices of recognized UEs.

    Returns:
        received_frame: The updated received frame with replicas removed.
    """
    y_resource_grids_cleaned = y_resource_grids

    for ue in new_identified_ues:
        # Extract the resource grid of the current ue
        ue_rg=resourse_grids[ue]
        for pos in slot_indices[ue]:
            y_replica_slot = y_resource_grids_cleaned[pos]
            
            if is_perfect_SIC:
                # Extract the Channel coefficient of the replica
                h_hat_replicas = channels[i]['coeff']
            else:
                # Estimate the channel coefficient of the replica
                phi_hat = tf.math.angle(tf.math.reduce_sum(y_replica_slot * tf.math.conj(ue_rg)))
                h_hat_replicas = tf.complex(tf.cos(phi_hat), tf.sin(phi_hat))
            
            
            clean_slot = y_replica_slot - ue_rg * h_hat_replicas
            y_resource_grids_cleaned = tf.tensor_scatter_nd_update(y_resource_grids_cleaned, [[pos]], tf.expand_dims(clean_slot, axis=0))

    return y_resource_grids_cleaned

def decode_irsa_frame(y_resource_grids, no, simulation_params, resourse_grids, bits, slot_indices, num_simulations, frame_size, num_ues_per_frame): 
    """
    Decode an IRSA frame.

    Args:
        y_resource_grids: The received IRSA frame after passing through the channel.
        no: The noise variance.
        simulation_params: Dictionary containing all simulation parameters.
        resourse_grids: List of resource grids for each UE.
        bits: List of original bits for each UE.
        slot_indices: List of slot indices for each UE.
        num_simulations: Number of simulations to run.
        frame_size: Total number of slots in the frame.
        num_ues_per_frame: Number of UEs per frame.
        probabilities: Probabilities for selecting number of replicas.

    Returns:
        identified_ues: List of identified UEs.
        slots_to_decode: List of slots to decode in future passes.
    """
    # Extract necessary parameters from the simulation_params dictionary
    channel_params = simulation_params["Channel parameters"]
    is_perfect_SIC = channel_params['is_perfect_SIC']
    
    num_ues = num_simulations * num_ues_per_frame

    # Start decoding the received signal slot by slot
    identified_ues = set()
    
    pass_num = 1
    while len(identified_ues) < num_ues:
        print(f"\nPass {pass_num}:")
        new_identified_ues = set()
        new_identified_positions = set()

        bits_hat, h_hat = decode_frame(y_resource_grids, no, num_simulations, frame_size, simulation_params)
        new_ues, new_positions, new_ues_found = search_new_identified_ues(bits_hat, bits, slot_indices, identified_ues)
        
        new_identified_ues.update(new_ues)
        new_identified_positions.update(new_positions)

        if new_ues_found:
            y_resource_grids = remove_replicas_of_newlly_identified_ues(y_resource_grids, resourse_grids, slot_indices, new_identified_ues, is_perfect_SIC, h_hat)
            identified_ues.update(new_identified_ues)
        else:
            print("No new UEs were identified.")
            break

        pass_num += 1

    return list(identified_ues)

def run_simulation(simulation_params, num_simulations, num_ues_per_frame, frame_size, probabilities, ebno_db):
    """
    Run a single simulation.

    Args:
        simulation_params: Dictionary containing all simulation parameters.
        num_simulations: Number of simulations to run.
        num_ues_per_frame: Number of UEs per frame.
        frame_size: Total number of slots in the frame.
        probabilities: Probabilities for selecting number of replicas.
        ebno_db: The Eb/No value in dB.

    Returns:
        identified_ues: List of identified UEs.
    """
    # Generate the IRSA frame
    irsa_hyper_frame, resource_grids, h_ues, replicas_indices, bits = generate_hyper_irsa_frame(simulation_params, num_simulations, num_ues_per_frame, frame_size, probabilities)
    # Pass the IRSA frame through the AWGN channel
    received_frame, no = pass_through_awgn(irsa_hyper_frame, ebno_db, simulation_params)
    # Decode the IRSA frame
    identified_ues = decode_irsa_frame(received_frame, no, simulation_params, resource_grids, bits, replicas_indices, num_simulations, frame_size, num_ues_per_frame)
    
    return identified_ues

#%%
# Simulation parameters
simulation_params = {
    "Carrier parameters": {
        "num_resource_blocks": 1,
        "numerology": 0,
        "pilot_indices": [3, 9],
        "num_ofdm_symbols": 14
    },
    "Transport block parameters": {
        "num_bits_per_symbol": 2,
        "coderate": 0.5
    },
    "Channel parameters": {
        "is_phase_shift_applied": True,
        "is_perfect_SIC": False
    }
}
# Number of UEs per frame
num_ues_per_frame = 6
# Number of simulations to run
num_simulations = 1000
# Number of slots in the frame
frame_size = 10
# Probabilities for selecting number of replicas
probabilities = [0, 0.3, 0.15, 0.55]
# Eb/No value in dB
ebno_db = 100

# Run the simulation
identified_ues = run_simulation(simulation_params, num_simulations, num_ues_per_frame, frame_size, probabilities, ebno_db)
# Print the total number of ues and the identified ues
num_ues = num_simulations * num_ues_per_frame
print(f"\nTotal number of UEs: {num_ues}")
print(f"Identified UEs: {len(identified_ues)}")


#%%
# Plotting the performance of IRSA with varying number of UEs per frame
# Simulation parameters
simulation_params = {
    "Carrier parameters": {
        "num_resource_blocks": 1,
        "numerology": 0,
        "pilot_indices": [3, 9],
        "num_ofdm_symbols": 14
    },
    "Transport block parameters": {
        "num_bits_per_symbol": 2,
        "coderate": 0.5
    },
    "Channel parameters": {
        "is_phase_shift_applied": True,
        "is_perfect_SIC": False
    }
}
# Number of simulations to run
num_simulations = 1000
# Number of slots in the frame
frame_size = 15
# Probabilities for selecting number of replicas
probabilities = [0, 0.3, 0.15, 0.55]
# Eb/No value in dB
ebno_db = 100

# Initialize lists to store results
total_ues_per_frame_list = []
identified_ues_per_frame_list = []

# Vary the number of UEs per frame from 1 to frame_size - 1
for num_ues_per_frame in range(1, frame_size+1):
    # Run the simulation
    identified_ues = run_simulation(simulation_params, num_simulations, num_ues_per_frame, frame_size, probabilities, ebno_db)
    
    # Total number of UEs per frame
    total_ues_per_frame = num_ues_per_frame
    total_ues_per_frame_list.append(total_ues_per_frame)
    
    # Number of identified UEs per frame
    identified_ues_per_frame = len(identified_ues) / num_simulations
    identified_ues_per_frame_list.append(identified_ues_per_frame)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(total_ues_per_frame_list, identified_ues_per_frame_list, 'bo-', label='Identified UEs per Frame')
plt.xlabel('Total Number of UEs per Frame')
plt.ylabel('Number of Identified UEs per Frame')
plt.title('Performance of IRSA with Varying Number of UEs per Frame')
plt.legend()
plt.grid(True)
plt.show()

#%%
# Save the statistics to a CSV file
import csv

# File path
file_path = 'irsa_performance.csv'
# Write the statistics to the CSV file
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Total UEs per Frame', 'Identified UEs per Frame'])
    for i in range(len(total_ues_per_frame_list)):
        writer.writerow([total_ues_per_frame_list[i], identified_ues_per_frame_list[i]])
        
print(f"Statistics saved to {file_path}")
