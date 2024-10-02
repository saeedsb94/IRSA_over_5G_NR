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

def generate_ue_resource_grid(simulation_params):
    """
    Create the UE resource grid to be transmitted and the indices of the replicas for one UE.

    Args:
        simulation_params: Dictionary containing all simulation parameters.

    Returns:
        symbols_rg: The UE resource grid to be transmitted.
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
    resource_grid = sn.ofdm.ResourceGrid(
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
    n = int(resource_grid.num_data_symbols * num_bits_per_symbol)
    k = int(n * coderate)
    
    # LDPC Encoder: Encodes the information bits into coded bits
    encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)
    
    # QAM Mapper: Maps blocks of information bits to constellation symbols
    mapper = sn.mapping.Mapper("qam", num_bits_per_symbol)
    
    # Resource Grid Mapper: Maps symbols onto an OFDM resource grid
    resource_grid_mapper = sn.ofdm.ResourceGridMapper(resource_grid)
    
    # Transmission
    # Generate random binary bits for the UE
    bits = binary_source([k])
    
    # Encode the bits using LDPC encoder
    codewords = encoder(tf.expand_dims(bits, axis=0))
    
    # Map the encoded bits to QAM symbols
    symbols = mapper(codewords)
    
    # Map the QAM symbols onto the OFDM resource grid
    resource_grid = resource_grid_mapper(tf.expand_dims(tf.expand_dims(symbols, axis=1), axis=1))
    
    # Remove the extra dimensions added during mapping
    resource_grid = tf.squeeze(tf.squeeze(resource_grid, axis=1), axis=1)
    
    return resource_grid, bits

def generate_replicas_indices(simulation_params):
    """Generate random number of replicas for a single UE and their positions within the frame.
    
    Args:
        simulation_params: Dictionary containing all simulation parameters.
        
    Returns:
        list: A list containing the slot indices selected by the UE.
    """
    # Extract necessary parameters from the simulation_params dictionary
    num_slots_per_frame = simulation_params["IRSA Parameters"]["num_slots_per_frame"]
    max_replicas = simulation_params["IRSA Parameters"]["max_replicas"]
    
    # Generate a random number of replicas for the UE
    num_replicas = np.random.randint(1, max_replicas + 1)
    
    # Generate slot indices for the UE
    slot_indices = np.random.choice(num_slots_per_frame, num_replicas, replace=False)
    
    return slot_indices

def generate_irsa_frame(simulation_params):
    """
    Generate an IRSA frame based on the given simulation parameters.

    Args:
        simulation_params: Dictionary containing all simulation parameters.

    Returns:
        irsa_frame: The generated IRSA frame.
        resource_grid_list: List of resource grids for each UE.
        h_ues: Channel coefficients for each UE in each slot.
        replicas_indices_list: List of replica indices for each UE.
        original_bits_list: List of original bits for each UE.
    """
    # Extract necessary parameters from the simulation_params dictionary
    irsa_params = simulation_params["IRSA Parameters"]
    num_slots_per_frame = irsa_params['num_slots_per_frame']
    num_ues = irsa_params['num_ues']

    carrier_params = simulation_params["Carrier parameters"]
    num_ofdm_symbols = carrier_params['num_ofdm_symbols']
    num_resource_blocks = carrier_params['num_resource_blocks']
    fft_size = 12 * num_resource_blocks

    channel_params = simulation_params["Channel parameters"]
    channel_type = channel_params['type']

    # Create the output frame tensor
    irsa_frame = tf.zeros([num_slots_per_frame, num_ofdm_symbols, fft_size], dtype=tf.complex64)

    # Generate random phase shifts for each UE in each slot
    if channel_type == "Only Phase Shift":
        phi_ues = tf.random.uniform(shape=[num_slots_per_frame, num_ues], minval=0, maxval=2 * np.pi)
        h_ues = tf.complex(tf.cos(phi_ues), tf.sin(phi_ues))
    else:
        h_ues = tf.ones([num_slots_per_frame, num_ues], dtype=tf.complex64)

    # Lists to store the resource grids, replica indices, and original bits for each UE
    resource_grid_list = []
    replicas_indices_list = []
    original_bits_list = []

    for i in range(num_ues):
        # Generate the resource grid and original bits for the current UE
        resource_grid, bits = generate_ue_resource_grid(simulation_params)
        resource_grid_list.append(resource_grid)
        original_bits_list.append(bits)

        # Generate the replica indices for the current UE
        replicas_indices = generate_replicas_indices(simulation_params)
        replicas_indices_list.append(replicas_indices)

        # Print the indices of the replicas for the current UE
        print(f"UE {i+1} replicas indices: {replicas_indices}")

        # Create an empty tensor for the full frame of the current UE with the same shape as irsa_frame
        ue_frame = tf.zeros_like(irsa_frame)

        # Scatter the replicas to the empty tensor to form the full frame while passing through the channel
        for replica_idx in replicas_indices:
            h_ue = h_ues[replica_idx, i]
            resource_grid_with_channel = resource_grid * h_ue
            ue_frame = tf.tensor_scatter_nd_update(ue_frame, [[replica_idx]], resource_grid_with_channel)

        # Add the replicas to the output tensor
        irsa_frame += ue_frame

    return irsa_frame, resource_grid_list, h_ues, replicas_indices_list, original_bits_list

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

def decode_irsa_frame(received_frame, no, simulation_params, resource_grid_list, original_bits_list, replicas_indices_list):
    """
    Decode an IRSA frame.

    Args:
        received_frame: The received IRSA frame after passing through the channel.
        no: The noise variance.
        simulation_params: Dictionary containing all simulation parameters.
        resource_grid_list: List of resource grids for each UE.
        original_bits_list: List of original bits for each UE.
        replicas_indices_list: List of replica indices for each UE.

    Returns:
        identified_ues: List of identified UEs.
        slots_to_decode: List of slots to decode in future passes.
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

    irsa_params = simulation_params["IRSA Parameters"]
    num_slots_per_frame = irsa_params['num_slots_per_frame']
    num_ues = irsa_params['num_ues']

    # Create instances of needed objects
    resource_grid = sn.ofdm.ResourceGrid(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=12 * num_resource_blocks,
        subcarrier_spacing=1e3 * (15 * 2 ** numerology),
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=pilot_indices
    )
    ls_est = sn.ofdm.LSChannelEstimator(resource_grid, interpolation_type="nn")
    demapper = sn.mapping.Demapper("app", "qam", num_bits_per_symbol)
    
    # Calculate the number of coded bits (n) and information bits (k) in a resource grid
    n = int(resource_grid.num_data_symbols * num_bits_per_symbol)
    k = int(n * coderate)
    
    # LDPC Encoder: Encodes the information bits into coded bits
    encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)
    
    # LDPC Decoder: Decodes the coded bits back into information bits
    decoder = sn.fec.ldpc.LDPC5GDecoder(encoder, hard_out=True)

    # Start decoding the received signal slot by slot
    decoded_bits = []
    identified_replicas = {i: [] for i in range(num_ues)}
    slots_to_decode = list(range(num_slots_per_frame))
    undecoded_ues = list(range(num_ues))

    pass_num = 1
    while slots_to_decode:
        print(f"\nPass {pass_num}:")
        new_identified_replicas = []
        slots_to_ignore = []
        for slot_index in slots_to_decode:
            received_rg = received_frame[slot_index, :, :]
            h_hat, err_var = ls_est([tf.expand_dims(tf.expand_dims(tf.expand_dims(received_rg, axis=0), axis=1), axis=1), no])
            h_hat = tf.squeeze(h_hat)
            received_rg_equalized = received_rg / h_hat
            received_symbols = tf.reshape(tf.gather(received_rg_equalized, data_indices, axis=0), -1)
            llr = demapper([received_symbols, no])
            bits_hat = decoder(tf.expand_dims(llr, axis=0))
            decoded_bits.append(bits_hat)

            print(f"Processing slot {slot_index}:")
            for i in undecoded_ues:
                is_match = tf.reduce_all(tf.equal(bits_hat, original_bits_list[i]))
                print(f"    UE {i+1} : {is_match.numpy()}")

                if is_match.numpy():
                    new_identified_replicas.append(i)
                    identified_replicas[i].append(replicas_indices_list[i])
                    slots_to_ignore.append(slot_index)
                    undecoded_ues.remove(i)
                    break

        for slot_index in slots_to_ignore:
            print(f"Ignoring slot {slot_index} as it has been successfully decoded.")
            slots_to_decode.remove(slot_index)

        if new_identified_replicas:
            for ue in new_identified_replicas:
                for pos in replicas_indices_list[ue]:
                    if pos not in slots_to_ignore:
                        y_replica_slot = received_frame[pos, :, :]
                        phi_hat = tf.math.angle(tf.math.reduce_sum(y_replica_slot * tf.math.conj(resource_grid_list[ue])))
                        h_hat_2 = tf.complex(tf.cos(phi_hat), tf.sin(phi_hat))
                        clean_slot = y_replica_slot - resource_grid_list[ue] * h_hat_2
                        received_frame = tf.tensor_scatter_nd_update(received_frame, [[pos]], clean_slot)
                        print(f"Removing replica of UE {ue+1} from slot {pos}.")

        if not new_identified_replicas:
            print("No new slots were decoded.")
            break
        
        
        print(f"Slots to decode in future passes: {slots_to_decode}")

        pass_num += 1
    
    identified_ues = [ue+1 for ue in range(num_ues) if ue not in undecoded_ues]    
    return identified_ues, slots_to_decode




#%%
# Define a container to save all simulation parameters
simulation_params = {
    "Carrier parameters": {
        "numerology": 0,
        "num_resource_blocks": 1,
        "num_ofdm_symbols": 14,
        "pilot_indices": [2, 11],
    },
    "Transport block parameters": {
        "num_bits_per_symbol": 2,
        "coderate": 0.5,
    },
    "IRSA Parameters": {
        "num_slots_per_frame": 10,
        "num_ues": 6,
        "max_replicas": 2,  # Add a variable to control the max number of replicas allowed
    },
    "Channel parameters": {
        "type": "Only Phase Shift",  # Options: "None", "Only Phase Shift"
    }
}

# Run the IRSA simulation
irsa_frame, resource_grid_list, h_ues, replicas_indices_list, original_bits_list = generate_irsa_frame(simulation_params)
received_frame, no = pass_through_awgn(irsa_frame, 100, simulation_params)
identified_ues, slots_to_decode = decode_irsa_frame(received_frame, no, simulation_params, resource_grid_list, original_bits_list, replicas_indices_list)

# Print the transmission report (Add some separation lines for better readability)
print("\n\n")
print("=====================================================")
print("Transmission Report:")
print(f"Number of UEs: {simulation_params['IRSA Parameters']['num_ues']}")
print(f"Number of slots per frame: {simulation_params['IRSA Parameters']['num_slots_per_frame']}")
print(f"Number of identified UEs: {len(identified_ues)}")
print(f"Identified UEs: {', '.join(map(str, identified_ues))}")
print("=====================================================")


# Plot the percen

# %%

# Define a container to save all simulation parameters
simulation_params = {
    "Carrier parameters": {
        "numerology": 0,
        "num_resource_blocks": 1,
        "num_ofdm_symbols": 14,
        "pilot_indices": [2, 11],
    },
    "Transport block parameters": {
        "num_bits_per_symbol": 2,
        "coderate": 0.5,
    },
    "IRSA Parameters": {
        "num_slots_per_frame": 10,
        "num_ues": None,
        "max_replicas": 2,  # Add a variable to control the max number of replicas allowed
    },
    "Channel parameters": {
        "type": "Only Phase Shift",  # Options: "None", "Only Phase Shift"
    }
}

# Plot the performance of the IRSA decoder for varying number of UEs
def plot_performance_varying_ues(simulation_params, max_num_ues, step=1):
    num_slots_per_frame = simulation_params["IRSA Parameters"]["num_slots_per_frame"]
    identified_ues_list = []

    for num_ues in range(1, max_num_ues + 1, step):
        simulation_params["IRSA Parameters"]["num_ues"] = num_ues
        irsa_frame, resource_grid_list, h_ues, replicas_indices_list, original_bits_list = generate_irsa_frame(simulation_params)
        received_frame, no = pass_through_awgn(irsa_frame, 100, simulation_params)
        identified_ues, slots_to_decode = decode_irsa_frame(received_frame, no, simulation_params, resource_grid_list, original_bits_list, replicas_indices_list)
        identified_ues_list.append(len(identified_ues))
        print(f"Total UEs: {num_ues}, Identified UEs: {len(identified_ues)}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_num_ues + 1, step), identified_ues_list, marker='o', linestyle='-', color='b')
    plt.xlabel("Total Number of UEs")
    plt.ylabel("Number of Identified UEs")
    plt.title(f"Performance of IRSA Decoder (Fixed {num_slots_per_frame} Slots per Frame)")
    plt.grid(True)
    plt.show()

# Set the maximum number of UEs to test
max_num_ues = 10
plot_performance_varying_ues(simulation_params, max_num_ues)
# %%
