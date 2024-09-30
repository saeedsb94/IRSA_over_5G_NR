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
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver

# Allow memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
    
print("The required libraries are imported successfully!")
print('sionna version:', sn.__version__)

#%%
# System parameters

# Carrier parameters
numerology = 0 # Numerology index
num_resource_blocks = 1 # Number of resource blocks
num_slots_per_frame = 1 # Number of slots per frame

# Transport block parameters
num_bits_per_symbol = 4 # QPSK
coderate = 0.5 # Code rate

# IRSA Parameters
num_frames = 1 # Number of frames
num_replicas_per_frame = 1 # Number of replicas per frame
num_ues = 1 # Number of UEs

# Create an instance of needed objects

# Set Resource grid parameters
resource_grid = sn.ofdm.ResourceGrid(num_ofdm_symbols=14,
                                     fft_size=12*num_resource_blocks, # Number of subcarriers
                                     subcarrier_spacing=1e3*(15*2**numerology), # Subcarrier spacing
                                     pilot_pattern="kronecker",
                                     pilot_ofdm_symbol_indices=[0, 2])

# Binary source
binary_source = sn.utils.BinarySource()

# Encoder
n = int(resource_grid.num_data_symbols*num_bits_per_symbol) # Number of coded bits in a resource grid
k = int(n*coderate) # Number of information bits in a resource grid
encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)

# QAM Mapper
mapper = sn.mapping.Mapper("qam", num_bits_per_symbol) # maps blocks of information bits to constellation symbols

# Resource grid mapper
resource_grid_mapper = sn.ofdm.ResourceGridMapper(resource_grid) # maps symbols onto an OFDM resource grid

# Channel components        
awgn_channel = sn.channel.AWGN()

ls_est = sn.ofdm.LSChannelEstimator(resource_grid, interpolation_type="nn")

# The demapper produces LLR for all coded bits
demapper = sn.mapping.Demapper("app", "qam", num_bits_per_symbol)

# The decoder provides hard-decisions on the information bits
decoder = sn.fec.ldpc.LDPC5GDecoder(encoder, hard_out=True)

output_frame = tf.zeros([num_slots_per_frame, resource_grid.num_ofdm_symbols, resource_grid.fft_size], dtype=tf.complex64)

phi_ues = tf.random.uniform(shape=[num_slots_per_frame, num_ues], minval=0, maxval=2*np.pi)
h_ues = tf.complex(tf.cos(phi_ues), tf.sin(phi_ues)) # Use tf.complex for complex numbers in TensorFlow

# Store the original bits for comparison
original_bits = []

# Store the positions of the replicas for each UE
replica_positions = {}

# Store the symbols_rg for each UE to reuse in SIC later
symbols_rg_store = {}

for i in range(num_ues):
    # Generate bits
    bits = binary_source([num_frames, k])
    original_bits.append(bits)
    
    # Encode bits
    codewords = encoder(bits)
    
    # Map codewords to symbols
    symbols = mapper(codewords)
    
    # Map symbols to resource grid (add two dimensions to the symbols tensor)
    symbols_rg = resource_grid_mapper(tf.expand_dims(tf.expand_dims(symbols, axis=1), axis=1))
    symbols_rg = tf.squeeze(tf.squeeze(symbols_rg, axis=1), axis=1)
    
    # Save the symbols_rg for later use in SIC
    symbols_rg_store[i] = symbols_rg
    
    # Create a list of unique random indices to select different positions of the replicas
    replicas_indices = np.random.choice(num_slots_per_frame, num_replicas_per_frame, replace=False)
    
    # Save the positions of the replicas for the current UE
    replica_positions[i] = replicas_indices
    
    # Print the indices of the replicas for the current UE
    print(f"UE {i+1} replicas indices: {replicas_indices}")
    
    # Create an empty tensor for full frame
    frame_ue = tf.zeros([num_slots_per_frame, resource_grid.num_ofdm_symbols, resource_grid.fft_size], dtype=tf.complex64)
    
    # Scatter the replicas to the empty tensor to form the full frame while passing through the channel
    for replica_idx in replicas_indices:
        # Get the channel for the current UE and replica index
        h_ue = h_ues[replica_idx, i]
        # pass the symbols through the channel
        symbols_rg = symbols_rg * h_ue
        frame_ue = tf.tensor_scatter_nd_update(frame_ue, [[replica_idx]], symbols_rg)
    
    # add the replicas to the output tensor
    output_frame += frame_ue

# Pass through AWGN channel
no = sn.utils.ebnodb2no(100, num_bits_per_symbol=num_bits_per_symbol, coderate=coderate)
y_combined = y = awgn_channel([output_frame, no])

# Extract the pilot and data indices
pilot_indices = resource_grid._pilot_ofdm_symbol_indices
data_indices = np.setdiff1d(np.arange(resource_grid.num_ofdm_symbols), pilot_indices)
# Decode the received signal slot by slot
decoded_bits = []
identified_replicas = {i: [] for i in range(num_ues)}
slots_to_decode = list(range(num_slots_per_frame))
undecoded_ues = list(range(num_ues))

pass_num = 1
while slots_to_decode:
    print(f"\nPass {pass_num}:")
    new_identified_replicas = []
    slots_to_ignore = []
    for slot in slots_to_decode:
        # Get the received signal for the current slot
        y_slot = y_combined[slot, :, :]
        
        # Estimate the channel for the current slot
        h_hat, err_var = ls_est([tf.expand_dims(tf.expand_dims(tf.expand_dims(y_slot, axis=0), axis=1), axis=1), no])
        h_hat = tf.squeeze(h_hat)
        
        # Perform the equalization
        y_symbols_rg_equalized = y_slot / h_hat
        
        # extract the data symbols
        y_symbols = tf.reshape(tf.gather(y_symbols_rg_equalized, data_indices, axis=0), -1)
        
        # Perform the demapping
        llr = demapper([y_symbols, no])
        
        # Perform the decoding
        bits_hat = decoder(tf.expand_dims(llr, axis=0))
        decoded_bits.append(bits_hat)
        
        # Print the processing report for the current slot
        print(f"Processing slot {slot}:")
        for i in undecoded_ues:
            is_match = tf.reduce_all(tf.equal(bits_hat, original_bits[i]))
            print(f"    UE {i+1} : {is_match.numpy()}")
            
            if is_match.numpy():
                # If the decoded bits match the original bits, we can identify the positions of the other replicas
                # Add all replica indices of the identified UE to new_identified_replicas
                new_identified_replicas.append(i)
                identified_replicas[i].append(replica_positions[i])
                # Mark the identified positions for removal after the loop
                slots_to_ignore.append(slot)
                # Remove the identified UE from the list of undecoded UEs
                undecoded_ues.remove(i)
                break  # Break the for loop when a UE is recognized in one slot
    
    # Remove identified slots from slots_to_decode
    for slot in slots_to_ignore:
        print(f"Ignoring slot {slot} as it has been successfully decoded.")
        slots_to_decode.remove(slot)
    
    # Only remove replicas if new UEs were identified
    if new_identified_replicas:
        # Remove the replicas of the discovered UEs at positions other than where they were discovered
        for ue in new_identified_replicas:
            for pos in replica_positions[ue]:
                if pos not in slots_to_ignore:
                    y_replica_slot = y_combined[pos, :, :]
                    # We start by re-estimate the channel for the current slot using the known data
                    # h_hat_2 = tf.math.de_no_nan(y_replica_slot, symbols_rg_store[ue])
                    h_hat_2 = tf.math.divide_no_nan(
                        y_replica_slot * tf.math.conj(symbols_rg_store[ue]),
                        tf.complex(tf.abs(symbols_rg_store[ue])**2, 0.0)
                    )
                    # use the estimated channel to remove the ue signal at the current slot
                    clean_slot = y_replica_slot - symbols_rg_store[ue] * h_hat_2
                    # update the received signal
                    y_combined = tf.tensor_scatter_nd_update(y_combined, [[pos]], clean_slot)
                    print(f"Removing replica of UE {ue+1} from slot {pos}.")
                
    # If no new replicas were identified, break the loop
    if not new_identified_replicas:
        print("No new slots were decoded.")
        break
    
    pass_num += 1

# Print only the identified UEs
identified_ues = [ue + 1 for ue in identified_replicas if identified_replicas[ue]]
print(f"Identified UEs: {', '.join(map(str, identified_ues))}")

# Print the remaining slots to discover
print(f"Slots to decode in future passes: {slots_to_decode}")


# Report
print("Transmission Report:")
print(f"Number of UEs: {num_ues}")
print(f"Number of frames: {num_frames}")
print(f"Number of replicas per frame: {num_replicas_per_frame}")
print(f"Number of slots per frame: {num_slots_per_frame}")
print(f"Resource grid dimensions: {resource_grid.num_ofdm_symbols} OFDM symbols x {resource_grid.fft_size} subcarriers")
print(f"Output frame shape: {output_frame.shape}")
