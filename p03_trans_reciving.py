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
num_slots_per_frame = 3 # Number of slots per frame

# Transport block parameters
num_bits_per_symbol = 4 # QPSK
coderate = 0.5 # Code rate

# IRSA Parameters
num_frames = 1 # Number of frames
num_replicas_per_frame =2 # Number of replicas per frame
num_ues = 2 # Number of UEs

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
    
    # Create a list of unique random indices to select different positions of the replicas
    replicas_indices = np.random.choice(num_slots_per_frame, num_replicas_per_frame, replace=False)
    
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
no = sn.utils.ebnodb2no(10, num_bits_per_symbol=num_bits_per_symbol, coderate=coderate)
y_combined = y = awgn_channel([output_frame, no])

# Extract the pilot and data indices
pilot_indices = resource_grid._pilot_ofdm_symbol_indices
data_indices = np.setdiff1d(np.arange(resource_grid.num_ofdm_symbols), pilot_indices)

# Decode the received signal slot by slot
decoded_bits = []
for slot in range(num_slots_per_frame):
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

# Compare the decoded bits with the original bits for each slot
for slot in range(num_slots_per_frame):
    for i in range(num_ues):
        are_equal = tf.reduce_all(tf.equal(decoded_bits[slot], original_bits[i]))
        print(f"The decoded bits in slot {slot+1} belong to UE {i+1}: {are_equal.numpy()}")

# Report
print("Transmission Report:")
print(f"Number of UEs: {num_ues}")
print(f"Number of frames: {num_frames}")
print(f"Number of replicas per frame: {num_replicas_per_frame}")
print(f"Number of slots per frame: {num_slots_per_frame}")
print(f"Resource grid dimensions: {resource_grid.num_ofdm_symbols} OFDM symbols x {resource_grid.fft_size} subcarriers")
print(f"Output frame shape: {output_frame.shape}")




# %%
# squeeze the output frame
print(f"Output frame shape after squeezing: {output_frame.shape}")

slot=6
output_frame[slot, :, :]

#%%
resource_grid = sn.ofdm.ResourceGrid( num_ofdm_symbols=14,
                                      fft_size=12*num_resource_blocks,# Number of subcarriers
                                      subcarrier_spacing=1e3*(15*2**numerology), # Subcarrier spacing
                                      num_tx=2,
                                      pilot_pattern="kronecker",
                                      pilot_ofdm_symbol_indices=[0,2]) 
resource_grid.pilot_pattern.show()
pilot=resource_grid.pilot_pattern.pilots



num_bits_per_symbol = 2 # QPSK

binary_source = sn.utils.BinarySource()

mapper = sn.mapping.Mapper("qam", num_bits_per_symbol)

# The resource grid mapper maps symbols onto an OFDM resource grid
rg_mapper = sn.ofdm.ResourceGridMapper(resource_grid)



n = int(resource_grid.num_data_symbols*num_bits_per_symbol) # Number of bits to be transmitted
bits = binary_source([1, resource_grid.num_tx,resource_grid.num_streams_per_tx, n])
symbols = mapper(bits)
symbols_rg = rg_mapper(symbols)
print(f"Resource grid shape: {symbols_rg.shape}")
#%%
symbols_rg = tf.squeeze(symbols_rg)
rg_ue1 = symbols_rg[0, :, :]
rg_ue2 = symbols_rg[1, :, :]
# Plot the resource grid of the two UEs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.abs(rg_ue1), aspect='auto')
plt.colorbar()
plt.title('UE1 Resource Grid')
plt.subplot(1, 2, 2)
plt.imshow(np.abs(rg_ue2), aspect='auto')
plt.colorbar()
plt.title('UE2 Resource Grid')
plt.show()


#%%

pusch_config = sn.nr.PUSCHConfig()
dmrs_grid=pusch_config.dmrs_grid
dmrs_grid=tf.squeeze(dmrs_grid)
# plot the DMRS grid
plt.figure(figsize=(10, 5))
plt.imshow(np.abs(dmrs_grid), aspect='auto')
plt.colorbar()
plt.title('DMRS Grid')
plt.show()
pusch_config_1 = pusch_config.clone()
dmrs_grid_1=pusch_config_1.dmrs_grid
dmrs_grid_1=tf.squeeze(dmrs_grid_1)
# plot the DMRS grid
plt.figure(figsize=(10, 5))
plt.imshow(np.abs(dmrs_grid_1), aspect='auto')
plt.colorbar()
plt.title('DMRS Grid')
plt.show()
are_equal=tf.reduce_all(tf.equal(dmrs_grid,dmrs_grid_1))
print("Are the tensors exactly equal? ", are_equal.numpy())
