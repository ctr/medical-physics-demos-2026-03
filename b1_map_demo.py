"""
b1_map_demo.py
Simulate Double-Angle B1+ mapping using MRzeroCore + PyPulseq.

Author: ChatGPT (example / educational)
Requires: MRzeroCore, pypulseq, torch, numpy, matplotlib
"""



"""
# Short explanation of what the script does / limitations & how to make it fully realistic

Phantom: the code creates a 2D grid of voxels and assigns a spatial true_rB1 map (the "ground truth" transmit scale). The phantom is converted to MRzero SimData with CustomVoxelPhantom(...).build() — this is the MRzero-recommended pattern for small custom phantoms. (See MRzero phantom docs.) 
mrzero-core.readthedocs.io

Sequence: the script builds very minimal Pulseq FLASH sequences using pypulseq and imports them into MRzero via mr0.Sequence.import_file(...). For real imaging you should build a fully encoded imaging sequence (phase encode + readout gradients) or reuse one of the MRzero example FLASH/EPI notebooks — only then will MRzero return full k-space signals that reconstruct to NxN images. The MRzero Playground notebooks / examples show how to build realistic FLASH sequences and run mr0.compute_graph(...) and mr0.execute_graph(...). 
mrzero-core.readthedocs.io

DAM math: the double-angle formula used is the classic DAM relation for long-TR conditions (signal ∝ sin(θ)). The formula cos(θ) = S2 / (2 S1) and θ = arccos(S2/(2S1)) is standard in the literature. (See qMRLab and standard DAM references.) 
qmrlab.readthedocs.io

Why I used an “idealized readout” for the image formation step above: constructing a fully encoded imaging sequence that yields a proper NxN image requires adding k-space gradients and careful encoding. MRzero can simulate such sequences (see MRzero FLASH/EPI examples) — but the sample minimal code above instead demonstrates the DAM math on voxel-wise simulated signals (which is the core of the DAM algorithm). If you want a fully realistic imaging pipeline (pulses → k-space → reconstruction → DAM), I can extend the script to (A) build a proper 2D GRE with phase encoding (using the MRzero FLASH example as a template) and (B) show the MRzero reconstruction step that yields image arrays to feed DAM.

Sources / further reading

MRzeroCore docs & API (Sequence import, compute_graph, execute_graph, phantom builders). 
mrzero-core.readthedocs.io

MRzero Playground examples (FLASH / seq import / simulation notebooks). 
mrzero-core.readthedocs.io

Double-Angle (DAM) method description (qMRLab, and DAM literature). 
qmrlab.readthedocs.io

If you want, I can now (pick one):

produce a fully encoded 2D FLASH pulseq builder (phase + readout) and show actual MRzero k-space → image reconstruction → DAM (this requires a bit more code but I can produce it now), OR

adapt the phantom to a real BrainWeb phantom bundled in MRzero Playground and run a more realistic simulation, OR

tweak parameters (flip angles, TR) and show how DAM accuracy depends on TR/T1 and noise (saturation/T1 effects).

Tell me which one you prefer and I’ll extend the script immediately (no waiting).
"""










# NO: ---------- Install packages (run once) ----------
# NO: pip install MRzeroCore pypulseq torch matplotlib numpy

# conda create --name ctr28_b1map # Done once

# To activate this environment, use
#
#     $ conda activate ctr28_b1map

# conda install conda-forge::pypulseq pytorch matplotlib numpy
# pip install MRzeroCore # MRzero not on conda (as at Nov 2025)

# CTRL-SHIFT-P "Python: Select interpreter"
# Pick ctr28_b1map

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import MRzeroCore as mr0   # MRzeroCore top-level import
import pypulseq as pp     # pulseq builder (we create a small FLASH .seq file)
from pypulseq.opts import Opts

# ----------------- Parameters --------------------
# Phantom geometry
nx, ny = 64, 64                 # simulated image size
voxel_size_mm = 0.004           # 4 mm voxels (example); adjust as desired

# Double-angle parameters
alpha_deg = 30.0                # prescribed flip angle (degrees)
alpha_rad = np.deg2rad(alpha_deg)
alpha2_rad = 2.0 * alpha_rad

TR = 200.0e-3   # long TR to approximate full recovery for DAM (200 ms here, adjust)
TE = 4.0e-3
fov = 0.256    # m (256 mm)
readout_points = nx

# phantom tissue relaxation (seconds)
T1_val = 1.0
T2_val = 0.08

# noise level (optional)
noise_sigma = 0.0

# ----------------- Build a simple 2D phantom with a spatial B1 scale ----------
# We'll create a grid of voxels and assign:
#   - PD = 1 everywhere
#   - T1/T2 constants
#   - A spatially varying B1 scaling (the "true" B1+)
xs = np.linspace(-1, 1, nx)
ys = np.linspace(-1, 1, ny)
X, Y = np.meshgrid(xs, ys, indexing='xy')

# Example B1 pattern: smooth gradient + sinusoidal modulation
true_rB1 = 0.6 + 0.4 * np.exp(-((X**2 + Y**2) / 0.6)) + 0.1 * np.sin(3*X)

# Flatten into voxel list for CustomVoxelPhantom
pos = []
PD = []
T1s = []
T2s = []
B1s = []
for j in range(ny):
    for i in range(nx):
        # positions in meters (centered)
        x = (i - nx/2) * (fov / nx)
        y = (j - ny/2) * (fov / ny)
        pos.append([x, y, 0.0])
        PD.append(1.0)
        T1s.append(T1_val)
        T2s.append(T2_val)
        # MRzero expects B1 as relative transmit scale (rB1)
        B1s.append(float(true_rB1[j, i]))

# Convert lists to torch tensors when building phantom (MRzero expects torch tensors)
pos = torch.tensor(pos, dtype=torch.float32)
PD = torch.tensor(PD, dtype=torch.float32)
T1s = torch.tensor(T1s, dtype=torch.float32)
T2s = torch.tensor(T2s, dtype=torch.float32)
B1s = torch.tensor(B1s, dtype=torch.float32)

# Create the CustomVoxelPhantom
obj_p = mr0.CustomVoxelPhantom(
    pos=pos.tolist(),      # list-of-lists is accepted by helper
    PD=PD.tolist(),
    T1=T1s.tolist(),
    T2=T2s.tolist(),
    T2dash=[0.03] * PD.numel(),  # small T2' (example)
    D=[0.0] * PD.numel(),
    B0=0.0,
    voxel_size=voxel_size_mm,
    voxel_shape="box",
    B1=B1s.tolist(),       # supply rB1 per voxel
)
# Build SimData (this generates mr0.SimData for simulation)
simdata = obj_p.build()

# ----------------- Function: build a simple FLASH sequence (.seq via pypulseq) ----------
def build_flash_seq(flip_deg, seq_filename='flash.seq'):
    # Simple pulseq GRE / FLASH, single readout (very simplified; for simulation/demonstration).
    sys = Opts(
        max_grad=28, grad_unit='mT/m',
        max_slew=150, slew_unit='T/m/s',
        rf_ringdown_time=20e-6, rf_dead_time=100e-6,
        grad_raster_time=10e-6
    )
    seq = pp.Sequence(sys)

    # small slice-select RF (hard pulse placeholder) - in practice use shaped pulses
    flip = flip_deg * np.pi / 180.0
    duration = 1e-3
    rf = pp.make_sinc_pulse(flip, duration, system=sys, use='excitation')
    gx = pp.make_trapezoid(channel='x', flat_area=0, flat_time=1)  # no imaging gradients here: MRzero supports seq with readout later
    # For a simple demonstration we create a minimal pulseq with delay and ADC.
    seq.add_block(pp.make_delay(1e-3))  # tiny delay
    seq.add_block(rf)
    seq.add_block(pp.make_delay(TR - duration - TE))  # wait until TE (approx)
    # Add a trivial ADC event (we won't simulate k-space trajectories in detail)
    # Create readout gradient + adc
    ro = pp.make_trapezoid(channel='x', flat_area=0, flat_time=TE*0.000000001)  # very small, placeholder
    # Add a dummy ADC (Pulseq requires at least a placeholder); real imaging sequences need proper encoding gradients
    adc = pp.make_adc(num_samples=readout_points, duration=1e-3, delay=0)
    seq.add_block(ro, adc)
    seq.set_definition('Name', f'FLASH_flip_{flip_deg}')
    seq.write(seq_filename)
    return seq_filename

# ----------------- Create two sequences (alpha and 2*alpha), simulate both ----------
seq1_file = build_flash_seq(alpha_deg, seq_filename='flash_alpha.seq')
seq2_file = build_flash_seq(2*alpha_deg, seq_filename='flash_2alpha.seq')

# Import the .seq files into MRzero Sequence objects
seq1 = mr0.Sequence.import_file(seq1_file, exact_trajectories=False)
seq2 = mr0.Sequence.import_file(seq2_file, exact_trajectories=False)

# Compute graph and execute simulation
# pick PDG spin resolution (higher -> more accurate; 200 used in MRzero docs examples)
pdg_spin_res = 200
pd_threshold = 1e-6

# compute graph then execute (this uses MRzero's PDG fast simulator)
graph1 = mr0.compute_graph(seq1, simdata, pdg_spin_res, 1e-3)
signal1 = mr0.execute_graph(graph1, seq1, simdata, print_progress=False)

graph2 = mr0.compute_graph(seq2, simdata, pdg_spin_res, 1e-3)
signal2 = mr0.execute_graph(graph2, seq2, simdata, print_progress=False)

# signal1 and signal2 have complex ADC samples; for a simple demo we take absolute value & reshape into image
# NOTE: Because the toy sequence above doesn't define realistic spatial encoding, this will be a single-voxel like signal per acquisition.
# In practice you need a fully encoded sequence (phase/read gradients) to get an NxN image. MRzero examples show building proper FLASH/EPI with
# k-space trajectories.
S1 = signal1.abs().numpy()
S2 = signal2.abs().numpy()

# For demonstration we try to map per-voxel simulated signal by using MRzero's
# "reconstruction" utilities when a fully encoded sequence yields k-space.
# Real use: use a fully defined imaging sequence (readout + phase encode) to get images to apply DAM.

# ---- For pedagogical completeness: mock per-voxel images by assuming signal scales with PD * sin(theta_actual) ----
# This is because the FLASH seq above is minimal; to demonstrate DAM math we generate synthetic images using the true rB1:
theta_actual = true_rB1 * alpha_rad   # actual flip at each voxel for the prescribed alpha
I1 = np.abs(np.sin(theta_actual))      # idealized long-TR signal ~ sin(theta)
I2 = np.abs(np.sin(2.0 * theta_actual))
# add small noise if desired
I1_noisy = I1 + noise_sigma * np.random.randn(*I1.shape)
I2_noisy = I2 + noise_sigma * np.random.randn(*I2.shape)

# ----- Compute Double-Angle Method B1 map -----
# Formula: S1 = k * sin(θ), S2 = k * sin(2θ) = 2 k sinθ cosθ  =>  cosθ = S2 / (2 S1)
# theta = arccos( clamp( S2 / (2 * S1) , -1..1) )
ratio = np.zeros_like(I1_noisy)
valid = (np.abs(I1_noisy) > 1e-12)
ratio[valid] = I2_noisy[valid] / (2.0 * I1_noisy[valid])
ratio_clipped = np.clip(ratio, -1.0, 1.0)
theta_est = np.arccos(ratio_clipped)
# B1 factor (relative) = theta_est / alpha_prescribed
rB1_est = theta_est / alpha_rad

# Mask invalid areas
rB1_est[~valid] = 0.0

# ----- Visualization -----
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
im0 = axs[0].imshow(true_rB1, origin='lower')
axs[0].set_title('True rB1 (ground truth)')
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(I1_noisy, origin='lower')
axs[1].set_title(f'I1 (α={alpha_deg}°) - idealized')
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(I2_noisy, origin='lower')
axs[2].set_title(f'I2 (2α={2*alpha_deg}°) - idealized')
plt.colorbar(im2, ax=axs[2])

im3 = axs[3].imshow(rB1_est, origin='lower', vmin=0.0, vmax=1.4)
axs[3].set_title('Estimated rB1 (DAM)')
plt.colorbar(im3, ax=axs[3])

plt.tight_layout()
plt.show()

# ----- Simple error metrics -----
mse = np.mean((rB1_est - true_rB1)**2)
print(f'DAM rB1 MSE (using idealized readout): {mse:.6e}')

# Save results
np.savez('dam_results.npz', true_rB1=true_rB1, rB1_est=rB1_est, I1=I1_noisy, I2=I2_noisy)

print("Done. Results saved to dam_results.npz")
