import modal
import os

sim_image = (
    modal.Image.from_registry("nvidia/cuda:12.0.1-devel-ubuntu22.04")
    .apt_install("python-is-python3", "python3-pip")
    .pip_install("torch", "pyflamegpu", "numpy", "h5py", extra_index_url="https://whl.flamegpu.com/whl/cuda120/")
    .env({"CUDA_PATH": "/usr/local/cuda"})
    .add_local_dir(".", remote_path="/root/compoxel") 
)

blender_image = (
    modal.Image.debian_slim()
    .apt_install("wget", "xz-utils", "libxrender1", "libxi6", "libxkbcommon0", "libgl1-mesa-glx", "libsm6", "libice6", "libxext6", "libx11-6")
    .run_commands(
        "wget https://download.blender.org/release/Blender4.1/blender-4.1.0-linux-x64.tar.xz",
        "tar -xf blender-4.1.0-linux-x64.tar.xz",
        "mv blender-4.1.0-linux-x64 /usr/local/blender",
        "/usr/local/blender/blender -b --python-expr \"import subprocess, sys; subprocess.run([sys.executable, '-m', 'ensurepip']); subprocess.run([sys.executable, '-m', 'pip', 'install', 'h5py', 'numpy'])\""
    )
)

app = modal.App("compoxel-v1-alembic")
shared_volume = modal.Volume.from_name("compoxel-storage", create_if_missing=True)

@app.function(gpu="T4", timeout=1800, image=sim_image, volumes={"/vol": shared_volume})
def run_simulation(total_frames: int):
    import sys
    sys.path.insert(0, "/root/compoxel")
    import torch
    import numpy as np
    import h5py
    import pyflamegpu
    from models import compoxel_builder
    import shutil

    layer_configs = [{"size": 5.0, "dim": 6}, {"size": 15.0, "dim": 2}]
    mac_threshold = 1.0 
    dense_threshold = 1000 
    mesh_res = 10
    cutoff_radius = 2.0  # THE HANDSHAKE RADIUS!

    print("[CLOUD: SIM] Booting Compoxel 1.0 Hybrid Engine...")
    model, star_agent, voxel_agents_dict, layer_totals, mesh_node_agent, global_mesh_dim = compoxel_builder.build_model(layer_configs, mac_threshold, dense_threshold, mesh_res, cutoff_radius)
    sim = pyflamegpu.CUDASimulation(model)

    print("[CLOUD: SIM] Pre-computing Mesh Nodes...")
    mesh_pop = pyflamegpu.AgentVector(mesh_node_agent, global_mesh_dim ** 3)
    cell_size = 5.0 / mesh_res
    for z in range(global_mesh_dim):
        for y in range(global_mesh_dim):
            for x in range(global_mesh_dim):
                idx = x + (y * global_mesh_dim) + (z * global_mesh_dim * global_mesh_dim)
                mesh_pop[idx].setVariableInt("global_x", x)
                mesh_pop[idx].setVariableInt("global_y", y)
                mesh_pop[idx].setVariableInt("global_z", z)
                mesh_pop[idx].setVariableFloat("node_x", -15.0 + (x * cell_size) + (cell_size / 2))
                mesh_pop[idx].setVariableFloat("node_y", -15.0 + (y * cell_size) + (cell_size / 2))
                mesh_pop[idx].setVariableFloat("node_z", -15.0 + (z * cell_size) + (cell_size / 2))
                mesh_pop[idx].setVariableInt("parent_voxel_id", (x // mesh_res) + ((y // mesh_res) * 6) + ((z // mesh_res) * 36))
                mesh_pop[idx].setVariableFloat("density", 0.0)
    sim.setPopulationData(mesh_pop)

    print("[CLOUD: SIM] Spawning Galaxies...")
    g1_pos = torch.tensor([-3.0, 0.0, 0.0]) + torch.randn((2500, 3)) * 0.8
    g1_vel = torch.randn((2500, 3)) * 0.05 + torch.tensor([5.0, 0.5, 0.0]) 
    g2_pos = torch.tensor([3.0, 0.0, 0.0]) + torch.randn((2500, 3)) * 0.8
    g2_vel = torch.randn((2500, 3)) * 0.05 + torch.tensor([-5.0, -0.5, 0.0]) 
    
    positions = torch.cat([g1_pos, g2_pos], dim=0).cuda()
    velocities = torch.cat([g1_vel, g2_vel], dim=0).cuda()
    masses = torch.ones((positions.shape[0], 1)).cuda()
    cpu_data = torch.cat([positions, masses, velocities], dim=1).cpu().numpy()

    star_pop = pyflamegpu.AgentVector(star_agent, len(cpu_data))
    for i in range(len(cpu_data)):
        star_pop[i].setVariableFloat("x", float(cpu_data[i][0])); star_pop[i].setVariableFloat("y", float(cpu_data[i][1])); star_pop[i].setVariableFloat("z", float(cpu_data[i][2]))
        star_pop[i].setVariableFloat("mass", float(cpu_data[i][3]))
        star_pop[i].setVariableFloat("vx", float(cpu_data[i][4])); star_pop[i].setVariableFloat("vy", float(cpu_data[i][5])); star_pop[i].setVariableFloat("vz", float(cpu_data[i][6]))
        star_pop[i].setVariableInt("is_dense", 1) # Force density for testing the Mesh!
    sim.setPopulationData(star_pop)
    
    for layer_num, total_voxels in layer_totals.items():
        pop = pyflamegpu.AgentVector(voxel_agents_dict[layer_num], total_voxels)
        for i in range(total_voxels):
            pop[i].setVariableInt("voxel_id", i); pop[i].setVariableFloat("total_mass", 0.0) 
            if layer_num == 1:
                pop[i].setVariableInt("star_count", 0); pop[i].setVariableInt("is_dense", 0)
        sim.setPopulationData(pop)

    print(f"[CLOUD: SIM] Executing {total_frames} frames of Compoxel 1.0 physics...")
    with h5py.File("/tmp/compoxel_frames.h5", "w") as h5f:
        for frame in range(total_frames):
            sim.step()
            sim.getPopulationData(star_pop)
            
            xs, ys, zs = np.zeros(len(cpu_data), dtype=np.float32), np.zeros(len(cpu_data), dtype=np.float32), np.zeros(len(cpu_data), dtype=np.float32)
            for i in range(len(cpu_data)):
                xs[i] = star_pop[i].getVariableFloat("x")
                ys[i] = star_pop[i].getVariableFloat("y")
                zs[i] = star_pop[i].getVariableFloat("z")
                
            grp = h5f.create_group(f"frame_{frame:04d}")
            grp.create_dataset("x", data=xs); grp.create_dataset("y", data=ys); grp.create_dataset("z", data=zs)
            print(f"Computed Frame {frame}/{total_frames}")

    shutil.move("/tmp/compoxel_frames.h5", "/vol/compoxel_frames.h5")
    shared_volume.commit() 

@app.function(image=blender_image, timeout=600, volumes={"/vol": shared_volume})
def export_to_alembic():
    import subprocess
    shared_volume.reload()
    blender_script = """
import bpy, numpy as np, h5py
with h5py.File("/vol/compoxel_frames.h5", 'r') as f:
    frame_keys = sorted(f.keys())
    base_data = np.stack([np.array(f[frame_keys[0]]['x']), np.array(f[frame_keys[0]]['y']), np.array(f[frame_keys[0]]['z'])], axis=-1).astype(np.float32).ravel()
    mesh = bpy.data.meshes.new("Stars"); mesh.from_pydata(base_data.reshape(-1, 3).tolist(), [], [])
    obj = bpy.data.objects.new("StarsObj", mesh); bpy.context.collection.objects.link(obj)
    sk_basis = obj.shape_key_add(name="Basis")

    for f_idx, key in enumerate(frame_keys):
        frame_data = np.stack([np.array(f[key]['x']), np.array(f[key]['y']), np.array(f[key]['z'])], axis=-1).astype(np.float32).ravel()
        sk = obj.shape_key_add(name=f"Frame_{f_idx}"); sk.data.foreach_set("co", frame_data)
        sk.value = 0.0; sk.keyframe_insert(data_path="value", frame=max(0, f_idx-1))
        sk.value = 1.0; sk.keyframe_insert(data_path="value", frame=f_idx)
        sk.value = 0.0; sk.keyframe_insert(data_path="value", frame=f_idx+1)

    bpy.context.view_layer.objects.active = obj; obj.select_set(True)
    bpy.ops.wm.alembic_export(filepath="/vol/compoxel_stars.abc", start=0, end=len(frame_keys)-1, selected=True, vcolors=False, export_custom_properties=False)
"""
    with open("/tmp/export.py", "w") as f: f.write(blender_script)
    subprocess.run(["/usr/local/blender/blender", "-b", "-P", "/tmp/export.py"], check=True)
    shared_volume.commit()

@app.local_entrypoint()
def main():
    TOTAL_FRAMES = 48
    print("[LOCAL] Starting Compoxel 1.0...")
    run_simulation.remote(TOTAL_FRAMES)
    export_to_alembic.remote()
    print("✅ Success! 'compoxel_stars.abc' is in your Modal cloud volume!")
