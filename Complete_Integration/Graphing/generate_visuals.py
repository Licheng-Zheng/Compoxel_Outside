import modal
import os
import io
import tarfile
import subprocess

# ---------------------------------------------------------------------------
# 1. CONTAINER IMAGE DEFINITIONS
# ---------------------------------------------------------------------------

# Physics Simulation Environment (PyTorch + PyFlameGPU + CUDA)
sim_image = (
    modal.Image.from_registry("nvidia/cuda:12.0.1-devel-ubuntu22.04")
    .apt_install("python-is-python3", "python3-pip")
    .pip_install("torch", "pyflamegpu", "numpy", "h5py", extra_index_url="https://whl.flamegpu.com/whl/cuda120/")
    .env({"CUDA_PATH": "/usr/local/cuda"})
    .add_local_dir(".", remote_path="/root/compoxel", ignore=["*.pyc", "__pycache__", "*.h5", "*.mp4", "*.abc", "frames/", "compoxel_storage/"]) 
)

# Graphics & Video Environment (Blender + FFmpeg + Virtual Framebuffer)
blender_image = (
    modal.Image.debian_slim()
    .apt_install(
        "wget", "xz-utils", "xvfb", "ffmpeg", 
        "libxrender1", "libxi6", "libxkbcommon0", "libgl1-mesa-glx", 
        "libsm6", "libice6", "libxext6", "libx11-6",
        "libegl1", "libopengl0", "libx11-xcb1" 
    )
    .run_commands(
        "wget https://download.blender.org/release/Blender4.1/blender-4.1.0-linux-x64.tar.xz",
        "tar -xf blender-4.1.0-linux-x64.tar.xz",
        "mv blender-4.1.0-linux-x64 /usr/local/blender",
        "/usr/local/blender/blender -b --python-expr \"import subprocess, sys; subprocess.run([sys.executable, '-m', 'ensurepip']); subprocess.run([sys.executable, '-m', 'pip', 'install', 'h5py', 'numpy'])\""
    )
    .add_local_dir(".", remote_path="/project", ignore=["*.pyc", "__pycache__", "*.h5", "*.mp4", "*.abc", "frames/", "compoxel_storage/"]) 
)

# ---------------------------------------------------------------------------
# 2. APP & STORAGE SETUP
# ---------------------------------------------------------------------------
app = modal.App("compoxel-unified-pipeline")
shared_volume = modal.Volume.from_name("compoxel-storage", create_if_missing=True)

# ---------------------------------------------------------------------------
# 3. REMOTE FUNCTIONS
# ---------------------------------------------------------------------------

@app.function(gpu="T4", timeout=1800, image=sim_image)
def bake_physics(algorithm_name: str, num_stars: int) -> bytes:
    import torch  
    import sys
    import numpy as np
    import h5py
    sys.path.insert(0, "/root/compoxel")
    import pyflamegpu
    from models import compoxel_builder

    torch.manual_seed(80)
    print(f"\n[CLOUD: SIM] Spawning N={num_stars} galaxy for {algorithm_name}...")
    
    raw_positions = torch.randn((num_stars, 3)).cuda() * 5.0
    positions = torch.clamp(raw_positions, -13.0, 13.0)
    velocities = torch.randn((num_stars, 3)).cuda() * 0.1
    masses = torch.ones((num_stars, 1)).cuda()
    cpu_data = torch.cat([positions, masses, velocities], dim=1).cpu().numpy()

    layer_configs = [{"size": 5.0, "dim": 6}, {"size": 15.0, "dim": 2}]
    cutoff_rad = 2.0
    force_dense_flag = 1
    
    if algorithm_name == "Naive":
        force_dense_flag = 1
    elif algorithm_name == "Tree":
        force_dense_flag = 0 
    elif algorithm_name == "Mesh":
        cutoff_rad = 0.25  

    model, star_agent, voxel_agents_dict, layer_totals, mesh_node_agent, global_mesh_dim = compoxel_builder.build_model(
        layer_configs, mac_threshold=1.0, dense_threshold=1000, mesh_resolution=10, cutoff_radius=cutoff_rad
    )
    sim = pyflamegpu.CUDASimulation(model)
    
    mesh_pop = pyflamegpu.AgentVector(mesh_node_agent, global_mesh_dim ** 3)
    cell_size = 5.0 / 10
    for z in range(global_mesh_dim):
        for y in range(global_mesh_dim):
            for x in range(global_mesh_dim):
                idx = x + (y * global_mesh_dim) + (z * global_mesh_dim * global_mesh_dim)
                mesh_pop[idx].setVariableInt("global_x", x); mesh_pop[idx].setVariableInt("global_y", y); mesh_pop[idx].setVariableInt("global_z", z)
                mesh_pop[idx].setVariableFloat("node_x", -15.0 + (x * cell_size) + (cell_size / 2))
                mesh_pop[idx].setVariableFloat("node_y", -15.0 + (y * cell_size) + (cell_size / 2))
                mesh_pop[idx].setVariableFloat("node_z", -15.0 + (z * cell_size) + (cell_size / 2))
                mesh_pop[idx].setVariableInt("parent_voxel_id", (x // 10) + ((y // 10) * 6) + ((z // 10) * 36))
                mesh_pop[idx].setVariableFloat("density", 0.0)
    sim.setPopulationData(mesh_pop)

    star_pop = pyflamegpu.AgentVector(star_agent, num_stars)
    for i in range(num_stars):
        star_pop[i].setVariableFloat("x", float(cpu_data[i][0])); star_pop[i].setVariableFloat("y", float(cpu_data[i][1])); star_pop[i].setVariableFloat("z", float(cpu_data[i][2]))
        star_pop[i].setVariableFloat("mass", float(cpu_data[i][3]))
        star_pop[i].setVariableFloat("vx", float(cpu_data[i][4])); star_pop[i].setVariableFloat("vy", float(cpu_data[i][5])); star_pop[i].setVariableFloat("vz", float(cpu_data[i][6]))
        star_pop[i].setVariableInt("is_dense", force_dense_flag) 
    sim.setPopulationData(star_pop)
    
    for layer_num, total_voxels in layer_totals.items():
        pop = pyflamegpu.AgentVector(voxel_agents_dict[layer_num], total_voxels)
        for i in range(total_voxels):
            pop[i].setVariableInt("voxel_id", i); pop[i].setVariableFloat("total_mass", 0.0) 
            if layer_num == 1:
                pop[i].setVariableInt("star_count", 0); pop[i].setVariableInt("is_dense", 0)
        sim.setPopulationData(pop)

    print(f"[CLOUD: SIM] Running 60 cinematic iterations...")
    os.makedirs("/tmp/h5_frames", exist_ok=True)
    
    for frame in range(60):
        sim.step()
        sim.getPopulationData(star_pop)
        
        xs = np.zeros(num_stars, dtype=np.float32)
        ys = np.zeros(num_stars, dtype=np.float32)
        zs = np.zeros(num_stars, dtype=np.float32)
        
        for i in range(num_stars):
            xs[i] = star_pop[i].getVariableFloat("x")
            ys[i] = star_pop[i].getVariableFloat("y")
            zs[i] = star_pop[i].getVariableFloat("z")
            
        with h5py.File(f"/tmp/h5_frames/frame_{frame:04d}.h5", "w") as h5f:
            h5f.create_dataset("x", data=xs)
            h5f.create_dataset("y", data=ys)
            h5f.create_dataset("z", data=zs)
        
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w:gz") as tar:
        tar.add("/tmp/h5_frames", arcname="h5_frames")
    return bio.getvalue()

@app.function(image=blender_image, timeout=1800, volumes={"/vol": shared_volume})
def export_to_alembic_volume(tar_bytes: bytes, algo: str, count_str: str):
    import shutil
    print(f"[CLOUD: ABC] Compiling HDF5 binary stream into Volume cache...")
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        tar.extractall(path="/tmp/")

    # Clean and isolate frame render directories directly inside the volume
    frame_dir = f"/vol/frames/{algo}_{count_str}"
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    blender_script = """
import bpy, numpy as np, h5py, glob, os

frame_files = sorted(glob.glob("/tmp/h5_frames/*.h5"))
num_frames = len(frame_files)
all_frame_data = []

for h5_file in frame_files:
    with h5py.File(h5_file, 'r') as f:
        x = np.array(f['x'])
        y = np.array(f['y'])
        z = np.array(f['z'])
        all_frame_data.append(np.stack([x, y, z], axis=-1))

data = np.array(all_frame_data)
mesh = bpy.data.meshes.new("Stars")
mesh.from_pydata(data[0].tolist(), [], [])
obj = bpy.data.objects.new("StarsObj", mesh)
bpy.context.collection.objects.link(obj)

sk_basis = obj.shape_key_add(name="Basis")
for f in range(num_frames):
    sk = obj.shape_key_add(name=f"Frame_{f}")
    sk.data.foreach_set("co", data[f].astype(np.float32).ravel())
    sk.value = 0.0; sk.keyframe_insert(data_path="value", frame=f-1)
    sk.value = 1.0; sk.keyframe_insert(data_path="value", frame=f)
    sk.value = 0.0; sk.keyframe_insert(data_path="value", frame=f+1)

bpy.context.view_layer.objects.active = obj; obj.select_set(True)
bpy.ops.wm.alembic_export(filepath="/vol/current.abc", start=0, end=num_frames-1, selected=True)
"""
    with open("/tmp/export.py", "w") as f: f.write(blender_script)
    subprocess.run(["/usr/local/blender/blender", "-b", "-P", "/tmp/export.py"], check=True, stdout=subprocess.DEVNULL)

    shared_volume.commit()
    print(f"[CLOUD: ABC] Intermediate Alembic cache successfully saved to Modal Volume.")

@app.function(image=blender_image, gpu="T4", timeout=1800, max_containers=10, volumes={"/vol": shared_volume})
def render_frame_batch(start_frame: int, end_frame: int, algo: str, count_str: str):
    import glob
    shared_volume.reload()
    
    blend_files = glob.glob("/project/*.blend")
    if not blend_files:
        raise FileNotFoundError("Could not find any template .blend file in the directory!")
    target_blend = blend_files[0]
    
    blender_script = f"""import bpy
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = '/vol/frames/{algo}_{count_str}/frame_'

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.preferences.addons['cycles'].preferences.get_devices()
for d in bpy.context.preferences.addons['cycles'].preferences.devices:
    d.use = True

for obj in bpy.data.objects:
    for mod in obj.modifiers:
        if mod.type == 'MESH_SEQUENCE_CACHE' and mod.cache_file:
            mod.cache_file.filepath = '/vol/current.abc'
"""
    with open("/tmp/fix_paths.py", "w") as f:
        f.write(blender_script)
        
    cmd = ["xvfb-run", "-a", "/usr/local/blender/blender", "-b", target_blend, "-P", "/tmp/fix_paths.py", "-s", str(start_frame), "-e", str(end_frame), "-a"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    
    shared_volume.commit() 
    print(f"[CLOUD: RENDER] Finished batch frames {start_frame} to {end_frame}")

@app.function(image=blender_image, timeout=1800, volumes={"/vol": shared_volume})
def stitch_video(algo: str, count_str: str) -> bytes:
    shared_volume.reload() 
    video_path = f"/tmp/compoxel_{algo}_{count_str}.mp4"
    
    ffmpeg_cmd = ["ffmpeg", "-y", "-framerate", "30", "-i", f"/vol/frames/{algo}_{count_str}/frame_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    with open(video_path, "rb") as f: 
        return f.read()

# ---------------------------------------------------------------------------
# 4. ORCHESTRATION PIPELINE (LOCAL ENTRYPOINT)
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    print("=== STARTING UNIFIED COMPOXEL PIPELINE ===")
    
    visual_algos = ["Naive", "Tree", "Mesh", "Hybrid"]
    star_counts = [100_000]
    
    TOTAL_FRAMES = 180
    NUM_BATCHES = 10
    run_summary = []

    for count in star_counts:
        count_str = f"{count / 1_000_000:.1f}m".replace(".0", "") if count >= 1_000_000 else f"{count // 1000}k"
        
        for algo in visual_algos:
            algo_lower = algo.lower()
            mp4_filename = f"compoxel_{algo_lower}_{count_str}.mp4"
            
            print(f"\n=======================================================")
            print(f" PROCESSING: {algo.upper()} @ {count_str.upper()} ENTITIES")
            print(f"=======================================================")
            
            try:
                # STEP 1: Compute physics simulation on GPU
                tar_bytes = bake_physics.remote(algo, num_stars=count)
                
                # STEP 2: Transpile results directly into the Cloud Volume (No local download)
                export_to_alembic_volume.remote(tar_bytes, algo_lower, count_str)
                
                # STEP 3: Setup render chunks for Map distribution
                frames_per_batch = max(1, TOTAL_FRAMES // NUM_BATCHES)
                batches = []
                for i in range(0, TOTAL_FRAMES, frames_per_batch):
                    start = i
                    end = min(i + frames_per_batch - 1, TOTAL_FRAMES - 1)
                    batches.append((start, end, algo_lower, count_str))
                
                print(f"[LOCAL] Spawning parallel cloud containers for frame distribution...")
                list(render_frame_batch.starmap(batches))
                
                # STEP 4: Stitch frames on the cloud node and download final MP4 video
                print(f"[LOCAL] Compressing images into structural high-fidelity MP4 wrapper...")
                video_bytes = stitch_video.remote(algo_lower, count_str)
                
                with open(mp4_filename, "wb") as f:
                    f.write(video_bytes)
                print(f"🎬 Success! Render complete and downloaded directly to: '{mp4_filename}'")
                
                run_summary.append((count_str, algo, "SUCCESS"))
                
            except Exception as e:
                print(f"❌ [DNF] Failed pipeline chain sequence for {algo} at {count_str}.")
                print(f"Error Diagnostic: {str(e)[:140]}...")
                run_summary.append((count_str, algo, "DNF"))
                continue

    # Final Dashboard Print
    print("\n==============================================")
    print("      END-TO-END AUTOMATION SUMMARY ROUTE      ")
    print("==============================================")
    print(f"{'Entities':<10} | {'Algorithm':<12} | {'Pipeline Status':<15}")
    print("-" * 46)
    for count_str, algo, status in run_summary:
        status_icon = "🟢" if status == "SUCCESS" else "🔴"
        print(f"{count_str:<10} | {algo:<12} | {status_icon} {status}")
    print("==============================================")