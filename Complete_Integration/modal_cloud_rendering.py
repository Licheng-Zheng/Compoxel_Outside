import modal
import subprocess
import os

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
        "mv blender-4.1.0-linux-x64 /usr/local/blender"
    )
    .add_local_dir(".", remote_path="/project") 
)

app = modal.App("compoxel-render-farm")

shared_volume = modal.Volume.from_name("compoxel-storage", create_if_missing=True)

@app.function(image=blender_image, volumes={"/vol": shared_volume})
def prep_volume():
    import shutil
    if os.path.exists("/vol/frames"):
        shutil.rmtree("/vol/frames")
    os.makedirs("/vol/frames", exist_ok=True)
    shared_volume.commit()
    print("[CLOUD] Shared volume cleaned and ready for new frames!")

@app.function(
    image=blender_image, 
    gpu="T4", 
    timeout=1800,           
    max_containers=10,   # FIX 1: Hard-cap at 10 to respect your account quota!
    volumes={"/vol": shared_volume} 
)
def render_frame_batch(start_frame: int, end_frame: int):
    import glob
    
    shared_volume.reload()
    
    blend_files = glob.glob("/project/*.blend")
    if not blend_files:
        raise FileNotFoundError("Could not find any .blend file in the directory!")
    target_blend = blend_files[0]
    
    blender_script = """import bpy
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = '/vol/frames/frame_'
for obj in bpy.data.objects:
    for mod in obj.modifiers:
        if mod.type == 'MESH_SEQUENCE_CACHE' and mod.cache_file:
            mod.cache_file.filepath = '/vol/compoxel_stars.abc'
"""
    with open("/tmp/fix_paths.py", "w") as f:
        f.write(blender_script)
        
    cmd = [
        "xvfb-run",
        "-a",
        "/usr/local/blender/blender",
        "-b",
        target_blend,
        "-P",
        "/tmp/fix_paths.py",
        "-s",
        str(start_frame),
        "-e",
        str(end_frame),
        "-a",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    
    shared_volume.commit() 
    print(f"[CLOUD] Rendered batch {start_frame} to {end_frame} successfully.")

@app.function(image=blender_image, timeout=1200, volumes={"/vol": shared_volume})
def stitch_video() -> bytes:
    print("[CLOUD] All frames generated! Stitching into final MP4...")
    
    shared_volume.reload() 
            
    video_path = "/tmp/compoxel_final_cinematic.mp4"
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "30",
        "-i",
        "/vol/frames/frame_%04d.png",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        video_path,
    ]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    with open(video_path, "rb") as f:
        return f.read()

@app.local_entrypoint()
def main():
    TOTAL_FRAMES = 48
    # FIX 2: Renamed to NUM_BATCHES. (Ensure these variables match your true intent!)
    NUM_BATCHES = 8
    
    print("[LOCAL] Formatting cloud volume for renders...")
    prep_volume.remote()
    
    frames_per_batch = max(1, TOTAL_FRAMES // NUM_BATCHES)
    batches = []
    for i in range(0, TOTAL_FRAMES, frames_per_batch):
        start = i
        end = min(i + frames_per_batch - 1, TOTAL_FRAMES - 1)
        batches.append((start, end))
    
    print(f"[LOCAL] Dispatching {TOTAL_FRAMES} frames across {len(batches)} batches...")
    print(f"[LOCAL] Note: Modal will process {min(10, len(batches))} batches concurrently based on your quota.")
    
    list(render_frame_batch.starmap(batches))
    
    print(f"[LOCAL] Swarm finished! Sending to the stitching node...")
    video_bytes = stitch_video.remote()
    
    with open("compoxel_final_cinematic.mp4", "wb") as f:
        f.write(video_bytes)
        
    print("✅ Render complete! Saved 'compoxel_final_cinematic.mp4' to your local folder.")