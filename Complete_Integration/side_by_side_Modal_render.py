import os

# Set the Modal authentication tokens before importing modal
os.environ["MODAL_TOKEN_ID"] = "ak-BEr1WywrgeeQX3BN4tqyvJ"
os.environ["MODAL_TOKEN_SECRET"] = "as-mssLXPuHWtZU8Ma5PBD6Um"

import modal
import subprocess

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
    
    # Clean up old frame renders to prevent ghost frames in the new video
    if os.path.exists("/vol/frames"):
        shutil.rmtree("/vol/frames")
    os.makedirs("/vol/frames", exist_ok=True)
    
    shared_volume.commit()
    print("[CLOUD] Shared volume cleaned and ready for new frames!")

@app.function(
    image=blender_image, 
    gpu="T4", 
    timeout=1800,           
    max_containers=10,   
    volumes={"/vol": shared_volume} 
)
def render_frame_batch(start_frame: int, end_frame: int, algo: str):
    import glob
    
    shared_volume.reload()
    
    blend_files = glob.glob("/project/*.blend")
    if not blend_files:
        raise FileNotFoundError("Could not find any .blend file in the directory!")
    target_blend = blend_files[0]
    
    # We are injecting a script that instances physical geometry onto the math points
    blender_script = f"""import bpy
import os

bpy.context.scene.render.image_settings.file_format = 'PNG'

# FIX 1: subdivisions=0 drops the vertex count from 21 Million down to 6 Million!
bpy.ops.mesh.primitive_ico_sphere_add(radius=0.03, subdivisions=0, location=(0, 0, 0))
star_template = bpy.context.active_object
star_template.name = 'StarTemplate'

# Apply the original glowing material from your .blend file to the star
if len(bpy.data.materials) > 0:
    star_template.data.materials.append(bpy.data.materials[0])

start_frame = {start_frame}
end_frame = {end_frame}
algo = "{algo.capitalize()}"

for frame in range(start_frame, end_frame + 1):
    print(f"\\n[BLENDER] Preparing Frame {{frame}}...")
    bpy.context.scene.frame_set(frame)
    bpy.context.scene.render.filepath = f'/vol/frames/frame_{{frame:04d}}.png'
    
    ply_path = f"/vol/{{algo}}/frame_{{frame:04d}}.ply"
    
    # FIX 2: Safely detach the template from the old point cloud before deleting it
    star_template.parent = None
    
    # 1. BULLETPROOF DELETE: Remove previous PLY imports (but protect the StarTemplate!)
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name != 'StarTemplate':
            bpy.data.objects.remove(obj, do_unlink=True)
            
    # Clean up orphan mesh memory
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
            
    # 2. IMPORT NEW RAW MATH POINTS
    if os.path.exists(ply_path):
        print(f"[BLENDER] Importing {{ply_path}}...")
        bpy.ops.wm.ply_import(filepath=ply_path)
        ply_obj = bpy.context.active_object
        
        # 3. RENDER THE POINTS AS PHYSICAL STARS
        ply_obj.instance_type = 'VERTS'
        
        # Parent the sphere to the point cloud to trigger the instancer
        star_template.parent = ply_obj
        
        # Hide the raw instancer math from the render engine
        ply_obj.show_instancer_for_render = False 
        
        print(f"[BLENDER] Rendering Frame {{frame}}...")
        # 4. EXPLICITLY RENDER THE FRAME
        bpy.ops.render.render(write_still=True)
        print(f"[BLENDER] Frame {{frame}} complete!")
    else:
        print(f"WARNING: Could not find file {{ply_path}}")

"""
    
    with open("/tmp/fix_paths.py", "w") as f:
        f.write(blender_script)
        
    cmd = f"xvfb-run -a /usr/local/blender/blender -b {target_blend} -P /tmp/fix_paths.py"
    # FIX 3: Removed 'capture_output=True' so Blender's progress streams live to your terminal!
    subprocess.run(cmd, shell=True, check=True)
    
    shared_volume.commit() 
    print(f"[CLOUD] Rendered {algo.capitalize()} batch {start_frame} to {end_frame} successfully.")

@app.function(image=blender_image, timeout=1200, volumes={"/vol": shared_volume})
def stitch_video(algo: str) -> bytes:
    print(f"[CLOUD] All frames generated! Stitching into final {algo} MP4...")
    
    shared_volume.reload() 
            
    video_path = f"/tmp/compoxel_{algo}_render.mp4"
    ffmpeg_cmd = f"ffmpeg -y -framerate 30 -i '/vol/frames/frame_%04d.png' -c:v libx264 -pix_fmt yuv420p {video_path}"
    subprocess.run(ffmpeg_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    with open(video_path, "rb") as f:
        return f.read()

@app.local_entrypoint()
def main(algo: str = "naive"):
    algo = algo.lower()
    valid_algos = ["naive", "tree", "mesh", "hybrid"]
    
    if algo not in valid_algos:
        print(f"❌ Error: Invalid algorithm '{algo}'. Must be one of: {valid_algos}")
        return

    # Scaling configuration: 800 frames / 40 batches = 20 frames per T4 container
    TOTAL_FRAMES = 60
    NUM_BATCHES = 10  
    
    print(f"[LOCAL] Formatting cloud volume for the {algo.upper()} render...")
    prep_volume.remote()
    
    frames_per_batch = max(1, TOTAL_FRAMES // NUM_BATCHES)
    batches = []
    for i in range(0, TOTAL_FRAMES, frames_per_batch):
        start = i
        end = min(i + frames_per_batch - 1, TOTAL_FRAMES - 1)
        batches.append((start, end, algo)) 
    
    print(f"[LOCAL] Dispatching {TOTAL_FRAMES} frames across {len(batches)} batches...")
    
    # Fire the distributed compute swarm
    list(render_frame_batch.starmap(batches))
    
    print(f"[LOCAL] Swarm finished! Sending to the stitching node...")
    video_bytes = stitch_video.remote(algo)
    
    output_filename = f"compoxel_{algo}_render.mp4"
    with open(output_filename, "wb") as f:
        f.write(video_bytes)
        
    print(f"✅ Render complete! Saved '{output_filename}' to your local folder.")