import modal
import time
import torch

image = (
    modal.Image.from_registry("nvidia/cuda:12.0.1-devel-ubuntu22.04")
    .apt_install("python-is-python3", "python3-pip")
    .pip_install("torch", "pyflamegpu", "numpy", "h5py", extra_index_url="https://whl.flamegpu.com/whl/cuda120/")
    .env({"CUDA_PATH": "/usr/local/cuda"})
    .add_local_dir(".", remote_path="/root/compoxel") 
)

app = modal.App("compoxel-benchmarks")

@app.function(gpu="T4", timeout=3600, image=image)
def benchmark_algorithm(num_stars: int, algorithm_name: str):
    import sys
    import numpy as np
    sys.path.insert(0, "/root/compoxel")
    import pyflamegpu
    
    torch.manual_seed(42)
    print(f"\n[CLOUD] Spawning N={num_stars} galaxy for {algorithm_name}...")
    
    # Clamp initial positions safely inside the [-15.0, 15.0] spatial bounds
    raw_positions = torch.randn((num_stars, 3)).cuda() * 5.0
    positions = torch.clamp(raw_positions, -13.0, 13.0)
    
    velocities = torch.randn((num_stars, 3)).cuda() * 0.1
    masses = torch.ones((num_stars, 1)).cuda()
    cpu_data = torch.cat([positions, masses, velocities], dim=1).cpu().numpy()

    # ==========================================
    # MODEL BUILDER: NAIVE O(N^2)
    # ==========================================
    if algorithm_name == "Naive":
        model = pyflamegpu.ModelDescription(f"Naive_NBody_{num_stars}")
        env = model.Environment()
        env.newPropertyFloat("dt", 0.05); env.newPropertyFloat("grav_constant", 0.0001)

        bf_msg = model.newMessageBruteForce("GravityBroadcast")
        bf_msg.newVariableFloat("x"); bf_msg.newVariableFloat("y"); bf_msg.newVariableFloat("z"); bf_msg.newVariableFloat("mass")

        star = model.newAgent("StarAgent")
        star.newVariableFloat("x"); star.newVariableFloat("y"); star.newVariableFloat("z")
        star.newVariableFloat("vx"); star.newVariableFloat("vy"); star.newVariableFloat("vz"); star.newVariableFloat("mass")

        bcast_code = """
        FLAMEGPU_AGENT_FUNCTION(broadcast_pos, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
            FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
            FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
            FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
            FLAMEGPU->message_out.setVariable<float>("mass", FLAMEGPU->getVariable<float>("mass"));
            return flamegpu::ALIVE;
        }
        """

        grav_code = """
        FLAMEGPU_AGENT_FUNCTION(calculate_gravity_and_move, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
            float x = FLAMEGPU->getVariable<float>("x"); float y = FLAMEGPU->getVariable<float>("y"); float z = FLAMEGPU->getVariable<float>("z");
            float vx = FLAMEGPU->getVariable<float>("vx"); float vy = FLAMEGPU->getVariable<float>("vy"); float vz = FLAMEGPU->getVariable<float>("vz");
            float G = FLAMEGPU->environment.getProperty<float>("grav_constant");
            float dt = FLAMEGPU->environment.getProperty<float>("dt");
            float fx = 0.0f, fy = 0.0f, fz = 0.0f;
            for (auto &msg : FLAMEGPU->message_in) {
                float ox = msg.getVariable<float>("x"); float oy = msg.getVariable<float>("y"); float oz = msg.getVariable<float>("z");
                if (ox != x || oy != y || oz != z) {
                    float dx = ox - x; float dy = oy - y; float dz = oz - z;
                    float dist_sq = dx*dx + dy*dy + dz*dz + 0.1f; 
                    float inv_dist = rsqrtf(dist_sq);
                    float force = G * msg.getVariable<float>("mass") * inv_dist * inv_dist * inv_dist;
                    fx += dx * force; fy += dy * force; fz += dz * force;
                }
            }
            vx += fx * dt; vy += fy * dt; vz += fz * dt;
            FLAMEGPU->setVariable<float>("vx", vx); FLAMEGPU->setVariable<float>("vy", vy); FLAMEGPU->setVariable<float>("vz", vz);
            FLAMEGPU->setVariable<float>("x", x + (vx * dt)); FLAMEGPU->setVariable<float>("y", y + (vy * dt)); FLAMEGPU->setVariable<float>("z", z + (vz * dt));
            return flamegpu::ALIVE;
        }
        """
        func_bcast = star.newRTCFunction("broadcast_pos", bcast_code)
        func_bcast.setMessageOutput("GravityBroadcast")
        model.newLayer().addAgentFunction(func_bcast)

        func_grav = star.newRTCFunction("calculate_gravity_and_move", grav_code)
        func_grav.setMessageInput("GravityBroadcast")
        model.newLayer().addAgentFunction(func_grav)

        sim = pyflamegpu.CUDASimulation(model)
        star_pop = pyflamegpu.AgentVector(star, num_stars)
        for i in range(num_stars):
            star_pop[i].setVariableFloat("x", float(cpu_data[i][0])); star_pop[i].setVariableFloat("y", float(cpu_data[i][1])); star_pop[i].setVariableFloat("z", float(cpu_data[i][2]))
            star_pop[i].setVariableFloat("mass", float(cpu_data[i][3]))
            star_pop[i].setVariableFloat("vx", float(cpu_data[i][4])); star_pop[i].setVariableFloat("vy", float(cpu_data[i][5])); star_pop[i].setVariableFloat("vz", float(cpu_data[i][6]))
        sim.setPopulationData(star_pop)

    # ==========================================
    # MODEL BUILDER: THE ABLATION STUDY (Tree vs Mesh vs Hybrid)
    # ==========================================
    elif algorithm_name in ["Tree", "Mesh", "Hybrid"]:
        from models import compoxel_builder
        layer_configs = [{"size": 5.0, "dim": 6}, {"size": 15.0, "dim": 2}]
        
        cutoff_rad = 2.0
        force_dense_flag = 1
        
        if algorithm_name == "Tree":
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
    else:
        raise ValueError(f"Unknown Algorithm: {algorithm_name}")

    print(f"[CLOUD] Warming up GPU kernels...")
    sim.step(); sim.step()
    
    print(f"[CLOUD] Running isolated CUDA timing...")
    total_time_ms = 0.0
    num_test_frames = 4
    
    for frame in range(num_test_frames):
        torch.cuda.synchronize() 
        start_time = time.perf_counter()
        
        sim.step()  
        
        torch.cuda.synchronize() 
        end_time = time.perf_counter()
        total_time_ms += (end_time - start_time) * 1000
        
    avg_frame_time = total_time_ms / num_test_frames
    print(f"[CLOUD] {algorithm_name} N={num_stars} completed in {avg_frame_time:.2f} ms/frame")

    # ==========================================
    # FINAL POSITION EXTRACTION (For MSE Calculation)
    # ==========================================
    sim.getPopulationData(star_pop)
    final_pos = np.zeros((num_stars, 3), dtype=np.float32)
    for i in range(num_stars):
        final_pos[i, 0] = star_pop[i].getVariableFloat("x")
        final_pos[i, 1] = star_pop[i].getVariableFloat("y")
        final_pos[i, 2] = star_pop[i].getVariableFloat("z")
    
    return avg_frame_time, final_pos

@app.local_entrypoint()
def main():
    import numpy as np 
    
    # Standard testing loop. Add higher values if you want!
    entity_counts = [5_000_000]
    algorithms = ["Naive", "Mesh", "Hybrid"]
    
    print("=== COMPOXEL SCALABILITY & ACCURACY MATRIX ===")
    
    # 1. Open in Write mode just once to create the file and add the header
    header = "Entity_Count,Naive_ms,Tree_ms,Tree_MSE,Tree_P99,Tree_Max,Mesh_ms,Mesh_MSE,Mesh_P99,Mesh_Max,Hybrid_ms,Hybrid_MSE,Hybrid_P99,Hybrid_Max\n"
    with open("table_1_and_3_results.csv", "w") as f:
        f.write(header)
        
    for count in entity_counts:
        times = {"Naive": "DNF", "Tree": "DNF", "Mesh": "DNF", "Hybrid": "DNF"}
        mses = {"Naive": "0.0", "Tree": "N/A", "Mesh": "N/A", "Hybrid": "N/A"}
        p99s = {"Naive": "0.0", "Tree": "N/A", "Mesh": "N/A", "Hybrid": "N/A"}
        max_errs = {"Naive": "0.0", "Tree": "N/A", "Mesh": "N/A", "Hybrid": "N/A"}
        baseline_pos = None
        
        for algo in algorithms:
            try:
                # Uncomment to skip Naive at high counts to save time!
                # if algo == "Naive" and count > 5_000_000:
                #     continue 
                
                time_ms, final_pos = benchmark_algorithm.remote(count, algo)
                times[algo] = f"{time_ms:.2f}"
                
                # Store Ground Truth or Calculate Errors against it
                if algo == "Naive":
                    baseline_pos = final_pos
                elif baseline_pos is not None:
                    # Calculate the EXACT 3D distance error for every single star
                    errors = np.sqrt(np.sum((final_pos - baseline_pos)**2, axis=1))
                    
                    mse = np.mean(errors**2)
                    p99 = np.percentile(errors, 99)
                    max_err = np.max(errors)
                    
                    mses[algo] = f"{mse:.6f}"
                    p99s[algo] = f"{p99:.6f}"
                    max_errs[algo] = f"{max_err:.6f}"
                    
            except Exception as e:
                print(f"[LOCAL] {algo} failed at {count} entities. Error: {e}")
                
        log_line = (f"{count},{times['Naive']},"
                    f"{times['Tree']},{mses['Tree']},{p99s['Tree']},{max_errs['Tree']},"
                    f"{times['Mesh']},{mses['Mesh']},{p99s['Mesh']},{max_errs['Mesh']},"
                    f"{times['Hybrid']},{mses['Hybrid']},{p99s['Hybrid']},{max_errs['Hybrid']}")
        
        print(f"RESULT LOGGED: {log_line}")
        
        # 2. Open in Append mode to safely save the row to the disk instantly
        with open("table_1_and_3_results.csv", "a") as f:
            f.write(log_line + "\n")
            
    print("\n✅ Physics Profiling complete! Check the CSV for those sweet P99 and Max Error rates.")