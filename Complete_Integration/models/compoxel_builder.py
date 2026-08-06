# models/compoxel_builder.py
import pyflamegpu

def build_model(layer_configs, mac_threshold=1.0, dense_threshold=1000, mesh_resolution=10, cutoff_radius=1.5):
    num_layers = len(layer_configs)
    model = pyflamegpu.ModelDescription(f"Compoxel_V1_Hybrid_TreePM")

    env = model.Environment()
    env.newPropertyFloat("env_min_bound", -15.0) 
    env.newPropertyFloat("mac_threshold", mac_threshold) 
    env.newPropertyFloat("dt", 0.05) 
    env.newPropertyFloat("grav_constant", 0.0001) 
    
    env.newPropertyInt("dense_entity_threshold", dense_threshold)
    env.newPropertyInt("mesh_resolution", mesh_resolution)
    env.newPropertyFloat("cutoff_radius", cutoff_radius) 

    l1_voxel_size = layer_configs[0]["size"]
    l1_grid_dim = layer_configs[0]["dim"]
    mesh_cell_size = l1_voxel_size / mesh_resolution
    env.newPropertyFloat("mesh_cell_size", mesh_cell_size)

    with open(r"/root/compoxel/kernels/compoxel_v1.cu", "r") as f:
        content = f.read()

    blocks = {}
    parts = content.split("// [")
    for part in parts[1:]:
        header_end = part.find("]")
        blocks[part[:header_end]] = part[header_end+1:].strip()

    voxel_agents = {}
    layer_totals = {}

    for i, config in enumerate(layer_configs):
        layer_num = i + 1
        v_size = config["size"]
        grid_dim = config["dim"]
        total_voxels = grid_dim ** 3

        env.newPropertyFloat(f"l{layer_num}_voxel_size", v_size)
        env.newPropertyInt(f"l{layer_num}_grid_dim", grid_dim)
        layer_totals[layer_num] = total_voxels

        bucket_msg = model.newMessageBucket(f"VoxelBucket_L{layer_num}")
        bucket_msg.setUpperBound(total_voxels)
        bucket_msg.newVariableFloat("x"); bucket_msg.newVariableFloat("y"); bucket_msg.newVariableFloat("z"); bucket_msg.newVariableFloat("mass")

        bf_msg = model.newMessageBruteForce(f"BruteForce_L{layer_num}")
        bf_msg.newVariableFloat("x"); bf_msg.newVariableFloat("y"); bf_msg.newVariableFloat("z"); bf_msg.newVariableFloat("mass")
        bf_msg.newVariableInt("voxel_id"); bf_msg.newVariableInt("parent_id")

        voxel_agent = model.newAgent(f"VoxelAgent_L{layer_num}")
        voxel_agent.newVariableInt("voxel_id"); voxel_agent.newVariableFloat("total_mass")
        voxel_agent.newVariableFloat("com_x"); voxel_agent.newVariableFloat("com_y"); voxel_agent.newVariableFloat("com_z")
        voxel_agent.newVariableInt("star_count"); voxel_agent.newVariableInt("is_dense")
        voxel_agents[layer_num] = voxel_agent

    global_mesh_dim = l1_grid_dim * mesh_resolution
    env.newPropertyInt("global_mesh_dim", global_mesh_dim)

    # --- MESSAGES ---
    dense_star_msg = model.newMessageBucket("DenseStarBucket")
    dense_star_msg.setUpperBound(global_mesh_dim ** 3) 
    dense_star_msg.newVariableFloat("x"); dense_star_msg.newVariableFloat("y"); dense_star_msg.newVariableFloat("z"); dense_star_msg.newVariableFloat("mass")

    mesh_density_msg = model.newMessageArray3D("MeshDensityArray")
    mesh_density_msg.setDimensions(global_mesh_dim, global_mesh_dim, global_mesh_dim)
    mesh_density_msg.newVariableFloat("density")

    force_grid_msg = model.newMessageArray3D("ForceGridArray")
    force_grid_msg.setDimensions(global_mesh_dim, global_mesh_dim, global_mesh_dim)
    force_grid_msg.newVariableFloat("force_x"); force_grid_msg.newVariableFloat("force_y"); force_grid_msg.newVariableFloat("force_z")

    # PHASE 4: The Spatial Radar Message
    spatial_msg = model.newMessageSpatial3D("SpatialRadar")
    spatial_msg.setRadius(cutoff_radius) 
    spatial_msg.setMin(-15.0, -15.0, -15.0)
    spatial_msg.setMax(15.0, 15.0, 15.0)
    # FIX: FLAME GPU automatically creates x, y, and z for Spatial3D messages! We only need to add mass.
    spatial_msg.newVariableFloat("mass")

    # --- AGENTS ---
    mesh_node = model.newAgent("MeshNode")
    mesh_node.newVariableInt("parent_voxel_id")
    mesh_node.newVariableInt("global_x"); mesh_node.newVariableInt("global_y"); mesh_node.newVariableInt("global_z")
    mesh_node.newVariableFloat("node_x"); mesh_node.newVariableFloat("node_y"); mesh_node.newVariableFloat("node_z"); mesh_node.newVariableFloat("density")

    star = model.newAgent("StarAgent")
    star.newVariableFloat("x"); star.newVariableFloat("y"); star.newVariableFloat("z"); star.newVariableFloat("mass")
    star.newVariableFloat("vx"); star.newVariableFloat("vy"); star.newVariableFloat("vz")
    star.newVariableInt("is_dense") 

    # --- EXECUTION LAYERS ---
    
    # 1. Macro TreePM Setup
    func_star_to_l1 = star.newRTCFunction("star_to_voxel_output", f"{blocks['HELPER_BLOCK']}\n{blocks['STAR_TO_L1_BLOCK']}")
    func_star_to_l1.setMessageOutput("VoxelBucket_L1")
    model.newLayer().addAgentFunction(func_star_to_l1)

    for current_layer in range(1, num_layers + 1):
        func_agg = voxel_agents[current_layer].newRTCFunction(f"voxel_l{current_layer}_aggregate", blocks['AGGREGATE_TEMPLATE'].replace("{CURRENT}", str(current_layer)))
        func_agg.setMessageInput(f"VoxelBucket_L{current_layer}")
        model.newLayer().addAgentFunction(func_agg)

        parent_calc = ""
        if current_layer < num_layers:
            parent_calc = f"""
            float min_bound = FLAMEGPU->environment.getProperty<float>("env_min_bound");
            float next_size = FLAMEGPU->environment.getProperty<float>("l{current_layer+1}_voxel_size");
            int next_dim = FLAMEGPU->environment.getProperty<int>("l{current_layer+1}_grid_dim");
            parent_id = calculate_voxel_hash(FLAMEGPU->getVariable<float>("com_x"), FLAMEGPU->getVariable<float>("com_y"), FLAMEGPU->getVariable<float>("com_z"), min_bound, next_size, next_dim);
            """
        bcast_code = blocks['VOXEL_BROADCAST_TEMPLATE'].replace("{CURRENT}", str(current_layer)).replace("{PARENT_CALCULATION}", parent_calc)
        func_bcast = voxel_agents[current_layer].newRTCFunction(f"voxel_l{current_layer}_broadcast", f"{blocks['HELPER_BLOCK']}\n{bcast_code}")
        func_bcast.setMessageOutput(f"BruteForce_L{current_layer}")
        model.newLayer().addAgentFunction(func_bcast)

        if current_layer < num_layers:
            step_code = blocks['STEP_UP_TEMPLATE'].replace("{CURRENT}", str(current_layer)).replace("{NEXT}", str(current_layer + 1))
            func_step = voxel_agents[current_layer].newRTCFunction(f"voxel_l{current_layer}_to_l{current_layer+1}_output", f"{blocks['HELPER_BLOCK']}\n{step_code}")
            func_step.setMessageOutput(f"VoxelBucket_L{current_layer+1}")
            model.newLayer().addAgentFunction(func_step)

    for current_layer in range(num_layers, 0, -1):
        if current_layer == num_layers:
            grav_code = blocks['GRAVITY_TOP_LAYER_TEMPLATE'].replace("{CURRENT}", str(current_layer))
        else:
            grav_code = blocks['GRAVITY_MID_LAYER_TEMPLATE'].replace("{CURRENT}", str(current_layer)).replace("{NEXT}", str(current_layer + 1))
            
        func_grav = star.newRTCFunction(f"star_gravity_l{current_layer}", f"{blocks['HELPER_BLOCK']}\n{grav_code}")
        func_grav.setMessageInput(f"BruteForce_L{current_layer}")
        model.newLayer().addAgentFunction(func_grav)

    func_grav_stars = star.newRTCFunction("star_gravity_stars", f"{blocks['HELPER_BLOCK']}\n{blocks['GRAVITY_STARS_TEMPLATE']}")
    func_grav_stars.setMessageInput("VoxelBucket_L1")
    model.newLayer().addAgentFunction(func_grav_stars)

    # 2. Fluid Mesh Pass
    func_star_mesh_out = star.newRTCFunction("star_to_mesh_bucket", f"{blocks['HELPER_BLOCK']}\n{blocks['STAR_TO_MESH_BLOCK']}")
    func_star_mesh_out.setMessageOutput("DenseStarBucket")
    model.newLayer().addAgentFunction(func_star_mesh_out)

    func_cic_gather = mesh_node.newRTCFunction("mesh_node_cic_gather", f"{blocks['HELPER_BLOCK']}\n{blocks['MESH_GATHER_BLOCK']}")
    func_cic_gather.setMessageInput("DenseStarBucket")
    model.newLayer().addAgentFunction(func_cic_gather)

    func_mesh_bcast = mesh_node.newRTCFunction("mesh_node_broadcast", f"{blocks['HELPER_BLOCK']}\n{blocks['MESH_BROADCAST_BLOCK']}")
    func_mesh_bcast.setMessageOutput("MeshDensityArray")
    model.newLayer().addAgentFunction(func_mesh_bcast)

    func_mesh_solver = mesh_node.newRTCFunction("mesh_node_force_solver", f"{blocks['HELPER_BLOCK']}\n{blocks['MESH_SOLVER_BLOCK']}")
    func_mesh_solver.setMessageInput("MeshDensityArray")
    func_mesh_solver.setMessageOutput("ForceGridArray")
    model.newLayer().addAgentFunction(func_mesh_solver)

    func_star_gather = star.newRTCFunction("star_mesh_gather", f"{blocks['HELPER_BLOCK']}\n{blocks['STAR_GATHER_BLOCK']}")
    func_star_gather.setMessageInput("ForceGridArray")
    model.newLayer().addAgentFunction(func_star_gather)

    # 3. P³M Micro Handshake Pass
    func_radar_out = star.newRTCFunction("broadcast_radar", blocks['RADAR_BROADCAST_BLOCK'])
    func_radar_out.setMessageOutput("SpatialRadar")
    model.newLayer().addAgentFunction(func_radar_out)

    func_handshake = star.newRTCFunction("star_p3m_handshake", blocks['P3M_HANDSHAKE_BLOCK'])
    func_handshake.setMessageInput("SpatialRadar")
    model.newLayer().addAgentFunction(func_handshake)

    # 4. Final Kinematic Integration
    func_integrate = star.newRTCFunction("star_integration", blocks['INTEGRATION_BLOCK'])
    model.newLayer().addAgentFunction(func_integrate)

    return model, star, voxel_agents, layer_totals, mesh_node, global_mesh_dim