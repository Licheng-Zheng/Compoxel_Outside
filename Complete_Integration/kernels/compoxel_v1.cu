// kernels/compoxel_v1.cu

// [HELPER_BLOCK]
__device__ int calculate_voxel_hash(float x, float y, float z, float min_bound, float voxel_size, int grid_dim) {
    int idx_x = floorf((x - min_bound) / voxel_size);
    int idx_y = floorf((y - min_bound) / voxel_size);
    int idx_z = floorf((z - min_bound) / voxel_size);
    idx_x = max(0, min(idx_x, grid_dim - 1));
    idx_y = max(0, min(idx_y, grid_dim - 1));
    idx_z = max(0, min(idx_z, grid_dim - 1));
    return idx_x + (idx_y * grid_dim) + (idx_z * grid_dim * grid_dim);
}

__device__ bool check_mac(float px, float py, float pz, int voxel_id, float voxel_size, int grid_dim, float min_bound, float threshold) {
    int z = voxel_id / (grid_dim * grid_dim);
    int y = (voxel_id / grid_dim) % grid_dim;
    int x = voxel_id % grid_dim;
    
    float cx = min_bound + (x * voxel_size) + (voxel_size * 0.5f);
    float cy = min_bound + (y * voxel_size) + (voxel_size * 0.5f);
    float cz = min_bound + (z * voxel_size) + (voxel_size * 0.5f);
    
    float dx = cx - px; float dy = cy - py; float dz = cz - pz;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    return dist > (threshold * voxel_size);
}

__device__ void apply_gravity(float &vx, float &vy, float &vz, float px, float py, float pz, float ox, float oy, float oz, float omass, float dt, float grav_constant) {
    float dx = ox - px; float dy = oy - py; float dz = oz - pz;
    float dist_sq = dx*dx + dy*dy + dz*dz + 0.1f; 
    float inv_dist = rsqrtf(dist_sq);
    float force = grav_constant * omass * inv_dist * inv_dist * inv_dist;
    vx += dx * force; vy += dy * force; vz += dz * force;
}

// ==========================================
// PHASE 2: TREE-PM MACRO LOGIC
// ==========================================

// [STAR_TO_L1_BLOCK]
FLAMEGPU_AGENT_FUNCTION(star_to_voxel_output, flamegpu::MessageNone, flamegpu::MessageBucket) {
    float x = FLAMEGPU->getVariable<float>("x"); float y = FLAMEGPU->getVariable<float>("y"); float z = FLAMEGPU->getVariable<float>("z");
    float min_bound = FLAMEGPU->environment.getProperty<float>("env_min_bound");
    float voxel_size = FLAMEGPU->environment.getProperty<float>("l1_voxel_size");
    int grid_dim = FLAMEGPU->environment.getProperty<int>("l1_grid_dim");

    int bucket_id = calculate_voxel_hash(x, y, z, min_bound, voxel_size, grid_dim);

    FLAMEGPU->message_out.setVariable<float>("x", x); FLAMEGPU->message_out.setVariable<float>("y", y); FLAMEGPU->message_out.setVariable<float>("z", z);
    FLAMEGPU->message_out.setVariable<float>("mass", FLAMEGPU->getVariable<float>("mass"));
    FLAMEGPU->message_out.setKey(bucket_id);
    return flamegpu::ALIVE;
}

// [AGGREGATE_TEMPLATE]
FLAMEGPU_AGENT_FUNCTION(voxel_l{CURRENT}_aggregate, flamegpu::MessageBucket, flamegpu::MessageNone) {
    int my_bucket_id = FLAMEGPU->getVariable<int>("voxel_id");
    float total_mass = 0.0f, weighted_x = 0.0f, weighted_y = 0.0f, weighted_z = 0.0f;
    int star_count = 0; 

    for (auto &msg : FLAMEGPU->message_in(my_bucket_id)) {
        float m = msg.getVariable<float>("mass");
        total_mass += m;
        weighted_x += msg.getVariable<float>("x") * m; 
        weighted_y += msg.getVariable<float>("y") * m; 
        weighted_z += msg.getVariable<float>("z") * m;
        star_count++;
    }
    
    if (total_mass > 0.0001f) {
        FLAMEGPU->setVariable<float>("com_x", weighted_x / total_mass);
        FLAMEGPU->setVariable<float>("com_y", weighted_y / total_mass);
        FLAMEGPU->setVariable<float>("com_z", weighted_z / total_mass);
    }
    FLAMEGPU->setVariable<float>("total_mass", total_mass);
    FLAMEGPU->setVariable<int>("star_count", star_count);
    int threshold = FLAMEGPU->environment.getProperty<int>("dense_entity_threshold");
    FLAMEGPU->setVariable<int>("is_dense", (star_count > threshold) ? 1 : 0);

    return flamegpu::ALIVE;
}

// [VOXEL_BROADCAST_TEMPLATE]
FLAMEGPU_AGENT_FUNCTION(voxel_l{CURRENT}_broadcast, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    float mass = FLAMEGPU->getVariable<float>("total_mass");
    if (mass < 0.0001f) return flamegpu::ALIVE; 
    
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("com_x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("com_y"));
    FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("com_z"));
    FLAMEGPU->message_out.setVariable<float>("mass", mass);
    FLAMEGPU->message_out.setVariable<int>("voxel_id", FLAMEGPU->getVariable<int>("voxel_id"));
    
    int parent_id = -1;
    {PARENT_CALCULATION} 
    FLAMEGPU->message_out.setVariable<int>("parent_id", parent_id);
    return flamegpu::ALIVE;
}

// [STEP_UP_TEMPLATE]
FLAMEGPU_AGENT_FUNCTION(voxel_l{CURRENT}_to_l{NEXT}_output, flamegpu::MessageNone, flamegpu::MessageBucket) {
    float mass = FLAMEGPU->getVariable<float>("total_mass");
    if (mass < 0.0001f) return flamegpu::ALIVE; 
    float min_bound = FLAMEGPU->environment.getProperty<float>("env_min_bound");
    float next_voxel_size = FLAMEGPU->environment.getProperty<float>("l{NEXT}_voxel_size");
    int next_grid_dim = FLAMEGPU->environment.getProperty<int>("l{NEXT}_grid_dim");
    
    float cx = FLAMEGPU->getVariable<float>("com_x"); float cy = FLAMEGPU->getVariable<float>("com_y"); float cz = FLAMEGPU->getVariable<float>("com_z");
    int bucket_id = calculate_voxel_hash(cx, cy, cz, min_bound, next_voxel_size, next_grid_dim);

    FLAMEGPU->message_out.setVariable<float>("x", cx); FLAMEGPU->message_out.setVariable<float>("y", cy); FLAMEGPU->message_out.setVariable<float>("z", cz);
    FLAMEGPU->message_out.setVariable<float>("mass", mass);
    FLAMEGPU->message_out.setKey(bucket_id);
    return flamegpu::ALIVE;
}

// [GRAVITY_TOP_LAYER_TEMPLATE]
FLAMEGPU_AGENT_FUNCTION(star_gravity_l{CURRENT}, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    float x = FLAMEGPU->getVariable<float>("x"); float y = FLAMEGPU->getVariable<float>("y"); float z = FLAMEGPU->getVariable<float>("z");
    float vx = FLAMEGPU->getVariable<float>("vx"); float vy = FLAMEGPU->getVariable<float>("vy"); float vz = FLAMEGPU->getVariable<float>("vz");
    
    float min_bound = FLAMEGPU->environment.getProperty<float>("env_min_bound");
    float curr_size = FLAMEGPU->environment.getProperty<float>("l{CURRENT}_voxel_size");
    int curr_dim = FLAMEGPU->environment.getProperty<int>("l{CURRENT}_grid_dim");
    float mac_threshold = FLAMEGPU->environment.getProperty<float>("mac_threshold");
    float dt = FLAMEGPU->environment.getProperty<float>("dt");
    float grav_constant = FLAMEGPU->environment.getProperty<float>("grav_constant");

    for (auto &msg : FLAMEGPU->message_in) {
        if (check_mac(x, y, z, msg.getVariable<int>("voxel_id"), curr_size, curr_dim, min_bound, mac_threshold)) { 
            apply_gravity(vx, vy, vz, x, y, z, msg.getVariable<float>("x"), msg.getVariable<float>("y"), msg.getVariable<float>("z"), msg.getVariable<float>("mass"), dt, grav_constant);
        }
    }
    FLAMEGPU->setVariable<float>("vx", vx); FLAMEGPU->setVariable<float>("vy", vy); FLAMEGPU->setVariable<float>("vz", vz);
    return flamegpu::ALIVE;
}

// [GRAVITY_MID_LAYER_TEMPLATE]
FLAMEGPU_AGENT_FUNCTION(star_gravity_l{CURRENT}, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    float x = FLAMEGPU->getVariable<float>("x"); float y = FLAMEGPU->getVariable<float>("y"); float z = FLAMEGPU->getVariable<float>("z");
    float vx = FLAMEGPU->getVariable<float>("vx"); float vy = FLAMEGPU->getVariable<float>("vy"); float vz = FLAMEGPU->getVariable<float>("vz");
    
    float min_bound = FLAMEGPU->environment.getProperty<float>("env_min_bound");
    float curr_size = FLAMEGPU->environment.getProperty<float>("l{CURRENT}_voxel_size");
    int curr_dim = FLAMEGPU->environment.getProperty<int>("l{CURRENT}_grid_dim");
    float next_size = FLAMEGPU->environment.getProperty<float>("l{NEXT}_voxel_size");
    int next_dim = FLAMEGPU->environment.getProperty<int>("l{NEXT}_grid_dim");
    float mac_threshold = FLAMEGPU->environment.getProperty<float>("mac_threshold");
    float dt = FLAMEGPU->environment.getProperty<float>("dt");
    float grav_constant = FLAMEGPU->environment.getProperty<float>("grav_constant");

    for (auto &msg : FLAMEGPU->message_in) {
        bool my_mac_passes = check_mac(x, y, z, msg.getVariable<int>("voxel_id"), curr_size, curr_dim, min_bound, mac_threshold);
        bool parent_mac_passes = check_mac(x, y, z, msg.getVariable<int>("parent_id"), next_size, next_dim, min_bound, mac_threshold);
        
        if (my_mac_passes && !parent_mac_passes) {
            apply_gravity(vx, vy, vz, x, y, z, msg.getVariable<float>("x"), msg.getVariable<float>("y"), msg.getVariable<float>("z"), msg.getVariable<float>("mass"), dt, grav_constant);
        }
    }
    FLAMEGPU->setVariable<float>("vx", vx); FLAMEGPU->setVariable<float>("vy", vy); FLAMEGPU->setVariable<float>("vz", vz);
    return flamegpu::ALIVE;
}

// [GRAVITY_STARS_TEMPLATE]
FLAMEGPU_AGENT_FUNCTION(star_gravity_stars, flamegpu::MessageBucket, flamegpu::MessageNone) {
    float x = FLAMEGPU->getVariable<float>("x"); float y = FLAMEGPU->getVariable<float>("y"); float z = FLAMEGPU->getVariable<float>("z");
    float vx = FLAMEGPU->getVariable<float>("vx"); float vy = FLAMEGPU->getVariable<float>("vy"); float vz = FLAMEGPU->getVariable<float>("vz");
    int is_dense = FLAMEGPU->getVariable<int>("is_dense");
    
    float min_bound = FLAMEGPU->environment.getProperty<float>("env_min_bound");
    float l1_size = FLAMEGPU->environment.getProperty<float>("l1_voxel_size");
    int l1_dim = FLAMEGPU->environment.getProperty<int>("l1_grid_dim");
    float mac_threshold = FLAMEGPU->environment.getProperty<float>("mac_threshold");
    float dt = FLAMEGPU->environment.getProperty<float>("dt");
    float grav_constant = FLAMEGPU->environment.getProperty<float>("grav_constant");

    int my_x = max(0, min((int)floorf((x - min_bound) / l1_size), l1_dim - 1));
    int my_y = max(0, min((int)floorf((y - min_bound) / l1_size), l1_dim - 1));
    int my_z = max(0, min((int)floorf((z - min_bound) / l1_size), l1_dim - 1));
    int my_voxel_id = my_x + my_y * l1_dim + my_z * l1_dim * l1_dim;

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = my_x + dx; int ny = my_y + dy; int nz = my_z + dz;
                if (nx >= 0 && nx < l1_dim && ny >= 0 && ny < l1_dim && nz >= 0 && nz < l1_dim) {
                    int neighbor_id = nx + ny * l1_dim + nz * l1_dim * l1_dim;
                    
                    if (!check_mac(x, y, z, neighbor_id, l1_size, l1_dim, min_bound, mac_threshold)) {
                        if (is_dense == 1 && my_voxel_id == neighbor_id) continue; // P3M mesh takes over here

                        for (auto &msg : FLAMEGPU->message_in(neighbor_id)) {
                            float ox = msg.getVariable<float>("x"); float oy = msg.getVariable<float>("y"); float oz = msg.getVariable<float>("z");
                            if (ox != x || oy != y || oz != z) { 
                                apply_gravity(vx, vy, vz, x, y, z, ox, oy, oz, msg.getVariable<float>("mass"), dt, grav_constant);
                            }
                        }
                    }
                }
            }
        }
    }
    FLAMEGPU->setVariable<float>("vx", vx); FLAMEGPU->setVariable<float>("vy", vy); FLAMEGPU->setVariable<float>("vz", vz);
    return flamegpu::ALIVE;
}


// ==========================================
// PHASE 3: PARTICLE-MESH (FLUID GRID)
// ==========================================

// [STAR_TO_MESH_BLOCK]
FLAMEGPU_AGENT_FUNCTION(star_to_mesh_bucket, flamegpu::MessageNone, flamegpu::MessageBucket) {
    if (FLAMEGPU->getVariable<int>("is_dense") == 0) return flamegpu::ALIVE;

    float x = FLAMEGPU->getVariable<float>("x"); float y = FLAMEGPU->getVariable<float>("y"); float z = FLAMEGPU->getVariable<float>("z");
    float min_bound = FLAMEGPU->environment.getProperty<float>("env_min_bound");
    float cell_size = FLAMEGPU->environment.getProperty<float>("mesh_cell_size");
    int global_dim = FLAMEGPU->environment.getProperty<int>("global_mesh_dim");

    int bucket_id = calculate_voxel_hash(x, y, z, min_bound, cell_size, global_dim);

    FLAMEGPU->message_out.setVariable<float>("x", x); FLAMEGPU->message_out.setVariable<float>("y", y); FLAMEGPU->message_out.setVariable<float>("z", z);
    FLAMEGPU->message_out.setVariable<float>("mass", FLAMEGPU->getVariable<float>("mass"));
    FLAMEGPU->message_out.setKey(bucket_id);
    return flamegpu::ALIVE;
}

// [MESH_GATHER_BLOCK]
FLAMEGPU_AGENT_FUNCTION(mesh_node_cic_gather, flamegpu::MessageBucket, flamegpu::MessageNone) {
    float my_x = FLAMEGPU->getVariable<float>("node_x"); float my_y = FLAMEGPU->getVariable<float>("node_y"); float my_z = FLAMEGPU->getVariable<float>("node_z");
    int my_gx = FLAMEGPU->getVariable<int>("global_x"); int my_gy = FLAMEGPU->getVariable<int>("global_y"); int my_gz = FLAMEGPU->getVariable<int>("global_z");
    
    float cell_size = FLAMEGPU->environment.getProperty<float>("mesh_cell_size");
    int global_dim = FLAMEGPU->environment.getProperty<int>("global_mesh_dim");
    float accumulated_density = 0.0f;
    
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = my_gx + dx; int ny = my_gy + dy; int nz = my_gz + dz;
                if (nx >= 0 && nx < global_dim && ny >= 0 && ny < global_dim && nz >= 0 && nz < global_dim) {
                    int neighbor_bucket = nx + (ny * global_dim) + (nz * global_dim * global_dim);
                    for (auto &msg : FLAMEGPU->message_in(neighbor_bucket)) {
                        float dx_dist = fabsf(msg.getVariable<float>("x") - my_x);
                        float dy_dist = fabsf(msg.getVariable<float>("y") - my_y);
                        float dz_dist = fabsf(msg.getVariable<float>("z") - my_z);

                        if (dx_dist < cell_size && dy_dist < cell_size && dz_dist < cell_size) {
                            float weight = (1.0f - (dx_dist / cell_size)) * (1.0f - (dy_dist / cell_size)) * (1.0f - (dz_dist / cell_size));
                            accumulated_density += msg.getVariable<float>("mass") * weight;
                        }
                    }
                }
            }
        }
    }
    FLAMEGPU->setVariable<float>("density", accumulated_density);
    return flamegpu::ALIVE;
}

// [MESH_BROADCAST_BLOCK]
FLAMEGPU_AGENT_FUNCTION(mesh_node_broadcast, flamegpu::MessageNone, flamegpu::MessageArray3D) {
    FLAMEGPU->message_out.setVariable<float>("density", FLAMEGPU->getVariable<float>("density"));
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<int>("global_x"), FLAMEGPU->getVariable<int>("global_y"), FLAMEGPU->getVariable<int>("global_z"));
    return flamegpu::ALIVE;
}

// [MESH_SOLVER_BLOCK]
FLAMEGPU_AGENT_FUNCTION(mesh_node_force_solver, flamegpu::MessageArray3D, flamegpu::MessageArray3D) {
    float my_x = FLAMEGPU->getVariable<float>("node_x"); float my_y = FLAMEGPU->getVariable<float>("node_y"); float my_z = FLAMEGPU->getVariable<float>("node_z");
    int my_gx = FLAMEGPU->getVariable<int>("global_x"); int my_gy = FLAMEGPU->getVariable<int>("global_y"); int my_gz = FLAMEGPU->getVariable<int>("global_z");
    float cell_size = FLAMEGPU->environment.getProperty<float>("mesh_cell_size");
    int global_dim = FLAMEGPU->environment.getProperty<int>("global_mesh_dim");
    float G = FLAMEGPU->environment.getProperty<float>("grav_constant");

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    int window_radius = 5; 
    
    for (int dz = -window_radius; dz <= window_radius; dz++) {
        for (int dy = -window_radius; dy <= window_radius; dy++) {
            for (int dx = -window_radius; dx <= window_radius; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                
                int nx = my_gx + dx; int ny = my_gy + dy; int nz = my_gz + dz;
                if (nx >= 0 && nx < global_dim && ny >= 0 && ny < global_dim && nz >= 0 && nz < global_dim) {
                    auto msg = FLAMEGPU->message_in.at(nx, ny, nz);
                    float omass = msg.getVariable<float>("density");
                    
                    if (omass > 0.0001f) {
                        float ox = my_x + (dx * cell_size); float oy = my_y + (dy * cell_size); float oz = my_z + (dz * cell_size);
                        float ddx = ox - my_x; float ddy = oy - my_y; float ddz = oz - my_z;
                        float dist_sq = ddx*ddx + ddy*ddy + ddz*ddz + 0.05f; 
                        float inv_dist = rsqrtf(dist_sq);
                        float force = G * omass * inv_dist * inv_dist * inv_dist;
                        fx += ddx * force; fy += ddy * force; fz += ddz * force;
                    }
                }
            }
        }
    }
    FLAMEGPU->message_out.setVariable<float>("force_x", fx); FLAMEGPU->message_out.setVariable<float>("force_y", fy); FLAMEGPU->message_out.setVariable<float>("force_z", fz);
    FLAMEGPU->message_out.setIndex(my_gx, my_gy, my_gz);
    return flamegpu::ALIVE;
}

// [STAR_GATHER_BLOCK]
FLAMEGPU_AGENT_FUNCTION(star_mesh_gather, flamegpu::MessageArray3D, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("is_dense") == 0) return flamegpu::ALIVE;

    float x = FLAMEGPU->getVariable<float>("x"); float y = FLAMEGPU->getVariable<float>("y"); float z = FLAMEGPU->getVariable<float>("z");
    float min_bound = FLAMEGPU->environment.getProperty<float>("env_min_bound");
    float cell_size = FLAMEGPU->environment.getProperty<float>("mesh_cell_size");
    int global_dim = FLAMEGPU->environment.getProperty<int>("global_mesh_dim");

    int gx = (int)floorf((x - min_bound) / cell_size);
    int gy = (int)floorf((y - min_bound) / cell_size);
    int gz = (int)floorf((z - min_bound) / cell_size);

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    for (int dz = 0; dz <= 1; dz++) {
        for (int dy = 0; dy <= 1; dy++) {
            for (int dx = 0; dx <= 1; dx++) {
                float corner_x = min_bound + ((gx + dx) * cell_size); float corner_y = min_bound + ((gy + dy) * cell_size); float corner_z = min_bound + ((gz + dz) * cell_size);
                float dist_x = fabsf(x - corner_x); float dist_y = fabsf(y - corner_y); float dist_z = fabsf(z - corner_z);
                
                if (dist_x < cell_size && dist_y < cell_size && dist_z < cell_size) {
                    float weight = (1.0f - (dist_x / cell_size)) * (1.0f - (dist_y / cell_size)) * (1.0f - (dist_z / cell_size));
                    int idx_x = max(0, min(gx + dx, global_dim - 1)); int idx_y = max(0, min(gy + dy, global_dim - 1)); int idx_z = max(0, min(gz + dz, global_dim - 1));
                    
                    auto msg = FLAMEGPU->message_in.at(idx_x, idx_y, idx_z);
                    fx += msg.getVariable<float>("force_x") * weight; fy += msg.getVariable<float>("force_y") * weight; fz += msg.getVariable<float>("force_z") * weight;
                }
            }
        }
    }
    
    // Add grid force to velocity!
    FLAMEGPU->setVariable<float>("vx", FLAMEGPU->getVariable<float>("vx") + fx);
    FLAMEGPU->setVariable<float>("vy", FLAMEGPU->getVariable<float>("vy") + fy);
    FLAMEGPU->setVariable<float>("vz", FLAMEGPU->getVariable<float>("vz") + fz);
    return flamegpu::ALIVE;
}

// ==========================================
// PHASE 4: P³M HANDSHAKE & KINEMATICS
// ==========================================

// [RADAR_BROADCAST_BLOCK]
FLAMEGPU_AGENT_FUNCTION(broadcast_radar, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    if (FLAMEGPU->getVariable<int>("is_dense") == 0) return flamegpu::ALIVE; // Only needed in the mesh zones
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
    FLAMEGPU->message_out.setVariable<float>("mass", FLAMEGPU->getVariable<float>("mass"));
    return flamegpu::ALIVE;
}

// [P3M_HANDSHAKE_BLOCK]
FLAMEGPU_AGENT_FUNCTION(star_p3m_handshake, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("is_dense") == 0) return flamegpu::ALIVE;

    float x = FLAMEGPU->getVariable<float>("x"); float y = FLAMEGPU->getVariable<float>("y"); float z = FLAMEGPU->getVariable<float>("z");
    float vx = FLAMEGPU->getVariable<float>("vx"); float vy = FLAMEGPU->getVariable<float>("vy"); float vz = FLAMEGPU->getVariable<float>("vz");
    
    float rc = FLAMEGPU->environment.getProperty<float>("cutoff_radius");
    float G = FLAMEGPU->environment.getProperty<float>("grav_constant");
    float dt = FLAMEGPU->environment.getProperty<float>("dt");
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    for (auto &msg : FLAMEGPU->message_in(x, y, z)) {
        float ox = msg.getVariable<float>("x"); float oy = msg.getVariable<float>("y"); float oz = msg.getVariable<float>("z");
        
        float dx = ox - x; float dy = oy - y; float dz = oz - z;
        float dist_sq = dx*dx + dy*dy + dz*dz + 0.001f; 
        float dist = sqrtf(dist_sq);

        if (dist < rc && dist > 0.0001f) {
            float omass = msg.getVariable<float>("mass");
            float inv_dist = 1.0f / dist;
            
            // 1. Calculate Exact Micro Force
            float f_micro_exact = G * omass * inv_dist * inv_dist * inv_dist;

            // 2. Erfc Splitting to subtract the Mesh Grid's blurry overlap
            float alpha = 2.0f / rc; 
            float blend_factor = erfcf(alpha * dist); 
            float f_corrected = f_micro_exact * blend_factor;

            fx += dx * f_corrected;
            fy += dy * f_corrected;
            fz += dz * f_corrected;
        }
    }

    FLAMEGPU->setVariable<float>("vx", vx + (fx * dt));
    FLAMEGPU->setVariable<float>("vy", vy + (fy * dt));
    FLAMEGPU->setVariable<float>("vz", vz + (fz * dt));
    return flamegpu::ALIVE;
}

// [INTEGRATION_BLOCK]
FLAMEGPU_AGENT_FUNCTION(star_integration, flamegpu::MessageNone, flamegpu::MessageNone) {
    float vx = FLAMEGPU->getVariable<float>("vx");
    float vy = FLAMEGPU->getVariable<float>("vy");
    float vz = FLAMEGPU->getVariable<float>("vz");
    float dt = FLAMEGPU->environment.getProperty<float>("dt");

    FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") + (vx * dt));
    FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") + (vy * dt));
    FLAMEGPU->setVariable<float>("z", FLAMEGPU->getVariable<float>("z") + (vz * dt));

    return flamegpu::ALIVE;
}