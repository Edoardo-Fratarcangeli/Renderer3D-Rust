struct Uniforms {
    view_projection: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
};

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) instance_color: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
};

@vertex
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    
    // Transform normal to world space (using rotation part of model matrix)
    // For non-uniform scaling, we should use inverse-transpose, but for uniform scale/rotation only, map is fine.
    let normal_matrix = mat3x3<f32>(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz
    );
    
    var out: VertexOutput;
    out.color = model.color;
    out.instance_color = instance.color.rgb;
    out.world_normal = normalize(normal_matrix * model.normal);
    out.clip_position = uniforms.view_projection * model_matrix * vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Basic Directional Light
    let light_dir = normalize(vec3<f32>(1.0, 2.0, 1.0));
    let diffuse_strength = max(dot(in.world_normal, light_dir), 0.1); // Add ambient 0.1
    
    // Mix vertex color (grid) with instance color (objects)
    let mixed_color = in.color * in.instance_color;
    
    let result = mixed_color * diffuse_strength;

    return vec4<f32>(result, 1.0);
}
