#version 450

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
    int need_srgb_conv;
} push_constants;
layout(binding = 0, set = 0) uniform sampler2D font_texture;

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;

// 0-255 sRGB  from  0-1 linear
vec3 srgb_from_linear(vec3 rgb) {
  bvec3 cutoff = lessThan(rgb, vec3(0.0031308));
  vec3 lower = rgb * vec3(3294.6);
  vec3 higher = vec3(269.025) * pow(rgb, vec3(1.0 / 2.4)) - vec3(14.025);
  return mix(higher, lower, vec3(cutoff));
}

vec4 srgba_from_linear(vec4 rgba) {
  return vec4(srgb_from_linear(rgba.rgb), 255.0 * rgba.a);
}

// 0-1 linear  from  0-255 sRGB
vec3 linear_from_srgb(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(10.31475));
    vec3 lower = srgb / vec3(3294.6);
    vec3 higher = pow((srgb + vec3(14.025)) / vec3(269.025), vec3(2.4));
    return mix(higher, lower, cutoff);
}

vec4 linear_from_srgba(vec4 srgba) {
    return vec4(linear_from_srgb(srgba.rgb * 255.0), srgba.a);
}

void main() {
    vec4 texture_color = texture(font_texture, v_tex_coords);
    if (push_constants.need_srgb_conv == 0) {
        f_color = v_color * texture_color;
    } else {
        f_color = srgba_from_linear(v_color * texture_color) / 255.0;
        f_color.a = pow(f_color.a, 1.6);
    }
}