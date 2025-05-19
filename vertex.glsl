#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

layout (location = 0) in vec3 position;
layout (location = 1) in float volume;
layout (location = 2) in float frequency;

uniform vec2 uResolution;
out float vol;
out float frq;

void main() {
    vec2  pos = position.xy;
    float AR = uResolution.y / uResolution.x;
    vol = volume;
    gl_Position = proj * view * model * vec4(pos, position.z, 1.0);
    frq = frequency;
}