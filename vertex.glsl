#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

in vec4 inPosition;
uniform vec2 uResolution; // = (window-width, window-height)
out float volume;
out float DEPTH;

void main() {
    vec2  pos = inPosition.xy;
    float AR = uResolution.y / uResolution.x;
    //pos.x *= AR;

    volume = inPosition.w;

    gl_Position = proj * view * model * vec4(pos, inPosition.z, 1.0);
    //gl_PointSize = volume * 5.0 + 5.0;
    gl_PointSize = 5.0;
    DEPTH = gl_Position.z;
}