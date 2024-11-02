#version 330 core

in vec2 pos;
uniform vec2 uResolution; // = (window-width, window-height)
out vec2 fragCoord;

void main() {
    //float AR = uResolution.y / uResolution.x;
    gl_Position = vec4(pos, 0.0, 1.0);
    fragCoord = (pos + 1.0) / 2.0;
}