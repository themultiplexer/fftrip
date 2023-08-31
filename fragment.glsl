#version 330 core

in float volume;
uniform sampler2D depthTexture;

void main() {
    gl_FragColor = vec4(volume, 0.9, 0.8, 1.0);
}