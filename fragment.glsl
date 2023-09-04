#version 330 core

in float volume;
in float freqency;


void main() {
    gl_FragColor = vec4(1.0 - freqency, 0.0, freqency, 0.0) + 0.5f * vec4(vec3(volume), 0.0);
}