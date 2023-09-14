#version 330 core

in float volume;
in float freqency;
uniform int gradient;


void main() {
    if(gradient == 0){
        gl_FragColor = vec4(1.0 - freqency, 0.0, freqency, 0.0) + 0.5f * vec4(vec3(volume), 1.0);
    }
    else {
        gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0) + 0.5f * vec4(vec3(volume), 1.0);
    }
}