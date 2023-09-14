#version 330 core

in float volume;
in float freqency;
uniform int color_mode;
uniform vec4 color;


void main() {
    if(color_mode == 0){
        gl_FragColor = vec4(1.0 - freqency, 0.0, freqency, 0.0) + 0.5f * vec4(vec3(volume), 1.0);
    }
    else if (color_mode == 1) {
        gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0) + 0.5f * vec4(vec3(volume), 1.0);
    }
    else if (color_mode == 2) {
        gl_FragColor = color + 0.1f * vec4(vec3(volume), 1.0);
    }
    else if (color_mode == 3) {
        gl_FragColor = color + 0.5f * vec4(vec3(volume), 1.0);
    }
}