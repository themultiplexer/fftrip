#version 330 core

in float vol;
in float frq;
uniform int color_mode;
uniform vec4 color;


void main() {
    if(color_mode == 0){
        gl_FragColor = vec4(vec3(1.0 - frq, 0.0, frq) * 0.5f, 1.0) + vec4(0.5f * vec3(vol), 1.0);
    }
    else if (color_mode == 1) {
        gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0) + 0.5f * vec4(vec3(vol), 1.0);
    }
    else if (color_mode == 2) {
        gl_FragColor = color + 0.1f * vec4(vec3(vol), 1.0);
    }
    else if (color_mode == 3) {
        gl_FragColor = color + 0.5f * vec4(vec3(vol), 1.0);
    }
}