#version 330 core

in float vol;
in float frq;
uniform int color_mode;
uniform vec4 color;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    if(color_mode == 0){
        vec3 col = hsv2rgb(vec3(frq, 1.0 - 0.5 * vol, vol * 2.0));
        gl_FragColor = vec4(col, 1.0);
    }
    else if (color_mode == 1) {
        gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0) + 0.5f * vec4(vec3(vol), 1.0);
    }
    else if (color_mode == 2) {
        gl_FragColor = color;
    }
    else if (color_mode == 3) {
        gl_FragColor = color;
    }
}