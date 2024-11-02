#version 330 core

uniform sampler2D pTexture;
in vec2 fragCoord;

void main() {
    gl_FragColor = texture(pTexture, fragCoord);
    //gl_FragColor = vec4(fragCoord, 0.0, 1.0);
}