uniform sampler2D fbo_texture;
varying vec2 f_texcoord;

void main() {
    vec2 texcoord = f_texcoord;
    //texcoord.x += sin(texcoord.y * 4*2*3.14159) / 100.0;
    gl_FragColor = texture2D(fbo_texture, texcoord);
}