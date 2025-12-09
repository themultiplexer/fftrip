#version 330 core
in vec2 TexCoords;
out vec4 color;

uniform sampler2D text;
uniform vec3 textColor;
uniform float volume;
uniform float width;

const float smoothing = 0.1f;

void main()
{
    if(true) {
        float distance = texture2D(text, TexCoords).r;
        float width1 = 0.1 + 0.5;
        float alpha1 = smoothstep(width1 - smoothing, width1 + smoothing, distance);
        float width2 = 0.05 + (width / 20.0);
        float alpha2 = smoothstep(width2 - smoothing, width2 + smoothing, distance);
        color = vec4(textColor.rgb, alpha2 - alpha1);
    } else {
        float distance = texture2D(text, TexCoords).r;
        float width1 = 0.5;
        float alpha1 = smoothstep(width1 - smoothing, width1 + smoothing, distance);
        //float width2 = abs(sin(3.0f)) * 0.1 + 0.4;
        float width2 = (10.0 - volume) * 0.01 + 0.4;
        float alpha2 = smoothstep(width2 - smoothing, width2 + smoothing, distance);
        color = vec4(textColor.rgb, alpha2 - alpha1 * (1.0 - clamp(volume * 0.1 - 0.2, 0.0f, 1.0f)));
    }
}