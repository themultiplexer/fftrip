#version 330 core
in vec2 TexCoords;
out vec4 color;

uniform sampler2D text;
uniform vec3 textColor;
uniform float time;
uniform float volume;

const float smoothing = 0.01f;

vec2 rotate(vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, s, -s, c);
	return m * v;
}

void main()
{
    float glyphShape = texture(text, TexCoords).r;

    if(false) {
        float distance = texture2D(text, TexCoords).r;
        float width1 = abs(sin(3.0f * time)) * 0.1 + 0.5;
        float alpha1 = smoothstep(width1 - smoothing, width1 + smoothing, distance);
        float width2 = abs(sin(3.0f * time)) * 0.1 + 0.4;
        float alpha2 = smoothstep(width2 - smoothing, width2 + smoothing, distance);
        color = vec4(textColor.rgb, alpha2 - alpha1);
    } else {
        float distance = texture2D(text, TexCoords).r;
        float width1 = 0.5;
        float alpha1 = smoothstep(width1 - smoothing, width1 + smoothing, distance);
        float width2 = abs(sin(3.0f * time)) * 0.1 + 0.4;
        //float width2 = (10.0 - volume) * 0.01 + 0.4;
        float alpha2 = smoothstep(width2 - smoothing, width2 + smoothing, distance);
        color = vec4(textColor.rgb, alpha2 - alpha1 * (1.0 - clamp(volume * 0.1 - 0.2, 0.0f, 1.0f)));
    }
}