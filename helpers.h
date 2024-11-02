#include <string>
#include <tuple>
#include <vector>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <sstream>
#include <iostream>
#include <fstream>

void GLAPIENTRY
MessageCallback( GLenum source,
                 GLenum type,
                 GLuint id,
                 GLenum severity,
                 GLsizei length,
                 const GLchar* message,
                 const void* userParam )
{
  fprintf( stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
           ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
            type, severity, message );
}

std::vector<std::tuple<GLenum, std::string, std::string>> loadShaderContent(std::string vertex_path, std::string fragment_path) {
	std::stringstream vertex;
	vertex << std::ifstream(vertex_path).rdbuf();
	std::stringstream fragment;
	fragment << std::ifstream(fragment_path).rdbuf();
	return {
		std::make_tuple(GL_VERTEX_SHADER, vertex.str(), vertex_path),
		std::make_tuple(GL_FRAGMENT_SHADER, fragment.str(), fragment_path),
	};
}

bool loadShaders(GLuint* program, std::vector<std::tuple<GLenum, std::string, std::string>> shaders) {
	for (const auto& s : shaders) {
		GLenum type = std::get<0>(s);
		const std::string& source = std::get<1>(s);

		const GLchar* src = source.c_str();

		GLuint shader = glCreateShader(type);
		glShaderSource(shader, 1, &src, nullptr);
		glCompileShader(shader);

		GLint compiled = 0;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
		if (!compiled) {
			GLint length = 0;
			glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);

			if (length > 1) {
				std::string log(length, '\0');
				glGetShaderInfoLog(shader, length, &length, &log[0]);
				printf("Shader (%s) compile failed:\n%s\n", source.c_str(), log.c_str());
			}
			else {
				printf("Shader (%s) compile failed.\n", source.c_str());
			}

			return false;
		}

		glAttachShader(*program, shader);
	}

	glLinkProgram(*program);

	GLint linked = 0;
	glGetProgramiv(*program, GL_LINK_STATUS, &linked);

	if (!linked) {
		GLint length = 0;
		glGetProgramiv(*program, GL_INFO_LOG_LENGTH, &length);

		if (length > 1) {
			std::string log(length, '\0');
			glGetProgramInfoLog(*program, length, &length, &log[0]);
			printf("Program (%s) link failed:\n%s", std::get<2>(shaders[0]).c_str(), log.c_str());
		}
		else {
			printf("Program (%s) link failed.\n", std::get<2>(shaders[0]).c_str());
		}

		return false;
	}
	return true;
}