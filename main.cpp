
#include <cstdio>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glext.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <kissfft/kiss_fft.h>
#include <math.h>
#include <rtaudio/RtAudio.h>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define NUM_SEGMENTS 63
#define VERT_LENGTH 4 //x,y,z,volume
#define NUM_POINTS (NUM_SEGMENTS+1)
#define SPHERE_LAYERS 8

GLuint program;
GLuint vao, vbo;
float radius = 1.0f;
RtAudio adc(RtAudio::Api::LINUX_PULSE);
GLfloat circleVertices[NUM_POINTS * VERT_LENGTH * SPHERE_LAYERS];

kiss_fft_cfg cfg;

float rawdata[1024];
float freqs[1024];

float red_rawdata[NUM_POINTS];
float red_freqs[NUM_POINTS];
float last_freqs[NUM_POINTS];

double lastTime = glfwGetTime();
int nbFrames = 0;

enum VisMode{
  LINES,
  CIRCLE,
  SPHERE,
  SPHERE_SPIRAL
};

VisMode mode = CIRCLE;

int record(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
           double streamTime, RtAudioStreamStatus status, void *userData) {
  if (status){
    std::cout << "Stream overflow detected!" << std::endl;
    return 0;
    }
  
  //printf("%d \n", nBufferFrames);
  kiss_fft_cpx in[1024] = {};
  for (unsigned int i = 0; i < nBufferFrames; i++) {
    in[i].r = ((float *)inputBuffer)[i];
    rawdata[i] = ((float *)inputBuffer)[i];
  }

  kiss_fft_cpx out[1024] = {};
  kiss_fft(cfg, in, out);
  for (int i = 0; i < 1024; i++) {
    freqs[i] = sqrt(out[i].r * out[i].r + out[i].i * out[i].i);
  }

  int sample_group = 1024 / NUM_POINTS;
  int fft_group = 400 / NUM_POINTS;
  for(int i = 0; i < NUM_POINTS; i++) {
    red_rawdata[i] = 0;
    red_freqs[i] = 0;
    for(int j = 0; j < sample_group; j++) {
      red_rawdata[i] += rawdata[i * sample_group + j];
    }
    for(int j = 0; j < fft_group; j++) {
      red_freqs[i] += freqs[i * fft_group + j + 40];
    }
    red_freqs[i] += last_freqs[i];
    red_freqs[i] /= 2.0;
  }

  for(int i = 0; i < NUM_POINTS; i++) {
    last_freqs[i] = red_freqs[i];
  }
  // Do something with the data in the "inputBuffer" buffer.
  return 0;
}

void getdevices() {
  // Get the list of device IDs
  std::vector<unsigned int> ids = adc.getDeviceIds();
  if (ids.size() == 0) {
    std::cout << "No devices found." << std::endl;
    exit(0);
  }

  // Scan through devices for various capabilities
  RtAudio::DeviceInfo info;
  for (unsigned int n = 0; n < ids.size(); n++) {

    info = adc.getDeviceInfo(ids[n]);

    // Print, for example, the name and maximum number of output channels for
    // each device
    std::cout << "device name = " << info.name << std::endl;
    std::cout << "device id = " << ids[n] << std::endl;
    std::cout << ": maximum input channels = " << info.inputChannels << std::endl;
    std::cout << ": maximum output channels = " << info.outputChannels << std::endl;
  }
}

static void set_camera(float cam_x, float cam_y, float cam_z, float target_z) {
    glUseProgram(program);
    glm::mat4 trans = glm::mat4(1.0f);
    trans = glm::rotate(trans, glm::radians(0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    GLint uniTrans = glGetUniformLocation(program, "model");
    glUniformMatrix4fv(uniTrans, 1, GL_FALSE, glm::value_ptr(trans));
    
    glm::mat4 view = glm::lookAt(
        glm::vec3(cam_x, cam_y, cam_z),
        glm::vec3(0.0f, 0.0f, target_z),
        glm::vec3(0.0f, 0.0f, 1.0f)
    );
    GLint uniView = glGetUniformLocation(program, "view");
    glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));

    glm::mat4 proj = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 1.0f, 10.0f);
    GLint uniProj = glGetUniformLocation(program, "proj");
    glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    } else if (key == GLFW_KEY_L) {
      mode = VisMode::LINES;
    } else if (key == GLFW_KEY_C) {
      mode = VisMode::CIRCLE;
    } else if (key == GLFW_KEY_S) {
      mode = VisMode::SPHERE;
    } else if (key == GLFW_KEY_X) {
      mode = VisMode::SPHERE_SPIRAL;
    } else if (key == GLFW_KEY_F) {
      set_camera(0.0f, -2.5f, 2.5f, 1.0f);
    }
}

static void resize(GLFWwindow* window, int width, int height){
	  glViewport(0,0,width,height);
    glUseProgram(program);

    GLfloat uResolution[2] = { (float)width, (float)height };
    glUniform2fv(glGetUniformLocation(program, "uResolution"), 1, uResolution);
}

bool Initialize() {
  getdevices();

  RtAudio::StreamParameters parameters;
  parameters.deviceId = adc.getDefaultInputDevice();
  //parameters.deviceId = 132;
  parameters.nChannels = 1;
  parameters.firstChannel = 0;
  unsigned int sampleRate = 48000;
  unsigned int bufferFrames = 1024;

  if (adc.openStream(NULL, &parameters, RTAUDIO_FLOAT32, sampleRate, &bufferFrames, &record)) {
    std::cout << '\n' << adc.getErrorText() << '\n' << std::endl;
    exit(0); // problem with device settings
  }

  // Stream is open ... now start it.
  if (adc.startStream()) {
    std::cout << adc.getErrorText() << std::endl;
  }

  cfg = kiss_fft_alloc(1024, 0, NULL, NULL);

  // shaders
  std::ifstream v("../vertex.glsl");
  std::stringstream vertex;
  vertex << v.rdbuf();
  std::ifstream f("../fragment.glsl");
  std::stringstream fragment;
  fragment << f.rdbuf();

  std::vector<std::pair<GLenum, std::string>> shaders = {
      {GL_VERTEX_SHADER, vertex.str()},
      {GL_FRAGMENT_SHADER, fragment.str()},
  };
  printf("Creating program \n");
  program = glCreateProgram();
  printf("Created program \n");

  for (const auto &s : shaders) {
    GLenum type = s.first;
    const std::string &source = s.second;

    const GLchar *src = source.c_str();

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
        printf("Shader compile failed:\n%s\n", log.c_str());
      } else {
        printf("Shader compile failed.\n");
      }

      return false;
    }

    glAttachShader(program, shader);
  }

  glLinkProgram(program);

  GLint linked = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &linked);

  if (!linked) {
    GLint length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);

    if (length > 1) {
      std::string log(length, '\0');
      glGetProgramInfoLog(program, length, &length, &log[0]);
      printf("Program link failed:\n%s", log.c_str());
    } else {
      printf("Program link failed.\n");
    }

    return false;
  }

  set_camera(0.0f, -0.1f, 5.0f, 0.0f);

  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);

  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(circleVertices), circleVertices, GL_STATIC_DRAW);
  glUseProgram(program);

  glVertexAttribPointer(0, VERT_LENGTH, GL_FLOAT, GL_FALSE, VERT_LENGTH * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_BLEND);

  glLineWidth(10.0);
  printf("Init finished \n");
  return true;
}

void Render() {
  glClear(GL_COLOR_BUFFER_BIT);

  double currentTime = glfwGetTime();
  nbFrames++;
  if ( currentTime - lastTime >= 1.0 ){ // If last prinf() was more than 1 sec ago
      // printf and reset timer
      printf("%f ms/frame\n", 1000.0/double(nbFrames));
      nbFrames = 0;
      lastTime += 1.0;
  }

  if (mode == LINES) {
    for (int i = 0; i <= NUM_SEGMENTS; i++) {
      circleVertices[i * VERT_LENGTH] = i * 0.0275 - 0.9;
      circleVertices[i * VERT_LENGTH + 1] = red_freqs[i] * 0.1 - 0.9;
      circleVertices[i * VERT_LENGTH + 2] = 0.0; 
      circleVertices[i * VERT_LENGTH + 3] = red_freqs[i] * 0.1;
    }
  } else if (mode == CIRCLE) {
    for (int i = 0; i <= NUM_SEGMENTS; i++) {
      float theta = 2.0f * M_PI * float(i) / float(NUM_SEGMENTS) - M_PI;
      float r = red_freqs[i] * 0.1 + 0.5;

      float x = radius * r * cosf(theta);
      float y = radius * r * sinf(theta);

      circleVertices[i * VERT_LENGTH] = x;
      circleVertices[i * VERT_LENGTH + 1] = y;
      circleVertices[i * VERT_LENGTH + 2] = 0.0;
      circleVertices[i * VERT_LENGTH + 3] = red_freqs[i] * 0.1;
    }
  } else if (mode == SPHERE) {
    for (int c = 0; c < SPHERE_LAYERS; c++) {
      for (int i = 0; i < NUM_POINTS; i++) {
        float theta = 2.0f * M_PI * float(i) / float(NUM_POINTS) - M_PI;

        float r = red_freqs[i] * 0.1 + 0.5;
        float layer = sin(((float)c / (float)SPHERE_LAYERS) * M_PI);
        float x = radius * layer * r * cosf(theta);
        float y = radius * layer * r * sinf(theta);

        circleVertices[c * NUM_POINTS * VERT_LENGTH + i * VERT_LENGTH] = x;
        circleVertices[c * NUM_POINTS * VERT_LENGTH + i * VERT_LENGTH + 1] = y;
        circleVertices[c * NUM_POINTS * VERT_LENGTH + i * VERT_LENGTH + 2] = c * 0.2;
        circleVertices[c * NUM_POINTS * VERT_LENGTH + i * VERT_LENGTH + 3] = red_freqs[i] * 0.1;
      }
    }
  } else if (mode == SPHERE_SPIRAL) {
    for (int i = 0; i < NUM_POINTS * SPHERE_LAYERS; i++) {
      float theta = 2.0f * M_PI * float(i) / float(NUM_POINTS) - M_PI;

      float r = freqs[i] * 0.1 + 0.5;
      float percent = ((float)i / (float)(NUM_POINTS * SPHERE_LAYERS));
      float layer = sin(percent * M_PI);
      float x = radius * layer * r * cosf(theta);
      float y = radius * layer * r * sinf(theta);

      circleVertices[i * VERT_LENGTH] = x;
      circleVertices[i * VERT_LENGTH + 1] = y;
      circleVertices[i * VERT_LENGTH + 2] = percent * 2.0;
      circleVertices[i * VERT_LENGTH + 3] = freqs[i] * 0.1;
    }
  }

  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(circleVertices), circleVertices, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Use the shader program
  glUseProgram(program);

  // Bind vertex array object (VAO)
  glBindVertexArray(vao);

  // Draw the circle as a line loop
  if (mode == LINES) {
    glDrawArrays(GL_LINE_STRIP, 0, NUM_SEGMENTS + 1);
    glDrawArrays(GL_POINTS, 0, NUM_POINTS);
  } else if (mode == CIRCLE) {
    glDrawArrays(GL_LINE_LOOP, 0, NUM_SEGMENTS + 1);
    glDrawArrays(GL_POINTS, 0, NUM_POINTS);
  } else if (mode == SPHERE) {
    glDrawArrays(GL_LINE_STRIP, 0, SPHERE_LAYERS * NUM_POINTS);
    glDrawArrays(GL_POINTS, 0, SPHERE_LAYERS * NUM_POINTS);
  } else {
    glDrawArrays(GL_LINE_STRIP, 0, SPHERE_LAYERS * NUM_POINTS);
    //glDrawArrays(GL_POINTS, 0, SPHERE_LAYERS * NUM_POINTS);
  }
  
  

  // Unbind VAO and shader
  glBindVertexArray(0);
  glUseProgram(0);
}

int main() {
  const unsigned int client_width = 3840;
  const unsigned int client_height = 2160;

  if (!glfwInit())
      exit(EXIT_FAILURE);


  GLFWwindow* window = glfwCreateWindow(client_width, client_height, "My Title", NULL, NULL);
  if (!window) {
    printf("Failed to create window.\n");
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, key_callback);
  glfwSetFramebufferSizeCallback(window, resize);
  glewInit();

  printf("GL_VENDOR: %s\n", glGetString(GL_VENDOR));
  printf("GL_VERSION: %s\n", glGetString(GL_VERSION));
  printf("GL_RENDERER: %s\n", glGetString(GL_RENDERER));

  if (!Initialize()) {
    printf("Scene initialization failed.\n");
    return 1;
  }

  while (!glfwWindowShouldClose(window)) {
    Render();
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  if (adc.isStreamRunning()){
    adc.stopStream();
  }

  glfwDestroyWindow(window);

  return 0;
}
