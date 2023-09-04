#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <kissfft/kiss_fft.h>
#include <math.h>
#include <opencv4/opencv2/core/mat.hpp>
#include <rtaudio/RtAudio.h>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/opengl.hpp>
#include <opencv2/core/ocl.hpp>

#define VERT_LENGTH 4 // x,y,z,volume

#define SPHERE_LAYERS 8
#define FRAMES 1024
#define NUM_POINTS 64
#define POST_PROC true

GLFWwindow *window;
GLuint program;
GLuint vao, vbo;
GLuint ssaoFramebufferID, ssaoDepthTextureID;
float radius = 1.0f;
RtAudio adc(RtAudio::Api::LINUX_PULSE);
GLfloat circleVertices[NUM_POINTS * VERT_LENGTH * SPHERE_LAYERS];
/* Global */
GLuint fbo, fbo_texture, fbo_texture2, rbo_depth;

GLubyte* colorBuffer = 0;
GLubyte* colorBuffer2 = 0;
cv::ogl::Texture2D texture, texture2;


kiss_fft_cfg cfg;

const unsigned int screen_width = 3840;
const unsigned int screen_height = 2160;

float rawdata[FRAMES];
float freqs[FRAMES];

float red_rawdata[NUM_POINTS];
float red_freqs[NUM_POINTS];
float last_freqs[NUM_POINTS];

double lastTime = glfwGetTime();
int nbFrames = 0;

int zx = 38;
int zy = 21;
int dilation_size = 1;
int erosion_size = 1;
float angle = 0.4f;
float sensitivity = 0.1;
float lineWidth = 30.0;

void Render();
enum VisMode { LINES, CIRCLE, CIRCLE_FLAT, SPHERE, SPHERE_SPIRAL };
VisMode mode = CIRCLE;

template <typename MAT>
MAT effect1(MAT img) {
    float f = (red_freqs[0] * 0.1);
    cv::blur(img, img, cv::Size(10,10));
    cv::addWeighted(img, 0.0, img, 0.98 - red_freqs[0] * 0.001, 0.0, img);
    MAT rot = MAT(screen_height, screen_width, CV_8UC4);
    cv::Point2f center((img.cols - 1)/2.0, (img.rows - 1)/2.0);
    cv::Mat matRotation = cv::getRotationMatrix2D(center, angle * f , 1.0 );
    cv::warpAffine(img, rot, matRotation, img.size());
    cv::Mat element = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ), cv::Point( dilation_size, dilation_size ) );
    cv::dilate(rot, rot, element);
    MAT test2 = rot(cv::Rect(zx * f, zy * f, screen_width - 2*zx*f, screen_height - 2 * zy*f));
    MAT image2 = MAT(screen_height, screen_width, CV_8UC4);
    cv::resize(test2, image2, cv::Size(screen_width, screen_height));
    return image2;
}

template <typename MAT>
MAT effect2(MAT img) {
    cv::blur(img, img, cv::Size(10,10));
    cv::addWeighted(img, 0.0, img, 0.98 - red_freqs[0] * 0.001, 0.0, img);
    MAT rot = MAT(screen_height, screen_width, CV_8UC4);
    cv::Point2f center((img.cols - 1)/2.0, (img.rows - 1)/2.0);
    cv::Mat matRotation = cv::getRotationMatrix2D(center, angle , 1.0 );
    cv::warpAffine(img, rot, matRotation, img.size());

    cv::Mat element = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ), cv::Point( dilation_size, dilation_size ) );
    cv::dilate(rot, rot, element);
    MAT image2 = MAT(screen_height, screen_width, CV_8UC4);
    cv::resize(rot, image2, cv::Size(screen_width, screen_height));
    return image2;
}

template <typename MAT>
MAT effect3(MAT img) {
    float f = (red_freqs[0] * 0.05);
    cv::blur(img, img, cv::Size(10,10));
    cv::addWeighted(img, 0.0, img, 0.98 - red_freqs[0] * 0.001, 0.0, img);
    cv::Point2f center((img.cols - 1)/2.0, (img.rows - 1)/2.0);
    //cv::Mat matRotation = cv::getRotationMatrix2D( center, angle , 1.0 );
    //cv::warpAffine(img, img, matRotation, img.size());
    
    MAT test2 = img(cv::Rect(zx * f, zy* f, screen_width - 2*zx* f, screen_height - 2 * zy * f));
    cv::Mat element = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ), cv::Point( dilation_size, dilation_size ) );
    cv::dilate(test2, test2, element);
    MAT image2 = MAT(screen_height, screen_width, CV_8UC4);
    cv::resize(test2, image2, cv::Size(screen_width, screen_height));
    return image2;
}

template <typename MAT>
MAT effect4(MAT img) {
    cv::blur(img, img, cv::Size(10,10));
    cv::addWeighted(img, 0.0, img, 0.98 - red_freqs[0] * 0.001, 0.0, img);
    cv::Point2f center((img.cols - 1)/2.0, (img.rows - 1)/2.0);
    MAT rot = MAT(screen_height, screen_width, CV_8UC4);
    cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, -20);
    cv::warpAffine(img, rot, trans_mat, img.size());
    MAT test2 = rot(cv::Rect(zx, zy, screen_width - 2*zx, screen_height - 2 * zy));
    cv::Mat element = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ), cv::Point( dilation_size, dilation_size ) );
    cv::dilate(test2, test2, element);
    MAT image2 = MAT(screen_height, screen_width, CV_8UC4);
    cv::resize(test2, image2, cv::Size(screen_width, screen_height));
    return image2;
}

template <typename MAT>
MAT effect5(MAT img) {
    cv::blur(img, img, cv::Size(20,20));
    cv::addWeighted(img, 0.0, img, 0.98 - red_freqs[0] * 0.001, 0.0, img);
    cv::Point2f center((img.cols - 1)/2.0, (img.rows - 1)/2.0);
    //cv::Mat matRotation = cv::getRotationMatrix2D( center, angle , 1.0 );
    //cv::warpAffine(img, img, matRotation, img.size());
    MAT test2 = img(cv::Rect(zx, zy, screen_width - 2*zx, screen_height - 2 * zy));
    cv::Mat element = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ), cv::Point( dilation_size, dilation_size ) );
    cv::erode(test2, test2, element);
    MAT image2 = MAT(screen_height, screen_width, CV_8UC4);
    cv::resize(test2, image2, cv::Size(screen_width, screen_height));
    return image2;
}

template <typename MAT>
MAT effect6(MAT img) {
    cv::addWeighted(img, 0.0, img, 0.98 - red_freqs[0] * 0.001, 0.0, img);
    cv::Point2f center((img.cols - 1)/2.0, (img.rows - 1)/2.0);
    //cv::Mat matRotation = cv::getRotationMatrix2D( center, angle , 1.0 );
    //cv::warpAffine(img, img, matRotation, img.size());
    MAT test2 = img(cv::Rect(zx, zy, screen_width - 2*zx, screen_height - 2 * zy));
    cv::Mat eelement = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ), cv::Point( dilation_size, dilation_size ) );
    cv::dilate(test2, test2, eelement);
    cv::Mat delement = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ), cv::Point( dilation_size, dilation_size ) );
    cv::erode(test2, test2, delement);
    MAT image2 = MAT(screen_height, screen_width, CV_8UC4);
    cv::resize(test2, image2, cv::Size(screen_width, screen_height));
    return image2;
}

#define NUM_EFFECTS 6

template <typename MAT>
MAT (*effect)(MAT) = effect1;

template <typename MAT>
MAT (*effects[NUM_EFFECTS])(MAT) = {effect1, effect2, effect3, effect4, effect5, effect6};


int record(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
           double streamTime, RtAudioStreamStatus status, void *userData) {
  if (status) {
    std::cout << "Stream overflow detected!" << std::endl;
    return 0;
  }

  // printf("%d \n", nBufferFrames);
  kiss_fft_cpx in[FRAMES] = {};
  for (unsigned int i = 0; i < nBufferFrames; i++) {
    in[i].r = ((float *)inputBuffer)[i];
    rawdata[i] = ((float *)inputBuffer)[i];
  }

  kiss_fft_cpx out[FRAMES] = {};
  kiss_fft(cfg, in, out);
  for (int i = 0; i < FRAMES; i++) {
    freqs[i] = sqrt(out[i].r * out[i].r + out[i].i * out[i].i);
  }

  int sample_group = FRAMES / NUM_POINTS;
  int fft_group = (FRAMES / 3) / NUM_POINTS;
  for (int i = 0; i < NUM_POINTS; i++) {
    red_rawdata[i] = 0;
    red_freqs[i] = 0;
    for (int j = 0; j < sample_group; j++) {
      red_rawdata[i] += rawdata[i * sample_group + j];
    }
    for (int j = 0; j < fft_group; j++) {
      red_freqs[i] += freqs[i * fft_group + j + 25];
    }
    red_freqs[i] += last_freqs[i];
    red_freqs[i] /= 2.0;
  }

  for (int i = 0; i < NUM_POINTS; i++) {
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
    std::cout << ": maximum input channels = " << info.inputChannels
              << std::endl;
    std::cout << ": maximum output channels = " << info.outputChannels
              << std::endl;
  }
}

static void set_camera(float cam_x, float cam_y, float cam_z, float target_z) {
  glUseProgram(program);
  glm::mat4 trans = glm::mat4(1.0f);
  trans = glm::rotate(trans, glm::radians(0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
  GLint uniTrans = glGetUniformLocation(program, "model");
  glUniformMatrix4fv(uniTrans, 1, GL_FALSE, glm::value_ptr(trans));

  glm::mat4 view =
      glm::lookAt(glm::vec3(cam_x, cam_y, cam_z),
                  glm::vec3(0.0f, 0.0f, target_z), glm::vec3(0.0f, 0.0f, 1.0f));
  GLint uniView = glGetUniformLocation(program, "view");
  glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));

  glm::mat4 proj =
      glm::perspective(glm::radians(45.0f), 3840.0f / 2160.0f, 1.0f, 10.0f);
  GLint uniProj = glGetUniformLocation(program, "proj");
  glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action,
                         int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  } else if (key == GLFW_KEY_L) {
    mode = VisMode::LINES;
  } else if (key == GLFW_KEY_C) {
    mode = VisMode::CIRCLE;
  } else if (key == GLFW_KEY_V) {
    mode = VisMode::CIRCLE_FLAT;
  } else if (key == GLFW_KEY_S) {
    mode = VisMode::SPHERE;
  } else if (key == GLFW_KEY_X) {
    mode = VisMode::SPHERE_SPIRAL;
  } else if (key == GLFW_KEY_F) {
    set_camera(0.0f, -2.5f, 2.5f, 0.2f);
  } else if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9 && action == GLFW_PRESS) {
      int index = (key - GLFW_KEY_0);
      if (index < NUM_EFFECTS) {
        effect<cv::Mat> = effects<cv::Mat>[index];
        effect<cv::UMat> = effects<cv::UMat>[index];
        printf("Using effect %d\n", index);
      }
  }
    
}

static void resize(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
  glUseProgram(program);

  GLfloat uResolution[2] = {(float)width, (float)height};
  glUniform2fv(glGetUniformLocation(program, "uResolution"), 1, uResolution);

  glm::mat4 proj = glm::perspective(glm::radians(45.0f),
                                    (float)width / (float)height, 1.0f, 10.0f);
  GLint uniProj = glGetUniformLocation(program, "proj");
  glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));

  if(POST_PROC){
    glBindTexture(GL_TEXTURE_2D, fbo_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screen_width, screen_height, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);

    glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, screen_width,
                          screen_height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
  }
}

// Get the depth buffer value at this pixel.   
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

bool loadShaders(GLuint *program, std::vector<std::tuple<GLenum, std::string, std::string>> shaders) {
  for (const auto &s : shaders) {
    GLenum type = std::get<0>(s);
    const std::string &source = std::get<1>(s);

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
        printf("Shader (%s) compile failed:\n%s\n", source.c_str(), log.c_str());
      } else {
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
    } else {
      printf("Program (%s) link failed.\n", std::get<2>(shaders[0]).c_str());
    }

    return false;
  }
  return true;
}

bool Initialize() {
  getdevices();

  RtAudio::StreamParameters parameters;
  parameters.deviceId = adc.getDefaultInputDevice();
  // parameters.deviceId = 132;
  parameters.nChannels = 1;
  parameters.firstChannel = 0;
  unsigned int sampleRate = 48000;
  unsigned int bufferFrames = FRAMES;

  cfg = kiss_fft_alloc(FRAMES, 0, NULL, NULL);

  if (adc.openStream(NULL, &parameters, RTAUDIO_FLOAT32, sampleRate,
                     &bufferFrames, &record)) {
    std::cout << '\n' << adc.getErrorText() << '\n' << std::endl;
    exit(0); // problem with device settings
  }

  // Stream is open ... now start it.
  if (adc.startStream()) {
    std::cout << adc.getErrorText() << std::endl;
  }

  printf("Creating program \n");
  program = glCreateProgram();
  if (!loadShaders(&program, loadShaderContent("../vertex.glsl", "../fragment.glsl"))) {
    return false;
  }

  printf("Created program \n");
  set_camera(0.0f, -0.1f, 5.0f, 0.0f);


   if(POST_PROC){
      /* Texture */
      colorBuffer = new GLubyte[screen_width * screen_height * 4];
      memset(colorBuffer, 0, screen_width * screen_height * 4);

      colorBuffer2 = new GLubyte[screen_width * screen_height * 4];
      memset(colorBuffer2, 0, screen_width * screen_height * 4);

      
      glActiveTexture(GL_TEXTURE0);
      glGenTextures(1, &fbo_texture);
      glBindTexture(GL_TEXTURE_2D, fbo_texture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screen_width, screen_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
      texture = cv::ogl::Texture2D(cv::Size(screen_width, screen_height), cv::ogl::Texture2D::Format::RGBA, fbo_texture, false);
      glBindTexture(GL_TEXTURE_2D, 0);

      //glActiveTexture(GL_TEXTURE1);
      glGenTextures(1, &fbo_texture2);
      glBindTexture(GL_TEXTURE_2D, fbo_texture2);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screen_width, screen_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
      texture2 = cv::ogl::Texture2D(cv::Size(screen_width, screen_height), cv::ogl::Texture2D::Format::RGBA, fbo_texture2, false);
      glBindTexture(GL_TEXTURE_2D, 0);


      /* Depth buffer */
      glGenRenderbuffers(1, &rbo_depth);
      glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth);
      glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, screen_width, screen_height);
      glBindRenderbuffer(GL_RENDERBUFFER, 0);

      /* Framebuffer to link everything together */
      glGenFramebuffers(1, &fbo);
      glBindFramebuffer(GL_FRAMEBUFFER, fbo);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_texture, 0);
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo_depth);

      GLenum status;
      if ((status = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "glCheckFramebufferStatus: error %p",
                glewGetErrorString(status));
        return 0;
      }
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
   }



  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);

  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(circleVertices), circleVertices, GL_STATIC_DRAW);
  glUseProgram(program);

  glVertexAttribPointer(0, VERT_LENGTH, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (GLvoid*)(4 * sizeof(GLfloat)));
  glEnableVertexAttribArray(1);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  glEnable(GL_PROGRAM_POINT_SIZE);
  glLineWidth(lineWidth);

  printf("Init finished \n");
  return true;
}

float cur = 0.0;

cv::Mat image = cv::Mat(screen_height, screen_width, CV_8UC4);



void Render() {

  if(POST_PROC){
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  }
  glClear(GL_COLOR_BUFFER_BIT);

  double currentTime = glfwGetTime();
  nbFrames++;
  if (currentTime - lastTime >=
      1.0) { // If last prinf() was more than 1 sec ago
    // printf and reset timer
    printf("%f ms/frame\n", 1000.0 / double(nbFrames));
    nbFrames = 0;
    lastTime += 1.0;
  }

  std::vector<float> vertices;

  if (mode == LINES) {
    for (int i = 0; i < NUM_POINTS; i++) {
      float vert[5] = {i * (4.0f / NUM_POINTS) - 2.0f, red_freqs[i] * sensitivity, 0.0, red_freqs[i] * sensitivity, (float)i/NUM_POINTS};
      vertices.insert(vertices.end(), std::begin(vert), std::end(vert));
    }
  } else if (mode == CIRCLE) {
    for (int i = 0; i < NUM_POINTS; i++) {
      float theta = 2.0f * M_PI * float(i) / float(NUM_POINTS) - M_PI;
      float r = red_freqs[i] * sensitivity + 0.5;

      float x = radius * r * cosf(theta);
      float y = radius * r * sinf(theta);

      float vert[5] = {x, y, 0.0, red_freqs[i] * sensitivity, (float)i/NUM_POINTS};
      vertices.insert(vertices.end(), std::begin(vert), std::end(vert));
    }
  } else if (mode == CIRCLE_FLAT) {
    for (int i = 0; i < NUM_POINTS; i++) {
      float theta1 = 2.0f * M_PI * float(i) / float(NUM_POINTS) - M_PI;
      float theta2 = 2.0f * M_PI * float(i + 1) / float(NUM_POINTS) - M_PI;
      float r = red_freqs[i] * sensitivity + 0.5;

      float c[5] = {0.0, 0.0, 0.0, 0.0, (float)i/NUM_POINTS};
      vertices.insert(vertices.end(), std::begin(c), std::end(c));
      float a[5] = {radius * r * cosf(theta1), radius * r * sinf(theta1), 0.0,
                    red_freqs[i] * sensitivity, (float)i/NUM_POINTS};
      vertices.insert(vertices.end(), std::begin(a), std::end(a));
      float b[5] = {radius * r * sinf(theta2), radius * r * sinf(theta2), 0.0,
                    0.0, (float)i/NUM_POINTS};
      vertices.insert(vertices.end(), std::begin(b), std::end(b));
    }
  } else if (mode == SPHERE) {
    for (int c = 0; c < SPHERE_LAYERS; c++) {
      for (int i = 0; i < NUM_POINTS; i++) {
        float theta = 2.0f * M_PI * float(i) / float(NUM_POINTS) - M_PI;

        float r = red_freqs[i] * sensitivity + 0.5;
        float layer = sin(((float)c / (float)SPHERE_LAYERS) * M_PI);
        float x = radius * layer * r * cosf(theta);
        float y = radius * layer * r * sinf(theta);

        float vert[5] = {x, y, c * 0.2f, red_freqs[i] * sensitivity, (float)i/NUM_POINTS};
        vertices.insert(vertices.end(), std::begin(vert), std::end(vert));
      }
    }
  } else if (mode == SPHERE_SPIRAL) {
    for (int i = 0; i < NUM_POINTS; i++) {
      float theta = 5.0f * 2.0f * M_PI * float(i) / float(NUM_POINTS) - M_PI;

      float r = red_freqs[i] * sensitivity + 0.5;
      float percent = ((float)i / (float)(NUM_POINTS));
      float layer = sin(percent * M_PI);
      float x = radius * layer * r * cosf(theta);
      float y = radius * layer * r * sinf(theta);

      float vert[5] = {x, y, percent * 2.0f, red_freqs[i] * sensitivity, (float)i/NUM_POINTS};
      vertices.insert(vertices.end(), std::begin(vert), std::end(vert));
    }
  }

  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Use the shader program
  glUseProgram(program);

  // Bind vertex array object (VAO)
  glBindVertexArray(vao);

  // Draw the circle as a line loop
  if (mode == LINES) {
    glDrawArrays(GL_LINE_STRIP, 0, NUM_POINTS);
    glDrawArrays(GL_POINTS, 0, NUM_POINTS);
  } else if (mode == CIRCLE) {
    glDrawArrays(GL_LINE_STRIP, 0, NUM_POINTS);
    glDrawArrays(GL_POINTS, 0, NUM_POINTS);
  } else if (mode == CIRCLE_FLAT) {
    glDrawArrays(GL_LINE_STRIP, 0, NUM_POINTS * 3);
  } else {
    if (mode == SPHERE) {
      glDrawArrays(GL_LINE_STRIP_ADJACENCY_EXT, 0, SPHERE_LAYERS * NUM_POINTS);
      glDrawArrays(GL_POINTS, 0, SPHERE_LAYERS * NUM_POINTS);
    } else {
      glDrawArrays(GL_LINE_STRIP, 0, NUM_POINTS);
    }
    cur += 0.01;
    set_camera(sin(cur), cos(cur), 4.0f, 0.0f);
  }

  // Unbind VAO and shader
  glBindVertexArray(0);
  glUseProgram(0);

  if(POST_PROC){
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if(true) {
      cv::UMat u1, u2;
      cv::ogl::convertFromGLTexture2D(texture, u1);
      cv::ogl::convertFromGLTexture2D(texture2, u2);
      cv::add(u1, effect<cv::UMat>(u2), u1);
      
      cv::ogl::convertToGLTexture2D(u1, texture);
      u1.copyTo(u2);
      cv::ogl::convertToGLTexture2D(u2, texture2);


      
      texture.bind();
      glEnable(GL_TEXTURE_2D);
      glBegin(GL_QUADS);
      glTexCoord2f(0.0f, 0.0f);
      glVertex2f(-1.0f, -1.0f);
      glTexCoord2f(1.0f, 0.0f);
      glVertex2f(1.0f, -1.0f);
      glTexCoord2f(1.0f, 1.0f);
      glVertex2f(1.0f, 1.0f);
      glTexCoord2f(0.0f, 1.0f);
      glVertex2f(-1.0f, 1.0f);
      glEnd();
      glDisable(GL_TEXTURE_2D);

    } else{

        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
        glReadPixels(0, 0, screen_width, screen_height, GL_RGBA, GL_UNSIGNED_BYTE, colorBuffer);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

        if (true) {
          image.data = colorBuffer;
          image = image + cv::Mat(screen_height, screen_width, CV_8UC4, colorBuffer2);
          cv::Mat image2 = effect<cv::Mat>(image);
          glDrawPixels(screen_width, screen_height, GL_RGBA, GL_UNSIGNED_BYTE, image2.data);
          memcpy(colorBuffer2, image2.data, screen_width * screen_height * 4);
        } else {
          glDrawPixels(screen_width, screen_height, GL_RGBA, GL_UNSIGNED_BYTE, colorBuffer);      
        }
    }
  }
}

int main() {

  if (!glfwInit())
    exit(EXIT_FAILURE);

  //glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
  //window = glfwCreateWindow(screen_width, screen_height, "My Title", NULL, NULL);
  window = glfwCreateWindow(screen_width, screen_height, "My Title", glfwGetPrimaryMonitor(), nullptr);
  printf("Hi.\n");
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

  if (cv::ocl::haveOpenCL())
  {
      cv::ogl::ocl::initializeContextFromGL();
  }

  if (!Initialize()) {
    printf("Scene initialization failed.\n");
    return 1;
  }

  while (!glfwWindowShouldClose(window)) {
    Render();
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  if (adc.isStreamRunning()) {
    adc.stopStream();
  }

  /* free_resources */
  glDeleteRenderbuffers(1, &rbo_depth);
  glDeleteTextures(1, &fbo_texture);
  glDeleteFramebuffers(1, &fbo);

  glfwDestroyWindow(window);

  return 0;
}
