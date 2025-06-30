
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GLFW/glfw3.h>
#include <opencv2/core/hal/interface.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "helpers.h"
#ifdef _WIN32
#include <kiss_fft.h>
#else
#include <kissfft/kiss_fft.h>
#endif
#include <math.h>
#include <rtaudio/RtAudio.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "3rdparty/beatdetektor/cpp/BeatDetektor.h"
#include "audioanalyzer.h"
#include "colors.h"
#include "font_rendering.h"

using namespace std::chrono;
namespace fs = std::filesystem;

#define VERT_LENGTH 5  // x,y,z,frequency, volume
#define SPHERE_LAYERS 8
#define FRAMES 1024
#define NUM_POINTS FRAMES / 2
#define SHADER_PATH "../"

GLFWwindow *window;
GLuint program, font_program, pixel_program;
GLuint vao, vbo, vao2, vbo2;
GLuint ssaoFramebufferID, ssaoDepthTextureID;

RtAudio adc(RtAudio::Api::LINUX_PULSE);
GLfloat circleVertices[NUM_POINTS * VERT_LENGTH * SPHERE_LAYERS];
/* Global */
GLuint fbo, fbo_texture, fbo_texture2, rbo_depth;

cv::ogl::Texture2D texture, texture2;
cv::UMat background;

AudioAnalyzer *aanalyzer;

kiss_fft_cfg cfg;

unsigned int screen_width = 3840;
unsigned int screen_height = 2160;

std::vector<float> left_frequencies, right_frequencies;

double lastTime = glfwGetTime();
int nbFrames = 0;

int current_file_index = 0;
int current_font_index = 0;

int zx = 38;
int zy = 21;
int dilation_size = 1;
int erosion_size = 1;
float angle = 0.4f;
float sensitivity = 1.0;
float zoom_sensitivity = 0.5;
float line_width = 10.0;
float inner_radius = 1.0;
float radius = 0.5f;

bool post_processing_enabled = true;
bool reactive_zoom_enabled = false;
int color_mode = 0;
int background_mode = 0;
bool dynamic_color = false;
float cur = 0.0;
float zoom_speed;
milliseconds start;

cv::Mat image;
GLfloat color[4] = {1.0, 0.0, 0.0, 1.0};

enum VisMode {
    LINES,
    CIRCLE,
    CIRCLE_FLAT,
    SPHERE,
    SPHERE_SPIRAL,
    TEXT
};
enum BackgroundMode {
    OFF,
    ON,
    BEAT
};
VisMode mode = CIRCLE;

int stereo_mode = 2;
int color_cycle = 0;
milliseconds last_ms;
double lastBeat;
bool beat = false;
BeatDetektor *bd;
float reactive_frequency;
bool inverted = false;

cv::UMat u1, u2, u3;

cv::UMat effect1(cv::UMat img) {
    cv::addWeighted(img, 0.0, img, 0.9 - reactive_frequency * 0.001, 0.0, img);
    cv::blur(img, img, cv::Size(10, 10));
    
    cv::UMat rot;
    cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
    cv::Mat matRotation = cv::getRotationMatrix2D(center, angle * zoom_speed, 1.0);
    cv::warpAffine(img, rot, matRotation, img.size());
    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
    cv::dilate(rot, rot, element);
    cv::UMat test2, image2;

    if (inverted) {
        test2 = rot(cv::Rect(zx * zoom_speed, zy * zoom_speed, screen_width - 2 * zx * zoom_speed, screen_height - 2 * zy * zoom_speed));
        image2 = cv::UMat(screen_height, screen_width, CV_8UC4);
        cv::resize(test2, image2, cv::Size(screen_width, screen_height));
    } else {
        cv::resize(img, test2, cv::Size(screen_width - 32, screen_height - 18), 0.1, 0.1);
        cv::copyMakeBorder(test2, image2, 9, 9, 16, 16, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    }
    //cv::resize(img, test2, cv::Size(screen_width/2, screen_height/2), 0.1, 0.1);
    //cv::resize(test2, image2, cv::Size(screen_width, screen_height));

    return image2;
}

cv::UMat effect2(cv::UMat img) {
    cv::addWeighted(img, 0.0, img, 0.9 - reactive_frequency * 0.001, 0.0, img);
    cv::blur(img, img, cv::Size(10, 10));
    cv::UMat rot;
    cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
    cv::Mat matRotation = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(img, rot, matRotation, img.size());

    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
    cv::dilate(rot, rot, element);
    cv::UMat image2 = cv::UMat(screen_height, screen_width, CV_8UC4);
    cv::resize(rot, image2, cv::Size(screen_width, screen_height));
    return image2;
}

cv::UMat effect3(cv::UMat img) {
    cv::blur(img, img, cv::Size(10, 10));
    cv::addWeighted(img, 0.0, img, 0.98 - reactive_frequency * 0.001, 0.0, img);
    cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
    // cv::Mat matRotation = cv::getRotationMatrix2D( center, angle , 1.0 );
    // cv::warpAffine(img, img, matRotation, img.size());

    cv::UMat test1, test2;
    if (inverted) {
        test2 = img(cv::Rect(zx * zoom_speed, zy * zoom_speed, screen_width - 2 * zx * zoom_speed, screen_height - 2 * zy * zoom_speed));
    } else {
        cv::resize(img, test1, cv::Size(screen_width - 32, screen_height - 18), 0.1, 0.1);
        cv::copyMakeBorder(test1, test2, 9, 9, 16, 16, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    }
    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
    cv::dilate(test2, test2, element);
    cv::UMat image2 = cv::UMat(screen_height, screen_width, CV_8UC4);
    cv::resize(test2, image2, cv::Size(screen_width, screen_height));
    return image2;
}

cv::UMat effect4(cv::UMat img) {
    cv::blur(img, img, cv::Size(10, 10));
    cv::addWeighted(img, 0.0, img, 0.98 - reactive_frequency * 0.001, 0.0, img);
    cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
    cv::UMat rot;
    cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, -20);
    cv::warpAffine(img, rot, trans_mat, img.size());
    cv::UMat test2 = rot(cv::Rect(zx * zoom_speed, zy * zoom_speed, screen_width - 2 * zx * zoom_speed, screen_height - 2 * zy * zoom_speed));
    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
    cv::dilate(test2, test2, element);
    cv::UMat image2 = cv::UMat(screen_height, screen_width, CV_8UC4);
    cv::resize(test2, image2, cv::Size(screen_width, screen_height));
    return image2;
}

cv::UMat effect5(cv::UMat img) {
    cv::blur(img, img, cv::Size(20, 20));
    cv::addWeighted(img, 0.0, img, 0.98 - reactive_frequency * 0.001, 0.0, img);
    cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
    // cv::Mat matRotation = cv::getRotationMatrix2D( center, angle , 1.0 );
    // cv::warpAffine(img, img, matRotation, img.size());
    cv::UMat test2 = img(cv::Rect(zx * zoom_speed, zy * zoom_speed, screen_width - 2 * zx * zoom_speed, screen_height - 2 * zy * zoom_speed));
    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
    cv::erode(test2, test2, element);
    cv::UMat image2 = cv::UMat(screen_height, screen_width, CV_8UC4);
    cv::resize(test2, image2, cv::Size(screen_width, screen_height));
    return image2;
}

cv::UMat effect6(cv::UMat img) {
    cv::blur(img, img, cv::Size(20, 20));
    cv::addWeighted(img, 0.0, img, 0.98 - reactive_frequency * 0.001, 0.0, img);
    cv::UMat test2 = img(cv::Rect(zx * zoom_speed, zy * zoom_speed, screen_width - 2 * zx * zoom_speed, screen_height - 2 * zy * zoom_speed));
    cv::UMat image2 = cv::UMat(screen_height, screen_width, CV_8UC4);
    cv::resize(test2, image2, cv::Size(screen_width, screen_height));
    return image2;
}

cv::UMat effect7(cv::UMat img) {
    cv::addWeighted(img, 0.0, img, 0.98 - reactive_frequency * 0.001, 0.0, img);
    cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
    // cv::Mat matRotation = cv::getRotationMatrix2D( center, angle , 1.0 );
    // cv::warpAffine(img, img, matRotation, img.size());
    cv::UMat test2 = img(cv::Rect(zx * zoom_speed, zy * zoom_speed, screen_width - 2 * zx * zoom_speed, screen_height - 2 * zy * zoom_speed));
    cv::Mat eelement = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
    cv::dilate(test2, test2, eelement);
    cv::Mat delement = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(dilation_size, dilation_size));
    cv::erode(test2, test2, delement);
    cv::UMat image2 = cv::UMat(screen_height, screen_width, CV_8UC4);
    cv::resize(test2, image2, cv::Size(screen_width, screen_height));
    return image2;
}

cv::UMat effect8(cv::UMat img) {
    cv::blur(img, img, cv::Size(20, 20));
    cv::addWeighted(img, 0.0, img, 0.98 - reactive_frequency * 0.001, 0.0, img);
    auto image_rect = cv::Rect({}, img.size());
    auto roi = cv::Rect(zx, zy, screen_width, screen_height + 2 * zy * zoom_speed);
    auto intersection = image_rect & roi;
    auto inter_roi = intersection - roi.tl();
    cv::Mat test2 = cv::Mat::zeros(roi.size(), image.type());
    img(intersection).copyTo(test2(inter_roi));
    cv::UMat image2 = cv::UMat(screen_height, screen_width, CV_8UC4);
    cv::resize(test2, image2, cv::Size(screen_width, screen_height));
    return image2;
}

cv::UMat effect9(cv::UMat img) {
    cv::blur(img, img, cv::Size(20, 20));
    cv::addWeighted(img, 0.0, img, 0.98 - reactive_frequency * 0.001, 0.0, img);
    return img;
}

cv::UMat effect10(cv::UMat img) {
    cv::blur(img, img, cv::Size(20, 20));
    cv::addWeighted(img, 0.0, img, 0.98 - reactive_frequency * 0.001, 0.0, img);
    cv::UMat test = cv::UMat(screen_height - 2 * zy * zoom_speed, screen_width, CV_8UC4);
    cv::resize(img, test, cv::Size(screen_width, screen_height - 2 * zy * zoom_speed));
    auto image_rect = cv::Rect({}, test.size());
    auto roi = cv::Rect(0, 0, screen_width, screen_height);
    auto intersection = image_rect & roi;
    auto inter_roi = intersection - roi.tl();
    cv::Mat test2 = cv::Mat::zeros(roi.size(), image.type());
    test(intersection).copyTo(test2(inter_roi));
    cv::UMat image2 = cv::UMat(screen_height, screen_width, CV_8UC4);
    cv::resize(test2, image2, cv::Size(screen_width, screen_height));
    return image2;
}

#define NUM_EFFECTS 10
cv::UMat (*effect)(cv::UMat) = effect1;
cv::UMat (*effects[NUM_EFFECTS])(cv::UMat) = {effect1, effect2, effect3, effect4, effect5, effect6, effect7, effect8, effect9, effect10};

void load_font() {
    std::string path = SHADER_PATH + std::string("fonts/");
    std::string new_file;
    int cur_idx = 0;
    for (const auto &entry : fs::directory_iterator(path)) {
        if (entry.path().filename().string().ends_with("ttf") || entry.path().filename().string().ends_with("otf")) {
            if (cur_idx == current_font_index) {
                new_file = entry.path().string();
            }
            cur_idx += 1;
        }
    }
    if (!new_file.empty()) {
        LoadFontRendering(new_file);
    }
    current_font_index += 1;
    current_font_index %= cur_idx;
}

void load_background() {
    std::string path = SHADER_PATH + std::string("logos/");
    std::string new_file;
    int cur_idx = 0;
    for (const auto &entry : fs::directory_iterator(path)) {
        if (entry.path().filename().string().ends_with("png") || entry.path().filename().string().ends_with("jpg")) {
            if (cur_idx == current_file_index) {
                new_file = entry.path().string();
            }

            cur_idx += 1;
        }
    }
    if (!new_file.empty()) {
        cv::Mat black = cv::Mat(cv::Size(screen_width, screen_height), CV_8UC4, cv::Scalar(0, 0, 0, 0));
        cv::Mat image = cv::imread(new_file, cv::IMREAD_COLOR);

        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        cv::Mat alpha = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC1);
        channels.push_back(alpha);
        cv::merge(channels, image);

        cv::Rect roi(black.cols / 2.0 - (image.cols / 2.0), black.rows / 2.0 - (image.rows / 2.0), image.cols, image.rows);
        cv::flip(image, image, 0);
        image.copyTo(black(roi));

        black.copyTo(background);
    }
    current_file_index += 1;
    current_file_index %= cur_idx;
}

static void set_camera(glm::vec3 cam, glm::vec3 target = glm::vec3(0.0, 0.0, 0.0)) {
    GLuint programs[2] = {program, font_program};
    for (int i = 0; i < 2; i++) {
        glUseProgram(programs[i]);

        glm::mat4 trans = glm::mat4(1.0f);
        trans = glm::rotate(trans, glm::radians(0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        GLint uniTrans = glGetUniformLocation(programs[i], "model");
        glUniformMatrix4fv(uniTrans, 1, GL_FALSE, glm::value_ptr(trans));

        glm::mat4 view = glm::lookAt(cam, target, glm::vec3(0.0f, 0.0f, 1.0f));
        GLint uniView = glGetUniformLocation(programs[i], "view");
        glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));

        glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)screen_width / (float)screen_height, 0.1f, 40.0f);
        GLint uniProj = glGetUniformLocation(programs[i], "proj");
        glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));
    }
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {

    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        } else if (key == GLFW_KEY_W) {
            set_camera(glm::vec3(0.0f, -0.1f, 4.0f));
        } else if (key == GLFW_KEY_A) {
            set_camera(glm::vec3(1.0f, -0.1f, 6.0f), glm::vec3(1.0, 0.0,0.2));
        } else if (key == GLFW_KEY_S) {
            set_camera(glm::vec3(0.0f, -0.1f, 7.0f));
        } else if (key == GLFW_KEY_D) {
            set_camera(glm::vec3(-1.0f, -0.1f, 6.0f), glm::vec3(-1.0, 0.0,0.2));
        } else if (key == GLFW_KEY_L) {
            mode = VisMode::LINES;
        } else if (key == GLFW_KEY_C) {
            mode = VisMode::CIRCLE;
        } else if (key == GLFW_KEY_V) {
            mode = VisMode::CIRCLE_FLAT;
        } else if (key == GLFW_KEY_S) {
            mode = VisMode::SPHERE;
        } else if (key == GLFW_KEY_H) {
            stereo_mode += 1;
            stereo_mode %= 2;
        }else if (key == GLFW_KEY_X) {
            mode = VisMode::SPHERE_SPIRAL;
        } else if (key == GLFW_KEY_T) {
            mode = VisMode::TEXT;
        } else if (key == GLFW_KEY_I) {
            load_font();
        } else if (key == GLFW_KEY_P) {
            post_processing_enabled = !post_processing_enabled;
        } else if (key == GLFW_KEY_F) {
            set_camera(glm::vec3(0.0f, -2.5f, 2.5f), glm::vec3(0.0, 0.0,0.2));
        } else if (key == GLFW_KEY_R) {
            set_camera(glm::vec3(0.0f, -0.1f, 6.0f));
        } else if (key == GLFW_KEY_Z) {
            reactive_zoom_enabled = !reactive_zoom_enabled;
        } else if (key == GLFW_KEY_K) {
            inverted = !inverted;
        } else if (key == GLFW_KEY_G) {
            printf("Ceps \n");
            aanalyzer->ceps = !aanalyzer->ceps;
        }else if (key == GLFW_KEY_D) {
            color_mode += 1;
            color_mode %= 4;
        } else if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9) {
            int index = (key - GLFW_KEY_0);
            printf("Usietf %d\n", index);
            if (index < NUM_EFFECTS) {
                effect = effects[index];
                printf("Using effect %d\n", index);
            }
        } else if (key == GLFW_KEY_B) {
            background_mode += 1;
            background_mode %= 3;
        } else if (key == GLFW_KEY_KP_DECIMAL) {
            load_background();
        } else if (key == GLFW_KEY_F11) {
            GLFWmonitor *monitor = glfwGetPrimaryMonitor();
            const GLFWvidmode *mode = glfwGetVideoMode(monitor);
            glfwSetWindowMonitor(window, glfwGetPrimaryMonitor(), 0, 0, screen_width, screen_height, mode->refreshRate);
        } else if (key == GLFW_KEY_KP_ADD) {
            zoom_sensitivity += 0.05;
            zoom_sensitivity = std::clamp(zoom_sensitivity, 0.0f, 5.0f);
        } else if (key == GLFW_KEY_KP_SUBTRACT) {
            zoom_sensitivity -= 0.05;
            zoom_sensitivity = std::clamp(zoom_sensitivity, 0.0f, 5.0f);
        } else if (key == GLFW_KEY_KP_6) {
            line_width += 2.0;
            line_width = std::clamp(line_width, 2.0f, 10.0f);
        } else if (key == GLFW_KEY_KP_9) {
            line_width -= 2.0;
            line_width = std::clamp(line_width, 2.0f, 10.0f);
        } else if (key == GLFW_KEY_KP_5) {
            inner_radius -= 0.1;
            inner_radius = std::clamp(inner_radius, 0.0f, 10.0f);
        } else if (key == GLFW_KEY_KP_8) {
            inner_radius += 0.1;
            inner_radius = std::clamp(inner_radius, 0.0f, 10.0f);
        } else if (key == GLFW_KEY_KP_4) {
            sensitivity -= 0.1;
            sensitivity = std::clamp(sensitivity, 0.0f, 10.0f);
        } else if (key == GLFW_KEY_KP_7) {
            sensitivity += 0.1;
            sensitivity = std::clamp(sensitivity, 0.0f, 10.0f);
        }
    }

    
}

bool Initialize();

static void resize(GLFWwindow *window, int width, int height) {
    if (width == 0 || height == 0) {
        return;
    }

    printf("Resized %d, %d\n", width, height);
    // glViewport(0, 0, width, height);

    GLuint programs[2] = {program, font_program};
    for (int i = 0; i < 2; i++) {
        glUseProgram(programs[i]);

        GLfloat uResolution[2] = {(float)width, (float)height};
        glUniform2fv(glGetUniformLocation(programs[i], "iResolution"), 1, uResolution);

        glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 1.0f, 10.0f);
        glUniformMatrix4fv(glGetUniformLocation(programs[i], "proj"), 1, GL_FALSE, glm::value_ptr(proj));
    }

    if (post_processing_enabled) {
        glBindTexture(GL_TEXTURE_2D, fbo_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screen_width, screen_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);

        glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, screen_width, screen_height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }
}

bool Initialize() {
    aanalyzer = new AudioAnalyzer();
    aanalyzer->getdevices();
    aanalyzer->startRecording();
    load_background();

    u2 = cv::UMat(cv::Size(screen_width, screen_height), CV_8UC4);
    image = cv::Mat(screen_height, screen_width, CV_8UC4);
    bd = new BeatDetektor();

    start = duration_cast<milliseconds>(
        system_clock::now().time_since_epoch());

    printf("Creating program \n");
    program = glCreateProgram();
    if (!loadShaders(&program, loadShaderContent(SHADER_PATH + std::string("vertex.glsl"), SHADER_PATH + std::string("fragment.glsl")))) {
        return false;
    }
    font_program = glCreateProgram();
    if (!loadShaders(&font_program, loadShaderContent(SHADER_PATH + std::string("font_vertex.glsl"), SHADER_PATH + std::string("font_fragment.glsl")))) {
        return false;
    }
    pixel_program = glCreateProgram();
    if (!loadShaders(&pixel_program, loadShaderContent(SHADER_PATH + std::string("pixel_vertex.glsl"), SHADER_PATH + std::string("pixel_fragment.glsl")))) {
        return false;
    }
    printf("Created program \n");
    set_camera(glm::vec3(0.0f, -0.01f, 6.0f));

    if (post_processing_enabled) {
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

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERT_LENGTH * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, VERT_LENGTH * sizeof(float), (GLvoid *)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, VERT_LENGTH * sizeof(float), (GLvoid *)(4 * sizeof(GLfloat)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);

    GLfloat pixelVertices[12] = {-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0};

    glGenVertexArrays(1, &vao2);
    glGenBuffers(1, &vbo2);

    glBindVertexArray(vao2);
    glBindBuffer(GL_ARRAY_BUFFER, vbo2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(pixelVertices), pixelVertices, GL_STATIC_DRAW);
    glUseProgram(pixel_program);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);

    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_CULL_FACE);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //glEnable(GL_COLORSPACE_SCRGB_LINEAR_EXT);

    printf("Init finished \n");
    return true;
}

static int base = 40;
static float f(float x) {
    return (pow(base, x) - 1.0) / (base - 1.0);
}
static float g(float x) {
    return log(x * (base - 1.0) + 1.0) / log(base);
}

std::vector<float> create_vbo(std::vector<float> frequencies, float range) {
    std::vector<float> vertices;

    if (mode == LINES) {
        for (int i = 0; i < NUM_POINTS; i++) {
            float pct_frq = g(float(i) / float(NUM_POINTS));
            std::vector<float> vert;
            if (range == 1.0) {
                vert = {static_cast<float>(4.0 * (pct_frq * (2.0f) - 1.0f) - 2.0), frequencies[i] * sensitivity, 0.0, frequencies[i], pct_frq};
            } else {
                vert = {static_cast<float>(range * 4.0 * (pct_frq * (2.0f) - 1.0f) + range * 4.0), frequencies[i] * sensitivity, 0.0, frequencies[i], pct_frq};
            }
            vertices.insert(vertices.end(), std::begin(vert), std::end(vert));
        }
    } else if (mode == CIRCLE) {
        for (int i = 0; i < NUM_POINTS; i++) {
            float pct_frq = g(float(i) / float(NUM_POINTS));
            float theta = -2.0f * range *  M_PI * pct_frq - M_PI_2;
            float r = frequencies[i] * sensitivity + inner_radius;

            float x = radius * r * cosf(theta);
            float y = radius * r * sinf(theta);

            float vert[VERT_LENGTH] = {x, y, 0.0, frequencies[i], pct_frq};
            vertices.insert(vertices.end(), std::begin(vert), std::end(vert));
        }
    } else if (mode == CIRCLE_FLAT) {
        for (int i = 0; i < NUM_POINTS; i++) {
            float theta1 = 2.0f * M_PI * float(i) / float(NUM_POINTS) - M_PI;
            float theta2 = 2.0f * M_PI * float(i + 1) / float(NUM_POINTS) - M_PI;
            float r = frequencies[i] * sensitivity + inner_radius;

            float c[VERT_LENGTH] = {0.0, 0.0, 0.0, 0.0, (float)i / NUM_POINTS};
            vertices.insert(vertices.end(), std::begin(c), std::end(c));
            float b[VERT_LENGTH] = {radius * r * sinf(theta2), radius * r * sinf(theta2), 0.0, 0.0, (float)i / NUM_POINTS};
            vertices.insert(vertices.end(), std::begin(b), std::end(b));
            float a[VERT_LENGTH] = {radius * r * cosf(theta1), radius * r * sinf(theta1), 0.0, frequencies[i], (float)i / NUM_POINTS};
            vertices.insert(vertices.end(), std::begin(a), std::end(a));
        }
    } else if (mode == SPHERE) {
        for (int c = 0; c < SPHERE_LAYERS; c++) {
            for (int i = 0; i < NUM_POINTS; i++) {
                float pct_frq = g(float(i) / float(NUM_POINTS));
                float theta = 2.0f * M_PI * pct_frq - M_PI;

                float r = frequencies[i] * sensitivity + 0.5;
                float layer = sin(((float)c / (float)SPHERE_LAYERS) * M_PI);
                float x = radius * layer * r * cosf(theta);
                float y = radius * layer * r * sinf(theta);

                float vert[VERT_LENGTH] = {x, y, c * 0.2f, frequencies[i], (float)i / NUM_POINTS};
                vertices.insert(vertices.end(), std::begin(vert), std::end(vert));
            }
        }
    } else if (mode == SPHERE_SPIRAL) {
        for (int i = 0; i < NUM_POINTS; i++) {
            float pct_frq = g(float(i) / float(NUM_POINTS));
            float theta = 5.0f * 2.0f * M_PI * pct_frq - M_PI;

            float r = frequencies[i] * sensitivity + 1.5;
            float layer = sin(pct_frq * M_PI);
            float x = radius * layer * r * cosf(theta);
            float y = radius * layer * r * sinf(theta);

            float vert[VERT_LENGTH] = {x, y, pct_frq * 2.0f, frequencies[i], pct_frq};
            vertices.insert(vertices.end(), std::begin(vert), std::end(vert));
        }
    }

	return vertices;
}


void main_draw(std::vector<float> vertices) {
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// Use the shader program
	glUseProgram(program);
	glUniform1i(glGetUniformLocation(program, "color_mode"), color_mode);
	glUniform4fv(glGetUniformLocation(program, "color"), 1, color);
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
		glDrawArrays(GL_TRIANGLES, 0, NUM_POINTS * 3);
	} else {
		if (mode == SPHERE) {
			glDrawArrays(GL_LINE_STRIP_ADJACENCY_EXT, 0, SPHERE_LAYERS * NUM_POINTS);
			glDrawArrays(GL_POINTS, 0, SPHERE_LAYERS * NUM_POINTS);
		} else {
			glDrawArrays(GL_LINE_STRIP, 0, NUM_POINTS);
		}
		cur += 0.1;
		set_camera(glm::vec3(sin(cur), cos(cur), 4.0f));
	}

	// Unbind VAO and shader
	glBindVertexArray(0);
	glUseProgram(0);
}

void arbirtray_logarithm_thing(std::vector<float> &frequencies){
	for (int i = 0; i < FRAMES / 2; i++) {
        frequencies[i] *= log10(((float)i / (FRAMES / 2)) * 5.0 + 1.01);
        frequencies[i] = log10(frequencies[i] * 2.0 + 1.01);
    }
}

void Render() {
    glLineWidth(line_width);
    glPointSize(10.0);
    if (post_processing_enabled) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    }
    glClear(GL_COLOR_BUFFER_BIT);

    double currentTime = glfwGetTime();
    nbFrames++;
    if (currentTime - lastTime >= 1.0) {
        // printf("%f ms/frame\n", 1000.0 / double(nbFrames));
        nbFrames = 0;
        lastTime += 1.0;
    }

    left_frequencies = aanalyzer->getLeftFrequencies();
	right_frequencies = aanalyzer->getRightFrequencies();
    arbirtray_logarithm_thing(left_frequencies);
    arbirtray_logarithm_thing(right_frequencies);
    reactive_frequency = left_frequencies[10];
    bd->process(aanalyzer->getStreamTime(), left_frequencies);

    zoom_speed = reactive_zoom_enabled ? (reactive_frequency * zoom_sensitivity) : zoom_sensitivity * 5.0;

    // if((currentTime - lastBeat) - bd->bpm_offset > bd->current_bpm && bd->quality_avg > 200.0) {
    if (((bd->detection[0] && bd->detection[1]))) {
        beat = true;
    } else {
        beat = false;
    }

    if (color_mode == 2) {
        int freq_index = std::distance(std::begin(left_frequencies), std::max_element(std::begin(left_frequencies), std::end(left_frequencies)));
        int log_freq_index = ((float)freq_index / (float)NUM_POINTS);
        rgb rgbcolor = hsv2rgb(hsv{log_freq_index * 360.0, 1.0 - log_freq_index, 1.0});
        color[0] = rgbcolor.r;
        color[1] = rgbcolor.g;
        color[2] = rgbcolor.b;
    } else if (color_mode == 3) {
        milliseconds current_ms = duration_cast<milliseconds>(
            system_clock::now().time_since_epoch());
        if (reactive_frequency > 0.25 && (current_ms.count() - last_ms.count()) > 100) {
            if (color_cycle == 0) {
                color[0] = 1.0;
                color[1] = 0.0;
                color[2] = 0.0;
            } else {
                color[0] = 0.0;
                color[1] = 1.0;
                color[2] = 0.0;
            }

            color_cycle += 1;
            color_cycle %= 2;
            last_ms = current_ms;
        }
    }

    milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    float time = (double)((double)ms.count() - (double)start.count()) / 1000.0f;

    if (mode == TEXT) {
        glUseProgram(font_program);
        glUniform1f(glGetUniformLocation(font_program, "time"), (float)time);
        glUniform1f(glGetUniformLocation(font_program, "width"), (float)line_width);
        glUniform1f(glGetUniformLocation(font_program, "volume"), (float)left_frequencies[0]);
        glUseProgram(0);
        glEnable(GL_BLEND);
        RenderText(font_program, std::to_string(time), -2.0f, -0.5f, 0.005f, glm::vec3(color[0], color[1], color[2]));
        glDisable(GL_BLEND);
    } else {

		if (stereo_mode == 1)
		{
			glEnable(GL_BLEND);
			main_draw(create_vbo(left_frequencies, 1.0));
			main_draw(create_vbo(right_frequencies, 1.0));
			glDisable(GL_BLEND);
		} else {
			glEnable(GL_BLEND);
			main_draw(create_vbo(left_frequencies, 0.5));
			main_draw(create_vbo(right_frequencies, -0.5));
			glDisable(GL_BLEND);
		}
		
    }

    if (post_processing_enabled) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        cv::ogl::convertFromGLTexture2D(texture, u1);
        cv::add(u1, effect(u2), u1);  // Magic here
        u1.copyTo(u2);

        if ((background_mode == 1 || (background_mode == 2 && beat)) && mode != TEXT) {
            cv::add(u1, background, u1);
        }
        cv::ogl::convertToGLTexture2D(u1, texture);

        glUseProgram(pixel_program);
        glBindVertexArray(vao2);
        texture.bind();
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
        glUseProgram(0);
    }
}

int main() {
    if (!glfwInit())
        exit(EXIT_FAILURE);

    // glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
    window = glfwCreateWindow(screen_width, screen_height, "FFT rip", NULL, NULL);
    // window = glfwCreateWindow(screen_width, screen_height, "My Title", glfwGetPrimaryMonitor(), nullptr);
    printf("Hi.\n");
    if (!window) {
        printf("Failed to create window.\n");
        return 1;
    }

    glfwMakeContextCurrent(window);

    if (true) {
        GLFWmonitor *monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode *mode = glfwGetVideoMode(monitor);
        glfwSetWindowMonitor(window, glfwGetPrimaryMonitor(), 0, 0, screen_width, screen_height, mode->refreshRate);
    }

    // glfwSwapInterval(0); // Disable vsync
    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, resize);
    glewInit();

    printf("GL_VENDOR: %s\n", glGetString(GL_VENDOR));
    printf("GL_VERSION: %s\n", glGetString(GL_VERSION));
    printf("GL_RENDERER: %s\n", glGetString(GL_RENDERER));

    if (cv::ocl::haveOpenCL()) {
        cv::ogl::ocl::initializeContextFromGL();
    }

    if (LoadFontRendering(SHADER_PATH + std::string("fonts/DroidSansMono.ttf"))) {
        return -1;
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
