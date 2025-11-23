#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
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
#include <opencv2/core/mat.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <set>
#include <string>
#include <vector>
#include <array>
#include <thread>
#include <fstream>
#include <ranges>
#include <nlohmann/json.hpp>
#include "audioanalyzer.h"
#include "colors.h"
#include "font_rendering.h"


using namespace std::chrono;
namespace fs = std::filesystem;
using json = nlohmann::json;

struct VERT {
    float x;
    float y;
    float z;
    float vol;
    float frq;
};

#define VERT_LENGTH 5  // x,y,z,frequency, volume
#define SPHERE_LAYERS 8
#define FRAMES 1024
#define NUM_POINTS FRAMES / 2
#define SHADER_PATH "../"

enum VisMode {
    CIRCLE,
    LINES,
    OUTLINE,
    TEXT,
    SPHERE,
    SPHERE_SPIRAL,
    CIRCLE_FLAT
};

enum BackgroundMode {
    OFF,
    ON,
    BEAT
};

struct Preset {
    VisMode mode;
    BackgroundMode background_mode;
    int id;
    int effect_mode;
    int color_mode;
    int stereo_mode;
    bool reactive_zoom_enabled;
    bool inverted_background;
    bool inverted_displacement;
    bool inverted_direction;
    bool rotate_camera;
    bool move_to_beat;
    float sensitivity;
    float zoom_sensitivity;
    float line_width;
    float inner_radius;

    std::array<float, 3> camera_center;
    std::array<float, 3> camera_lookat;
};

void to_json(json& j, const Preset& p) {
    j = json{{"mode", p.mode},
    {"background_mode", p.background_mode},
    {"effect_mode", p.effect_mode},
    {"color_mode", p.color_mode},
    {"stereo_mode", p.stereo_mode},
    {"inverted_background", p.inverted_background},
    {"inverted_displacement", p.inverted_displacement},
    {"camera_center", p.camera_center},
    {"camera_lookat", p.camera_lookat},
    {"sensitivity", p.sensitivity},
    {"zoom_sensitivity", p.zoom_sensitivity},
    {"line_width", p.line_width},
    {"inner_radius", p.inner_radius},
    {"inverted_direction", p.inverted_direction},
    {"move_to_beat", p.move_to_beat},
    {"reactive_zoom_enabled", p.reactive_zoom_enabled},
    {"rotate_camera", p.rotate_camera},
    {"id", p.id}
    };
}

void from_json(const json& j, Preset& p) {
    j.at("mode").get_to(p.mode);
    j.at("background_mode").get_to(p.background_mode);
    j.at("effect_mode").get_to(p.effect_mode);
    j.at("color_mode").get_to(p.color_mode);
    j.at("stereo_mode").get_to(p.stereo_mode);
    j.at("inverted_background").get_to(p.inverted_background);
    j.at("inverted_displacement").get_to(p.inverted_displacement);
    j.at("camera_center").get_to(p.camera_center);
    j.at("camera_lookat").get_to(p.camera_lookat);
    j.at("sensitivity").get_to(p.sensitivity);
    j.at("zoom_sensitivity").get_to(p.zoom_sensitivity);
    j.at("line_width").get_to(p.line_width);
    j.at("inner_radius").get_to(p.inner_radius);
    j.at("inverted_direction").get_to(p.inverted_direction);
    j.at("move_to_beat").get_to(p.move_to_beat);
    j.at("reactive_zoom_enabled").get_to(p.reactive_zoom_enabled);
    j.at("rotate_camera").get_to(p.rotate_camera);
    j.at("id").get_to(p.id);
}

GLFWwindow *window;
GLuint program, font_program, pixel_program;
GLuint vao, vbo, vao2, vbo2;
GLuint ssaoFramebufferID, ssaoDepthTextureID;

RtAudio adc(RtAudio::Api::LINUX_PULSE);
GLfloat circleVertices[NUM_POINTS * VERT_LENGTH * SPHERE_LAYERS];
GLuint fbo, fbo_texture, fbo_texture2, rbo_depth;

cv::ogl::Texture2D texture, texture2;
cv::UMat background;

AudioAnalyzer *aanalyzer;
kiss_fft_cfg cfg;

unsigned int screen_width = 3840;
unsigned int screen_height = 2160;
float cam_speed = 0.004f;

bool font_loaded;

glm::vec3 initial_camera_center(0.0f, -0.01f, 1.0f);
glm::vec3 initial_camera_lookat(0.0f, 0.0f, 0.0f);

glm::vec3 camera_center = initial_camera_center;
glm::vec3 camera_lookat = initial_camera_lookat;

std::array<float, 1024> left_frequencies, right_frequencies;

int current_file_index = 0;
int current_font_index = 0;
int zx = 38;
int zy = 21;
int dilation_size = 1;
int erosion_size = 1;
float angle = 0.4f;
float y_offset = 0.0;

float radius_factor = 1.0f;
double rotation = 0.0f;

bool post_processing_enabled = true;
bool dynamic_color = false;
float cur = 0.0;
float zoom_speed;
system_clock::time_point start;

float effect_transition = 0.0;

cv::Mat image;
glm::vec4 color = {1.0, 0.0, 0.0, 1.0};

std::vector<Preset> presets;

int current_preset_index = 0;
Preset current_preset = { CIRCLE, OFF, 0, 0 };
Preset next_preset = current_preset;
int color_cycle = 0;

std::chrono::time_point<std::chrono::steady_clock> last_beat, last_num_press;
auto last_frame = std::chrono::high_resolution_clock::now();

bool wants_to_save = false;
bool did_type_number = false;
std::string typed_number = "";

bool beat = false;
float reactive_frequency;
float thresh = 0.5;

static double gamma1 = 0.0;

std::vector<glm::vec2> svg_points, svg_normals;

cv::UMat raw_mat, mixed_mat, mixed_mat2, temp_mixed_mat;

cv::UMat effect1(cv::UMat img) {
    cv::addWeighted(img, 0.0, img, 0.9 - reactive_frequency * 0.001, gamma1, img);
    cv::blur(img, img, cv::Size(10, 10));
    
    cv::UMat rot;
    cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
    cv::Mat matRotation = cv::getRotationMatrix2D(center, angle * zoom_speed, 1.0);
    cv::warpAffine(img, rot, matRotation, img.size());
    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
    cv::dilate(rot, rot, element);
    cv::UMat test2, image2;

    if (current_preset.inverted_background) {
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
    cv::addWeighted(img, 0.0, img, 0.9 - reactive_frequency * 0.001, gamma1, img);
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
    cv::addWeighted(img, 0.0, img, 0.98 - reactive_frequency * 0.001, gamma1, img);
    cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
    cv::Mat matRotation = cv::getRotationMatrix2D( center, angle , 1.0 );
    cv::warpAffine(img, img, matRotation, img.size());

    cv::UMat test1, test2;
    if (current_preset.inverted_background) {
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
    cv::addWeighted(img, 0.0, img, 0.98 - reactive_frequency * 0.001, gamma1, img);
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
    cv::addWeighted(img, 0.0, img, 0.98 - reactive_frequency * 0.001, gamma1, img);
    cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
    //cv::Mat matRotation = cv::getRotationMatrix2D( center, angle , 1.0 );
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
    cv::addWeighted(img, 0.0, img, 0.98 - reactive_frequency * 0.001, gamma1, img);
    cv::UMat test2 = img(cv::Rect(zx * zoom_speed, zy * zoom_speed, screen_width - 2 * zx * zoom_speed, screen_height - 2 * zy * zoom_speed));
    cv::UMat image2 = cv::UMat(screen_height, screen_width, CV_8UC4);
    cv::resize(test2, image2, cv::Size(screen_width, screen_height));
    return image2;
}

cv::UMat effect7(cv::UMat img) {
    cv::addWeighted(img, 0.0, img, 0.98 - reactive_frequency * 0.001, gamma1, img);
    cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
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
    cv::addWeighted(img, 0.0, img, 0.98 - reactive_frequency * 0.001, gamma1, img);
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
    cv::addWeighted(img, 0.0, img, 0.98 - reactive_frequency * 0.001, gamma1, img);
    return img;
}

cv::UMat effect10(cv::UMat img) {
    cv::blur(img, img, cv::Size(20, 20));
    cv::addWeighted(img, 0.0, img, 0.98 - reactive_frequency * 0.001, gamma1, img);
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

    if (fs::exists(path)) {
        for (const auto &entry : fs::directory_iterator(path)) {
            if (entry.path().filename().string().ends_with("png") || entry.path().filename().string().ends_with("jpg")) {
                if (cur_idx == current_file_index) {
                    new_file = entry.path().string();
                }

                cur_idx += 1;
            }
        }
    }

    if (cur_idx == 0)
    {
        return;
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

static void set_camera(glm::vec3 cam, glm::vec3 target = glm::vec3(0.0, 0.0, 0.0), float rot = 0.0f) {
    GLuint programs[2] = {program, font_program};
    for (int i = 0; i < 2; i++) {
        glUseProgram(programs[i]);

        glm::mat4 trans = glm::mat4(1.0f);
        trans = glm::rotate(trans, rot, glm::vec3(0.0f, 0.0f, 1.0f));
        GLint uniTrans = glGetUniformLocation(programs[i], "model");
        glUniformMatrix4fv(uniTrans, 1, GL_FALSE, glm::value_ptr(trans));

        glm::mat4 view = glm::lookAt(cam, target, glm::vec3(0.0f, 0.0f, 1.0f));
        GLint uniView = glGetUniformLocation(programs[i], "view");
        glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));

        glm::mat4 proj = glm::perspective(glm::radians(60.0f), (float)screen_width / (float)screen_height, 0.1f, 40.0f);
        GLint uniProj = glGetUniformLocation(programs[i], "proj");
        glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));
        glUseProgram(0);
    }
}


glm::vec2 lerp(glm::vec2 p1, glm::vec2 p2, double t) {
    // Ensure t is between 0 and 1
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;

    glm::vec2 interpolated_point;
    interpolated_point.x = p1.x * (1.0 - t) + p2.x * t;
    interpolated_point.y = p1.y * (1.0 - t) + p2.y * t;
    return interpolated_point;
}

glm::vec3 lerp(glm::vec3 p1, glm::vec3 p2, double t) {
    // Ensure t is between 0 and 1
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;

    glm::vec3 interpolated_point;
    interpolated_point.x = p1.x * (1.0 - t) + p2.x * t;
    interpolated_point.y = p1.y * (1.0 - t) + p2.y * t;
    interpolated_point.z = p1.z * (1.0 - t) + p2.z * t;
    return interpolated_point;
}

/* High frequency key check */
void loop_key_check() {
    bool w = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
    bool a = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
    bool s = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
    bool d = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;
    if (w) {
        camera_center += glm::vec3(0.0f, -cam_speed, 0.0f);
        camera_lookat += glm::vec3(0.0f, -cam_speed, 0.0f);
    }
    if (a) {
        camera_center += glm::vec3(cam_speed, 0.0f, 0.0f);
        camera_lookat += glm::vec3(cam_speed, 0.0f, 0.0f);
    }
    if (s) {
        camera_center += glm::vec3(0.0f,cam_speed, 0.0f);
        camera_lookat += glm::vec3(0.0f,cam_speed, 0.0f);
    }
    if (d) {
        camera_center += glm::vec3(-cam_speed, 0.0f, 0.0f);
        camera_lookat += glm::vec3(-cam_speed, 0.0f, 0.0f);
    }
    if (w || a || s || d) {
        set_camera(camera_center, camera_lookat);
    }
}

/* Check keys on key changed event */
static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        } else if (key == GLFW_KEY_H) {
            current_preset.stereo_mode += 1;
            current_preset.stereo_mode %= 6;
        } else if (key == GLFW_KEY_I) {
            current_preset.inverted_displacement = !current_preset.inverted_displacement;
        } else if (key == GLFW_KEY_L) {
            load_font();
        } else if (key == GLFW_KEY_Q) {
            current_preset.effect_mode += 1;
            current_preset.effect_mode %= NUM_EFFECTS;
            effect = effects[current_preset.effect_mode];
        } else if (key == GLFW_KEY_P) {
            post_processing_enabled = !post_processing_enabled;
        }  else if (key == GLFW_KEY_T) {
            current_preset.rotate_camera = !current_preset.rotate_camera;
        } else if (key == GLFW_KEY_F) {
            next_preset = current_preset;
            next_preset.mode = (VisMode)(next_preset.mode + 1);
            next_preset.mode = (VisMode)(next_preset.mode % 7);

            if (next_preset.effect_mode != TEXT) {
                next_preset.id = current_preset.id + 1;
                effect_transition = 0.0;
            }
        } else if (key == GLFW_KEY_R) {
            camera_center = glm::make_vec3(current_preset.camera_center.data());
            camera_lookat = glm::make_vec3(current_preset.camera_lookat.data());
            set_camera(camera_center, camera_lookat);
        } else if (key == GLFW_KEY_Z) {
            current_preset.reactive_zoom_enabled = !current_preset.reactive_zoom_enabled;
        } else if (key == GLFW_KEY_X) {
            current_preset.inverted_direction = !current_preset.inverted_direction;
        } else if (key == GLFW_KEY_Y) {
            current_preset.move_to_beat = !current_preset.move_to_beat;
        } else if (key == GLFW_KEY_K) {
            current_preset.inverted_background = !current_preset.inverted_background;
        } else if (key == GLFW_KEY_E) {
            current_preset.color_mode += 1;
            current_preset.color_mode %= 4;
        } else if (key == GLFW_KEY_C) {
            wants_to_save = !wants_to_save;
        } else if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9) {
            std::string numbers = "0123456789";
            auto now = std::chrono::steady_clock::now();
            int last = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_num_press).count();
            last_num_press = now;

            if (last < 400) {
                typed_number += numbers[key - GLFW_KEY_0];
            } else {
                typed_number = numbers[key - GLFW_KEY_0];
            }

            did_type_number = true;

        } else if (key == GLFW_KEY_B) {
            current_preset.background_mode = (BackgroundMode)(current_preset.background_mode + 1);
            current_preset.background_mode = (BackgroundMode)(current_preset.background_mode % 3);
        } else if (key == GLFW_KEY_KP_DECIMAL) {
            load_background();
        } else if (key == GLFW_KEY_F11) {
            GLFWmonitor *monitor = glfwGetPrimaryMonitor();
            const GLFWvidmode *mode = glfwGetVideoMode(monitor);
            glfwSetWindowMonitor(window, glfwGetPrimaryMonitor(), 0, 0, screen_width, screen_height, mode->refreshRate);
        } else if (key == GLFW_KEY_KP_ADD) {
            current_preset.zoom_sensitivity += 0.05;
            current_preset.zoom_sensitivity = std::clamp(current_preset.zoom_sensitivity, 0.0f, 5.0f);
        } else if (key == GLFW_KEY_KP_SUBTRACT) {
            current_preset.zoom_sensitivity -= 0.05;
            current_preset.zoom_sensitivity = std::clamp(current_preset.zoom_sensitivity, 0.0f, 5.0f);
        } else if (key == GLFW_KEY_KP_6) {
            current_preset.line_width += 1.0;
            current_preset.line_width = std::clamp(current_preset.line_width, 2.0f, 10.0f);
        } else if (key == GLFW_KEY_KP_9) {
            current_preset.line_width -= 1.0;
            current_preset.line_width = std::clamp(current_preset.line_width, 2.0f, 10.0f);
        } else if (key == GLFW_KEY_KP_5) {
            current_preset.inner_radius -= 0.02;
            current_preset.inner_radius = std::clamp(current_preset.inner_radius, 0.0f, 10.0f);
        } else if (key == GLFW_KEY_KP_8) {
            current_preset.inner_radius += 0.02;
            current_preset.inner_radius = std::clamp(current_preset.inner_radius, 0.0f, 10.0f);
        } else if (key == GLFW_KEY_KP_4) {
            current_preset.sensitivity -= 0.02;
            current_preset.sensitivity = std::clamp(current_preset.sensitivity, 0.0f, 10.0f);
        } else if (key == GLFW_KEY_KP_7) {
            current_preset.sensitivity += 0.02;
            current_preset.sensitivity = std::clamp(current_preset.sensitivity, 0.0f, 10.0f);
        }
    }
}

bool Initialize();

static void resize(GLFWwindow *window, int width, int height) {
    if (width == 0 || height == 0) {
        return;
    }

    printf("Resized %d, %d\n", width, height);

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

    mixed_mat = cv::UMat(cv::Size(screen_width, screen_height), CV_8UC4);
    mixed_mat2 = cv::UMat(cv::Size(screen_width, screen_height), CV_8UC4);
    temp_mixed_mat = cv::UMat(cv::Size(screen_width, screen_height), CV_8UC4);
    image = cv::Mat(screen_height, screen_width, CV_8UC4);

    start = system_clock::now();

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

    set_camera(camera_center, camera_lookat);

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
            fprintf(stderr, "glCheckFramebufferStatus: error %p", glewGetErrorString(status));
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

    return true;
}

static int base = 40;
static float f(float x) {
    return (pow(base, x) - 1.0f) / (base - 1.0f);
}
static float g(float x) {
    return log(x * (base - 1.0f) + 1.0f) / log(base);
}

std::vector<VERT> create_vbo(std::array<float, 1024> frequencies, glm::vec2 from, glm::vec2 to, float sign, Preset preset) {
    std::vector<VERT> vertices;

    glm::vec2 vec = from - to;
    float norm = glm::length(vec);

    if (preset.mode == LINES) {
        for (int i = 0; i < NUM_POINTS; i++) {
            float pct_frq = g((float)i / float(NUM_POINTS));
            float pct_frq2 = g((float)i / (float)NUM_POINTS);
            std::vector<float> vert;
            glm::vec2 p = lerp(from, to, pct_frq);
            p *= preset.inner_radius * radius_factor;
            
            p = p + glm::vec2(vec.y / norm, -vec.x / norm) * (frequencies[i] * preset.sensitivity * sign);
            vertices.push_back({p.x, p.y, 0.0, frequencies[i], pct_frq2});
        }
    } else if (preset.mode == CIRCLE) {
        for (int i = 0; i < NUM_POINTS; i++) {
            float pct_frq = g((float)i / float(NUM_POINTS));
            float theta = glm::distance(from, to) * M_PI * pct_frq;

            float r = (sign * frequencies[i] * current_preset.sensitivity) - (preset.inner_radius * radius_factor);

            float x = preset.inner_radius * radius_factor * r * cosf(theta);
            float y = preset.inner_radius * radius_factor * r * sinf(theta);

            vertices.push_back({x, y, 0.0, frequencies[i], pct_frq});
        }
    } else if (preset.mode == CIRCLE_FLAT) {
        const float goldenAngle = M_PI * (3.0f - std::sqrt(5.0f)); // ≈ 137.50776°

        for (size_t i = 0; i < NUM_POINTS; ++i)
        {
            float r = std::sqrt(float(i));        // Fibonacci-style radial growth
            float theta = i * goldenAngle;

            float x = r * std::cos(theta);
            float y = r * std::sin(theta);

            vertices.push_back({x*0.05f, y*0.05f, 0.0, frequencies[i], (float)i / NUM_POINTS});
        }
    } else if (preset.mode == SPHERE) {
        for (int c = 0; c < SPHERE_LAYERS; c++) {
            for (int i = 0; i < NUM_POINTS; i++) {
                float pct_frq = g(float(i) / float(NUM_POINTS));
                float theta = 2.0f * M_PI * pct_frq - M_PI;

                float r = frequencies[i] * current_preset.sensitivity + 1.0;
                float layer = sin(((float)c / (float)SPHERE_LAYERS) * M_PI);
                float x = preset.inner_radius * radius_factor * layer * r * cosf(theta);
                float y = preset.inner_radius * radius_factor * layer * r * sinf(theta);

                vertices.push_back({x, c * 0.1f, y, frequencies[i], (float)i / NUM_POINTS});
            }
        }
    } else if (preset.mode == SPHERE_SPIRAL) {
        for (int i = 0; i < NUM_POINTS; i++) {
            float pct_frq = g(float(i) / float(NUM_POINTS));
            float theta = 5.0 * 2.0f * M_PI * pct_frq - M_PI;

            float r = frequencies[i] * current_preset.sensitivity + 1.0;
            float layer = sin(pct_frq * M_PI);
            float x = preset.inner_radius * radius_factor * layer * r * cosf(theta);
            float y = preset.inner_radius * radius_factor * layer * r * sinf(theta);

            vertices.push_back({x, pct_frq - 0.5f, y, frequencies[i], pct_frq});
        }
    } else if (preset.mode == OUTLINE) {
        int last = -1;
        for (int j = 0; j < svg_points.size(); j++) {
            if (j < svg_points.size()) {
                float i_pct = f((float)j / (float)svg_points.size());
                int i = (i_pct) * NUM_POINTS;

                float displacement = (frequencies[i] * current_preset.sensitivity * 0.05f) * -sign;

                //float window = svg_points.size() / NUM_POINTS;
                float pct_frq = g(i_pct);
                glm::vec2 p = svg_points[j] + (svg_points[j] + svg_normals[j]) * displacement;
                p *= preset.inner_radius * radius_factor;
                vertices.push_back({p.x, -p.y + y_offset, 0, frequencies[i], pct_frq});
            }
        }
    }

	return vertices;
}

std::chrono::duration<double> get_relative_time() {
    using namespace std::chrono;
    auto now = high_resolution_clock::now();
    auto frame_time = duration<double>(now - last_frame);
    last_frame = high_resolution_clock::now();
    return frame_time;
}

void limit_framerate(duration<double> frame_time, double targetFPS) {
    auto targetTime = duration<double>(1.0 / targetFPS);

    if (frame_time < targetTime)
        std::this_thread::sleep_for(targetTime - frame_time);

}

void main_draw(std::vector<VERT> vertices) {
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size() * 5, &vertices[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// Use the shader program
	glUseProgram(program);
	glUniform1i(glGetUniformLocation(program, "color_mode"), current_preset.color_mode);
	glUniform4fv(glGetUniformLocation(program, "color"), 1, glm::value_ptr(color));
	// Bind vertex array object (VAO)
	glBindVertexArray(vao);

	// Draw the circle as a line loop
	if (current_preset.mode == LINES) {
		glDrawArrays(GL_LINE_STRIP, 0, NUM_POINTS);
		glDrawArrays(GL_POINTS, 0, NUM_POINTS);
	} else if (current_preset.mode == CIRCLE || current_preset.mode == CIRCLE_FLAT) {
		glDrawArrays(GL_LINE_STRIP, 0, NUM_POINTS);
		glDrawArrays(GL_POINTS, 0, NUM_POINTS);
	} else if (current_preset.mode == OUTLINE) {
		glDrawArrays(GL_LINE_STRIP, 0, svg_points.size());
		glDrawArrays(GL_POINTS, 0, svg_points.size());
	} if (current_preset.mode == SPHERE) {
        glDrawArrays(GL_LINE_STRIP_ADJACENCY_EXT, 0, SPHERE_LAYERS * NUM_POINTS);
        glDrawArrays(GL_POINTS, 0, SPHERE_LAYERS * NUM_POINTS);
    } else if (current_preset.mode == SPHERE_SPIRAL)  {
        glDrawArrays(GL_LINE_STRIP, 0, NUM_POINTS);
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

void calc_audio(){
    std::vector<float> raw_left_frequencies = aanalyzer->getLeftFrequencies();
	std::vector<float> raw_right_frequencies = aanalyzer->getRightFrequencies();

    float alpha = 0.7f;

    arbirtray_logarithm_thing(raw_left_frequencies);
    arbirtray_logarithm_thing(raw_right_frequencies);

    for (int i = 0; i < NUM_POINTS; i++) {
        left_frequencies[i] = (alpha * raw_left_frequencies[i]) + (1.0 - alpha) * left_frequencies[i];
        right_frequencies[i] = (alpha * raw_right_frequencies[i]) + (1.0 - alpha) * right_frequencies[i];
    }

    float sum = 0.0;
    for (int i = 1; i < 6; ++i) {
        sum += left_frequencies[i];
    }
    reactive_frequency = (float)sum / 6.0f;

    auto now = std::chrono::steady_clock::now();
    float beta = 0.0009f;
    bool lowpeak = (reactive_frequency > thresh);
    int beatms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_beat).count();
    last_beat = now;
    bool debounce = (beatms > 100);
    bool peaked = lowpeak && debounce;

    if (peaked) {
        beat = true;
        thresh = std::max(reactive_frequency - 0.2f, thresh);
    } else {
        thresh = std::max((beta * (reactive_frequency + 0.2f )) + (1.0f - beta) * thresh, 0.25f);
    }
}

void render(bool to_buffer){
    if (to_buffer) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    }
    glLineWidth(current_preset.line_width);
    glPointSize(10.0);
    glClear(GL_COLOR_BUFFER_BIT);
    zoom_speed = current_preset.reactive_zoom_enabled ? (reactive_frequency * current_preset.zoom_sensitivity * 5.0f) : current_preset.zoom_sensitivity * 10.0;

    if (current_preset.color_mode == 1) {
        int freq_index = std::distance(std::begin(left_frequencies), std::max_element(std::begin(left_frequencies), std::end(left_frequencies)));
        float log_freq_index = ((float)freq_index / (float)1024);
        rgb rgbcolor = hsv2rgb(hsv{log_freq_index * 3.5 * 360.0, 1.0, 1.0});
        color[0] = rgbcolor.r;
        color[1] = rgbcolor.g;
        color[2] = rgbcolor.b;
    } else if (current_preset.color_mode == 2) {
            std::vector<glm::vec4> f = {{1.0, 0.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 1.0}};
            color = f[color_cycle];
            color_cycle += 1;
            color_cycle %= 3;
    } else if (current_preset.color_mode == 3) {
        if (beat) {
            std::vector<glm::vec4> f = {{1.0, 0.0, 0.0, 1.0}, {0.0, 1.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 1.0}};
            color = f[color_cycle];
            color_cycle += 1;
            color_cycle %= 3;
        }
    }

    long time = duration_cast<nanoseconds>(system_clock::now() - start).count();
    
    bool lerpit = false;
    if (current_preset.id != next_preset.id) {
        if (effect_transition <= 1.0) {
            lerpit = true;
            effect_transition += 0.01f;
        } else {
            printf("done\n");
            current_preset = next_preset;
            camera_center = glm::make_vec3(current_preset.camera_center.data());
            camera_lookat = glm::make_vec3(current_preset.camera_lookat.data());
            set_camera(camera_center, camera_lookat);
            effect = effects[current_preset.effect_mode];
            last_frame = std::chrono::high_resolution_clock::now();
            lerpit = false;
        }
    }

    if (lerpit) {    
        camera_center = lerp(glm::make_vec3(current_preset.camera_center.data()), glm::make_vec3(next_preset.camera_center.data()), effect_transition);
        camera_lookat = lerp(glm::make_vec3(current_preset.camera_lookat.data()), glm::make_vec3(next_preset.camera_lookat.data()), effect_transition);
        set_camera(camera_center, camera_lookat);
    }

    if (current_preset.mode == TEXT) {
        if (!font_loaded) { 
            LoadFontRendering(SHADER_PATH + std::string("fonts/DroidSansMono.ttf"));
            font_loaded = true;
        }
        if (font_loaded) {        
            glUseProgram(font_program);
            glUniform1f(glGetUniformLocation(font_program, "time"), (float)time);
            glUniform1f(glGetUniformLocation(font_program, "width"), (float)current_preset.line_width);
            glUniform1f(glGetUniformLocation(font_program, "volume"), (float)left_frequencies[0]);
            glUseProgram(0);
            glEnable(GL_BLEND);
            RenderText(font_program, std::to_string(time), -2.0f, -0.5f, 0.001f, glm::vec3(color[0], color[1], color[2]));
            glDisable(GL_BLEND);
        }
    } else {

        std::vector<std::vector<std::vector<glm::vec2>>> main = {
                                                {{glm::vec2(-1.0, 0.0), glm::vec2(1.0, 0.0)}},
                                                {{glm::vec2(-1.0, 0.5), glm::vec2(1.0, 0.5)}, {glm::vec2(-1.0, -0.5), glm::vec2(1.0, -0.5)}, {glm::vec2(1.0, 0.45), glm::vec2(1.0, -0.45)}, {glm::vec2(-1.0, -0.45), glm::vec2(-1.0, 0.45)}}, 
                                                {{glm::vec2(-1.0, 0.1), glm::vec2(0.0, 0.1)}, {glm::vec2(0.0, 0.1), glm::vec2(1.0, 0.1)}}, 
                                                {{glm::vec2(-1.0, 0.2), glm::vec2(1.0, 0.2)}, {glm::vec2(-1.0, -0.2), glm::vec2(1.0, -0.2)}},
                                                {{glm::vec2(-1.0, 0.1), glm::vec2(1.0, 0.1)}, {glm::vec2(-1.0, -0.1), glm::vec2(1.0, -0.1)}},
                                                {{glm::vec2(1.0, 0.2), glm::vec2(-1.0, 0.2)}, {glm::vec2(1.0, -0.2), glm::vec2(-1.0, -0.2)}}
                                            };
        auto m = main[current_preset.stereo_mode];
        auto n = main[next_preset.stereo_mode];

        std::vector<std::vector<VERT>> vbos1;
        for (int i = 0; i < m.size(); i++) {
            float b = current_preset.inverted_displacement ? 1.0f : -1.0f;
            float s = current_preset.inverted_direction ? b : -b;
            vbos1.push_back(create_vbo(i%2==0 ? left_frequencies : right_frequencies, m[i][0], m[i][1], i%2==0? b : s, current_preset));
        }

        std::vector<std::vector<VERT>> vbos2;
        for (int i = 0; i < n.size(); i++) {
            float b = next_preset.inverted_displacement ? 1.0f : -1.0f;
            float s = next_preset.inverted_direction ? b : -b;
            vbos2.push_back(create_vbo(i%2==0 ? left_frequencies : right_frequencies, n[i][0], n[i][1], i%2==0? b : s, next_preset));
        }

        /* Here the transition (vertex interpolation) magic happens */
        if(lerpit) {
            for (int i = 0; i < std::max(vbos1.size(), vbos2.size()); i++) {
                std::vector<VERT> from = vbos1[std::min(i, (int)vbos1.size()-1)];
                std::vector<VERT> to = vbos2[std::min(i, (int)vbos2.size()-1)];
                std::vector<VERT> out = std::vector<VERT>(std::max(from.size(), to.size()));

                for (int i = 0; i < out.size(); i++) {
                    VERT f = from[std::min(i, (int)from.size()-1)];
                    VERT t = to[std::min(i, (int)to.size()-1)];
                    glm::vec3 o = lerp(glm::vec3(f.x, f.y, f.z), glm::vec3(t.x, t.y, t.z), effect_transition);
                    out[i].x = o.x;
                    out[i].y = o.y;
                    out[i].z = o.z;
                    out[i].vol = t.vol;
                    out[i].frq = t.frq;
                }

                main_draw(out);
            }
        } else {
            for (int i = 0; i < vbos1.size(); i++) {
                main_draw(vbos1[i]);
            }
        }
    }

    if (to_buffer) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
}

void main_loop(double current_time) {
    auto now = std::chrono::steady_clock::now();

    if (did_type_number && std::chrono::duration_cast<milliseconds>(now - last_num_press).count() > 400) {
        int index = std::stoi(typed_number);
        if (index < presets.size()) {
            current_preset_index = index;
            if (wants_to_save) {
                current_preset.camera_center = {camera_center.x, camera_center.y, camera_center.z};
                current_preset.camera_lookat = {camera_lookat.x, camera_lookat.y, camera_lookat.z};
                presets[current_preset_index] = current_preset;
                presets[current_preset_index].id = current_preset_index;
                json j = presets;
                std::ofstream outFile("presets.json");
                if (outFile.is_open()) {
                    outFile << j.dump(4);
                    outFile.close();
                } else {
                    std::cerr << "Failed to open file for writing\n";
                }
                wants_to_save = false;
            } else {
                next_preset = presets[current_preset_index];
                mixed_mat = cv::UMat(cv::Size(screen_width, screen_height), CV_8UC4);
        
                std::cerr << "Next preset\n";
                if (next_preset.mode == TEXT || current_preset.mode == TEXT) {
                    current_preset = next_preset;
                }
                effect_transition = 0.0;
                did_type_number = false;
            }
        }
    }

    calc_audio();

    if (false) {
        radius_factor = sin(current_time * 5.0) * 1.0 + 3.0;
        y_offset = sin(current_time * 5.0) * 1.0;
    }

    if (current_preset.rotate_camera) {
        rotation += current_time;
        set_camera(camera_center, camera_lookat, rotation);
    }

    if (current_preset.move_to_beat) {
        if (current_preset.inverted_displacement) {
            radius_factor = 1.0 + (-reactive_frequency * 0.5);
        } else {
            radius_factor = 1.0 + (reactive_frequency * 0.5);
        }
    } else if (false) {
        radius_factor = sin(current_time * 5.0) * 0.1;
    } else {
        radius_factor = 1.0;
    }

    if (post_processing_enabled) {
        render(true);
        
        cv::ogl::convertFromGLTexture2D(texture, raw_mat);
        cv::add(raw_mat, mixed_mat, mixed_mat);
        //cv::addWeighted(raw_mat, 0.5, mixed_mat, 0.5, gamma1, mixed_mat);
        //cv::subtract(mixed_mat, raw_mat, mixed_mat);
        mixed_mat = effect(mixed_mat);
        //cv::subtract(temp_mixed_mat, raw_mat, temp_mixed_mat);
        cv::add(raw_mat, mixed_mat, temp_mixed_mat);
        

        if (false) {            
            cv::add(raw_mat, mixed_mat2, mixed_mat2);
            mixed_mat2 = effect3(mixed_mat2);
            cv::add(temp_mixed_mat, mixed_mat2, temp_mixed_mat);
        }

        if ((current_preset.background_mode == 1 || (current_preset.background_mode == 2 && beat)) && current_preset.mode != TEXT) {
            cv::add(temp_mixed_mat, background, temp_mixed_mat);
        }
        cv::ogl::convertToGLTexture2D(temp_mixed_mat, texture);

        cv::ocl::finish();
        cv::ogl::ocl::finish();

        glUseProgram(pixel_program);
        glBindVertexArray(vao2);
        texture.bind();
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
        glUseProgram(0);

        // Compute binary mask of *non-black* pixels in maskColor
        //cv::Mat mask;
        //cv::inRange(raw_mat, cv::Scalar(1,1,1), cv::Scalar(255,255,255), mask);

        // Zero out img where mask is nonzero
        //mixed_mat.setTo(cv::Scalar(0,0,0), mask);
    } else {
        render(false);
    }
}

int main() {
    // Read JSON from file
    std::ifstream inFile("presets.json");
    if (inFile.is_open()) {
        json j_in;
        inFile >> j_in;
        presets = j_in.get<std::vector<Preset>>();
        std::cout << "Loaded people:\n";
    }

    current_preset = presets[1];
    next_preset = presets[1];
    effect = effects[current_preset.effect_mode];
    

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

    glfwSwapInterval(0); // Disable vsync
    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, resize);
    glewInit();

    printf("GL_VENDOR: %s\n", glGetString(GL_VENDOR));
    printf("GL_VERSION: %s\n", glGetString(GL_VERSION));
    printf("GL_RENDERER: %s\n", glGetString(GL_RENDERER));

    if (cv::ocl::haveOpenCL()) {
        cv::ogl::ocl::initializeContextFromGL();
        cv::ocl::setUseOpenCL(true);
    }

    if (!Initialize()) {
        printf("Scene initialization failed.\n");
        return 1;
    }

    std::ifstream f("../outlines/data.json");
    json data = json::parse(f);
    for (auto d : data["points"]) {
        svg_points.push_back(glm::vec2(d[0], d[1]));
    }
    svg_points.push_back(svg_points.front());
    for (auto d : data["normals"]) {
        svg_normals.push_back(glm::vec2(d[0], d[1]));
    }
    svg_normals.push_back(svg_normals.front());

    /* Not that easy:
    std::thread t([](){
        LoadFontRendering(SHADER_PATH + std::string("fonts/DroidSansMono.ttf"));
        font_loaded = true;
    });
    */

    camera_center = glm::make_vec3(current_preset.camera_center.data());
    camera_lookat = glm::make_vec3(current_preset.camera_lookat.data());
    set_camera(camera_center, camera_lookat);

    while (!glfwWindowShouldClose(window)) {
        duration<float> f = get_relative_time();
        loop_key_check();
        main_loop((double)std::chrono::duration_cast<milliseconds>(f).count() / 1000.0);
        glfwSwapBuffers(window);
        glfwPollEvents();
        limit_framerate(f, 120.0);
    }

    if (adc.isStreamRunning()) {
        adc.stopStream();
    }

    glDeleteRenderbuffers(1, &rbo_depth);
    glDeleteTextures(1, &fbo_texture);
    glDeleteFramebuffers(1, &fbo);

    glfwDestroyWindow(window);
    return 0;
}
