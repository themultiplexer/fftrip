#include "helpers.h"
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
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
#ifdef _WIN32
#include <kiss_fft.h>
#else
#include <kissfft/kiss_fft.h>
#endif
#include <math.h>
#include <opencv2/core/mat.hpp>
#include <rtaudio/RtAudio.h>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/opengl.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <numeric>
#include <algorithm>
#include <string>
#include <iostream>
#include <filesystem>
#include "colors.h"
#include <chrono>
#include <Tracy.hpp>
#include <TracyOpenGL.hpp>
#include <TracyOpenCL.hpp>
#include "font_rendering.h"
#include "3rdparty/beatdetektor/cpp/BeatDetektor.h"

using namespace std::chrono;
namespace fs = std::filesystem;

#define VERT_LENGTH 4 // x,y,z,volume

#define SPHERE_LAYERS 8
#define FRAMES 1024
#define NUM_POINTS 64
#define SHADER_PATH "../"

GLFWwindow* window;
GLuint program, font_program, pixel_program;
GLuint vao, vbo, vao2, vbo2;
GLuint ssaoFramebufferID, ssaoDepthTextureID;
float radius = 1.0f;
RtAudio adc(RtAudio::Api::LINUX_PULSE);
GLfloat circleVertices[NUM_POINTS * VERT_LENGTH * SPHERE_LAYERS];
/* Global */
GLuint fbo, fbo_texture, fbo_texture2, rbo_depth;

cv::ogl::Texture2D texture, texture2;
cv::UMat background;

kiss_fft_cfg cfg;

unsigned int screen_width = 3840;
unsigned int screen_height = 2160;

float rawdata[FRAMES];
float freqs[FRAMES];
float red_freqs[NUM_POINTS];

double lastTime = glfwGetTime();
int nbFrames = 0;

int current_file_index = 0;
int current_font_index = 0;

int zx = 38;
int zy = 21;
int dilation_size = 1;
int erosion_size = 1;
float angle = 0.4f;
float sensitivity = 0.1;
float zoom_sensitivity = 0.1;
float lineWidth = 10.0;


bool post_processing_enabled = true;
bool reactive_zoom_enabled = false;
int color_mode = 0;
int background_mode = 0;
bool dynamic_color = false;
float cur = 0.0;

milliseconds start;

cv::Mat image;
GLfloat color[4] = { 1.0, 0.0, 0.0, 1.0 };

enum VisMode { LINES, CIRCLE, CIRCLE_FLAT, SPHERE, SPHERE_SPIRAL, TEXT };
enum BackgroundMode { OFF, ON, BEAT };
VisMode mode = CIRCLE;

int color_cycle = 0;
milliseconds last_ms;

double lastBeat;
bool beat = false;

cv::UMat u1, u2, u3;

BeatDetektor *bd;


cv::UMat effect1(cv::UMat img) {
	float f = reactive_zoom_enabled ? (red_freqs[0] * zoom_sensitivity) : zoom_sensitivity * 5.0;
	cv::blur(img, img, cv::Size(10, 10));
	cv::addWeighted(img, 0.0, img, 0.98 - red_freqs[1] * 0.001, 0.0, img);
	cv::UMat rot;
	cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	cv::Mat matRotation = cv::getRotationMatrix2D(center, angle * f, 1.0);
	cv::warpAffine(img, rot, matRotation, img.size());
	cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
	cv::dilate(rot, rot, element);
	cv::UMat test2 = rot(cv::Rect(zx * f, zy * f, screen_width - 2 * zx * f, screen_height - 2 * zy * f));
	cv::UMat image2 = cv::UMat(screen_height, screen_width, CV_8UC4);
	cv::resize(test2, image2, cv::Size(screen_width, screen_height));
	return image2;
}

cv::UMat effect2(cv::UMat img) {
	cv::blur(img, img, cv::Size(10, 10));
	cv::addWeighted(img, 0.0, img, 0.98 - red_freqs[0] * 0.001, 0.0, img);
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
	float f = reactive_zoom_enabled ? (red_freqs[0] * zoom_sensitivity) : zoom_sensitivity * 5.0;
	cv::blur(img, img, cv::Size(10, 10));
	cv::addWeighted(img, 0.0, img, 0.98 - red_freqs[1] * 0.001, 0.0, img);
	cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	//cv::Mat matRotation = cv::getRotationMatrix2D( center, angle , 1.0 );
	//cv::warpAffine(img, img, matRotation, img.size());

	cv::UMat test2 = img(cv::Rect(zx * f, zy * f, screen_width - 2 * zx * f, screen_height - 2 * zy * f));
	cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
	cv::dilate(test2, test2, element);
	cv::UMat image2 = cv::UMat(screen_height, screen_width, CV_8UC4);
	cv::resize(test2, image2, cv::Size(screen_width, screen_height));
	return image2;
}

cv::UMat effect4(cv::UMat img) {
	float f = reactive_zoom_enabled ? (red_freqs[0] * zoom_sensitivity) : zoom_sensitivity * 5.0;
	cv::blur(img, img, cv::Size(10, 10));
	cv::addWeighted(img, 0.0, img, 0.98 - red_freqs[0] * 0.001, 0.0, img);
	cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	cv::UMat rot;
	cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, -20);
	cv::warpAffine(img, rot, trans_mat, img.size());
	cv::UMat test2 = rot(cv::Rect(zx* f, zy* f, screen_width - 2 * zx* f, screen_height - 2 * zy * f));
	cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
	cv::dilate(test2, test2, element);
	cv::UMat image2 = cv::UMat(screen_height, screen_width, CV_8UC4);
	cv::resize(test2, image2, cv::Size(screen_width, screen_height));
	return image2;
}

cv::UMat effect5(cv::UMat img) {
	cv::blur(img, img, cv::Size(20, 20));
	cv::addWeighted(img, 0.0, img, 0.98 - red_freqs[0] * 0.001, 0.0, img);
	cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	//cv::Mat matRotation = cv::getRotationMatrix2D( center, angle , 1.0 );
	//cv::warpAffine(img, img, matRotation, img.size());
	cv::UMat test2 = img(cv::Rect(zx, zy, screen_width - 2 * zx, screen_height - 2 * zy));
	cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
	cv::erode(test2, test2, element);
	cv::UMat image2 = cv::UMat(screen_height, screen_width, CV_8UC4);
	cv::resize(test2, image2, cv::Size(screen_width, screen_height));
	return image2;
}

cv::UMat effect6(cv::UMat img) {
	cv::blur(img, img, cv::Size(20, 20));
	cv::addWeighted(img, 0.0, img, 0.98 - red_freqs[0] * 0.001, 0.0, img);
	cv::UMat test2 = img(cv::Rect(zx, zy, screen_width - 2 * zx, screen_height - 2 * zy));
	cv::UMat image2 = cv::UMat(screen_height, screen_width, CV_8UC4);
	cv::resize(test2, image2, cv::Size(screen_width, screen_height));
	return image2;
}

cv::UMat effect7(cv::UMat img) {
	cv::addWeighted(img, 0.0, img, 0.98 - red_freqs[0] * 0.001, 0.0, img);
	cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
	//cv::Mat matRotation = cv::getRotationMatrix2D( center, angle , 1.0 );
	//cv::warpAffine(img, img, matRotation, img.size());
	cv::UMat test2 = img(cv::Rect(zx, zy, screen_width - 2 * zx, screen_height - 2 * zy));
	cv::Mat eelement = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1), cv::Point(dilation_size, dilation_size));
	cv::dilate(test2, test2, eelement);
	cv::Mat delement = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(dilation_size, dilation_size));
	cv::erode(test2, test2, delement);
	cv::UMat image2 = cv::UMat(screen_height, screen_width, CV_8UC4);
	cv::resize(test2, image2, cv::Size(screen_width, screen_height));
	return image2;
}

cv::UMat effect8(cv::UMat img) {
	float f = reactive_zoom_enabled ? (red_freqs[0] * zoom_sensitivity) : zoom_sensitivity * 5.0;
	cv::blur(img, img, cv::Size(20, 20));
	cv::addWeighted(img, 0.0, img, 0.98 - red_freqs[0] * 0.001, 0.0, img);
	auto image_rect = cv::Rect({}, img.size());
	auto roi = cv::Rect(zx, zy, screen_width, screen_height + 2 * zy * f);
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
	cv::addWeighted(img, 0.0, img, 0.98 - red_freqs[0] * 0.001, 0.0, img);
	return img;
}

cv::UMat effect10(cv::UMat img) {
	float f = reactive_zoom_enabled ? (red_freqs[0] * zoom_sensitivity) : zoom_sensitivity * 5.0;
	cv::blur(img, img, cv::Size(20, 20));
	cv::addWeighted(img, 0.0, img, 0.98 - red_freqs[0] * 0.001, 0.0, img);
	cv::UMat test = cv::UMat(screen_height - 2 * zy * f, screen_width, CV_8UC4);
	cv::resize(img, test, cv::Size(screen_width, screen_height - 2 * zy * f));
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
cv::UMat(*effect)(cv::UMat) = effect1;
cv::UMat(*effects[NUM_EFFECTS])(cv::UMat) = { effect1, effect2, effect3, effect4, effect5, effect6, effect7, effect8, effect9, effect10 };

void load_font() {
	std::string path = SHADER_PATH + std::string("fonts/");
	std::string new_file;
	int cur_idx = 0;
	for (const auto& entry : fs::directory_iterator(path)) {
		if (entry.path().filename().string().ends_with("ttf") || entry.path().filename().string().ends_with("otf"))
		{
			if (cur_idx == current_font_index)
			{
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
	for (const auto& entry : fs::directory_iterator(path)) {
		if (entry.path().filename().string().ends_with("png") || entry.path().filename().string().ends_with("jpg"))
		{
			if (cur_idx == current_file_index)
			{
				new_file = entry.path().string();
			}
			
			cur_idx += 1;
		}
	}
	if (!new_file.empty()) {
		cv::Mat black = cv::Mat(cv::Size(screen_width, screen_height), CV_8UC4, cv::Scalar(0, 0, 0, 0));
		cv::Mat image = cv::imread(new_file, cv::IMREAD_COLOR);

		std::vector<cv::Mat>channels;
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

int record(void* outputBuffer, void* inputBuffer, unsigned int nBufferFrames,
	double streamTime, RtAudioStreamStatus status, void* userData) {
	if (status) {
		std::cout << "Stream overflow detected!" << std::endl;
		return 0;
	}

	// printf("%d \n", nBufferFrames);
	kiss_fft_cpx in[FRAMES] = {};
	for (unsigned int i = 0; i < nBufferFrames; i++) {
		in[i].r = ((float*)inputBuffer)[i];
		rawdata[i] = ((float*)inputBuffer)[i];
	}

	kiss_fft_cpx out[FRAMES] = {};
	kiss_fft(cfg, in, out);
	for (int i = 0; i < FRAMES; i++) {
		freqs[i] = sqrt(out[i].r * out[i].r + out[i].i * out[i].i);
		//freqs[i] = (log10(freqs[i] + 0.1) + 1.0) * 2.0;
		freqs[i] *= log10(((float)i/FRAMES) * 10 + 1.01);
	}

	std::vector<float> fftdata;
	for (int i = 0; i < FRAMES; i++) {
		fftdata.push_back(sqrt(out[i].r * out[i].r + out[i].i * out[i].i));
	}

	bd->process(streamTime, fftdata);

	int sample_group = FRAMES / NUM_POINTS;
	int fft_group = (FRAMES / 3) / NUM_POINTS;
	for (int i = 0; i < NUM_POINTS; i++) {
		red_freqs[i] = 0;
		for (int j = 0; j < fft_group; j++) {
			red_freqs[i] += freqs[i * fft_group + j];
		}		
	}

	// Do something with the data in the "inputBuffer" buffer.
	return 0;
}

void getdevices() {
	// Get the list of device IDs
#ifdef _WIN32
	std::vector<unsigned int> ids(adc.getDeviceCount());
	std::iota(ids.begin(), ids.end(), 0);
#else
	std::vector<unsigned int> ids = adc.getDeviceIds();
#endif
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
	GLuint programs[2] = { program, font_program };
	for (int i = 0; i < 2; i++) {
		glUseProgram(programs[i]);

		glm::mat4 trans = glm::mat4(1.0f);
		trans = glm::rotate(trans, glm::radians(0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		GLint uniTrans = glGetUniformLocation(programs[i], "model");
		glUniformMatrix4fv(uniTrans, 1, GL_FALSE, glm::value_ptr(trans));

		glm::mat4 view = glm::lookAt(
			glm::vec3(cam_x, cam_y, cam_z),
			glm::vec3(0.0f, 0.0f, target_z),
			glm::vec3(0.0f, 0.0f, 1.0f)
		);
		GLint uniView = glGetUniformLocation(programs[i], "view");
		glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));

		glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)screen_width / (float)screen_height, 0.1f, 40.0f);
		GLint uniProj = glGetUniformLocation(programs[i], "proj");
		glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));
	}
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action,
	int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
	else if (key == GLFW_KEY_L) {
		mode = VisMode::LINES;
	}
	else if (key == GLFW_KEY_C) {
		mode = VisMode::CIRCLE;
	}
	else if (key == GLFW_KEY_V) {
		mode = VisMode::CIRCLE_FLAT;
	}
	else if (key == GLFW_KEY_S) {
		mode = VisMode::SPHERE;
	}
	else if (key == GLFW_KEY_X) {
		mode = VisMode::SPHERE_SPIRAL;
	}
	else if (key == GLFW_KEY_T && action == GLFW_PRESS) {
		mode = VisMode::TEXT;
	}
	else if (key == GLFW_KEY_I && action == GLFW_PRESS) {
		load_font();
	}
	else if (key == GLFW_KEY_P && action == GLFW_PRESS) {
		post_processing_enabled = !post_processing_enabled;
	}
	else if (key == GLFW_KEY_F) {
		set_camera(0.0f, -2.5f, 2.5f, 0.2f);
	}
	else if (key == GLFW_KEY_R && action == GLFW_PRESS) {
		set_camera(0.0f, -0.1f, 6.0f, 0.0f);
	}
	else if (key == GLFW_KEY_Z && action == GLFW_PRESS) {
		reactive_zoom_enabled = !reactive_zoom_enabled;
	}
	else if (key == GLFW_KEY_D && action == GLFW_PRESS) {
		color_mode += 1;
		color_mode %= 4;
	}
	else if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9 && action == GLFW_PRESS) {
		
		int index = (key - GLFW_KEY_0);
		printf("Usietf %d\n", index);
		if (index < NUM_EFFECTS) {
			effect = effects[index];
			printf("Using effect %d\n", index);
		}
	}
	else if (key == GLFW_KEY_B && action == GLFW_PRESS)  {
		background_mode += 1;
		background_mode %= 3;
	}
	else if (key == GLFW_KEY_KP_DECIMAL && action == GLFW_PRESS) {
		load_background();
	}
	else if (key == GLFW_KEY_F11 && action == GLFW_PRESS) {
		GLFWmonitor* monitor = glfwGetPrimaryMonitor();
		const GLFWvidmode* mode = glfwGetVideoMode(monitor);
		glfwSetWindowMonitor(window, glfwGetPrimaryMonitor(), 0, 0, screen_width, screen_height, mode->refreshRate);
	}
	else if (key == GLFW_KEY_KP_ADD && action == GLFW_PRESS) {
		zoom_sensitivity += 0.05;
		zoom_sensitivity = std::clamp(zoom_sensitivity, 0.0f, 1.0f);
	}
	else if (key == GLFW_KEY_KP_SUBTRACT && action == GLFW_PRESS) {
		zoom_sensitivity -= 0.05;
		zoom_sensitivity = std::clamp(zoom_sensitivity, 0.0f, 1.0f);
	}
	else if (key == GLFW_KEY_KP_6 && action == GLFW_PRESS) {
		lineWidth += 2.0;
		lineWidth = std::clamp(lineWidth, 2.0f, 10.0f);
	}
	else if (key == GLFW_KEY_KP_9 && action == GLFW_PRESS) {
		lineWidth -= 2.0;
		lineWidth = std::clamp(lineWidth, 2.0f, 10.0f);
	}

}

bool Initialize();

static void resize(GLFWwindow* window, int width, int height) {
	if (width == 0 || height == 0) {
		return;
	}

	printf("Resized %d, %d\n", width, height);
	//glViewport(0, 0, width, height);

	GLuint programs[2] = { program, font_program };
	for (int i = 0; i < 2; i++) {
		glUseProgram(programs[i]);

		GLfloat uResolution[2] = { (float)width, (float)height };
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
	getdevices();
	load_background();

	u2 = cv::UMat(cv::Size(screen_width, screen_height), CV_8UC4);
	image = cv::Mat(screen_height, screen_width, CV_8UC4);
	bd = new BeatDetektor();
	
	start = duration_cast< milliseconds >(
		system_clock::now().time_since_epoch()
	);

	RtAudio::StreamParameters parameters;
	parameters.deviceId = adc.getDefaultInputDevice();
	// parameters.deviceId = 132;
	parameters.nChannels = 1;
	parameters.firstChannel = 0;
	unsigned int sampleRate = 48000;
	unsigned int bufferFrames = FRAMES;

	cfg = kiss_fft_alloc(FRAMES, 0, NULL, NULL);
#ifdef _WIN32
	bool failure = false;
	try
	{
		adc.openStream(NULL, &parameters, RTAUDIO_FLOAT32, sampleRate, &bufferFrames, &record);
		adc.startStream();
	}
	catch (const RtAudioError e)
	{
		std::cout << '\n' << e.getMessage() << '\n' << std::endl;
	}
#else
	if (adc.openStream(NULL, &parameters, RTAUDIO_FLOAT32, sampleRate, &bufferFrames, &record)) {
		std::cout << '\n' << adc.getErrorText() << '\n' << std::endl;
		exit(0); // problem with device settings
	}
	// Stream is open ... now start it.
	if (adc.startStream()) {
		std::cout << adc.getErrorText() << std::endl;
	}
#endif

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
	set_camera(0.0f, -0.01f, 6.0f, 0.0f);


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

	glVertexAttribPointer(0, VERT_LENGTH, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (GLvoid*)(4 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

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

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	glUseProgram(0);

	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_CULL_FACE);
	
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	printf("Init finished \n");
	return true;
}


void CreateVBO() {
	std::vector<float> vertices;

	if (mode == LINES) {
		for (int i = 0; i < NUM_POINTS; i++) {
			float vert[5] = { i * (4.0f / NUM_POINTS) - 2.0f, red_freqs[i] * sensitivity, 0.0, red_freqs[i] * sensitivity, (float)i / NUM_POINTS };
			vertices.insert(vertices.end(), std::begin(vert), std::end(vert));
		}
	}
	else if (mode == CIRCLE) {
		for (int i = 0; i < NUM_POINTS; i++) {
			float theta = 2.0f * M_PI * float(i) / float(NUM_POINTS) - M_PI;
			float r = red_freqs[i] * sensitivity + 0.5;

			float x = radius * r * cosf(theta);
			float y = radius * r * sinf(theta);

			float vert[5] = { x, y, 0.0, red_freqs[i] * sensitivity, (float)i / NUM_POINTS };
			vertices.insert(vertices.end(), std::begin(vert), std::end(vert));
		}
	}
	else if (mode == CIRCLE_FLAT) {
		for (int i = 0; i < NUM_POINTS; i++) {
			float theta1 = 2.0f * M_PI * float(i) / float(NUM_POINTS) - M_PI;
			float theta2 = 2.0f * M_PI * float(i + 1) / float(NUM_POINTS) - M_PI;
			float r = red_freqs[i] * sensitivity + 0.5;

			float c[5] = { 0.0, 0.0, 0.0, 0.0, (float)i / NUM_POINTS };
			vertices.insert(vertices.end(), std::begin(c), std::end(c));
			float b[5] = { radius * r * sinf(theta2), radius * r * sinf(theta2), 0.0, 0.0, (float)i / NUM_POINTS };
			vertices.insert(vertices.end(), std::begin(b), std::end(b));
			float a[5] = { radius * r * cosf(theta1), radius * r * sinf(theta1), 0.0, red_freqs[i] * sensitivity, (float)i / NUM_POINTS };
			vertices.insert(vertices.end(), std::begin(a), std::end(a));
		}
	}
	else if (mode == SPHERE) {
		for (int c = 0; c < SPHERE_LAYERS; c++) {
			for (int i = 0; i < NUM_POINTS; i++) {
				float theta = 2.0f * M_PI * float(i) / float(NUM_POINTS) - M_PI;

				float r = red_freqs[i] * sensitivity + 0.5;
				float layer = sin(((float)c / (float)SPHERE_LAYERS) * M_PI);
				float x = radius * layer * r * cosf(theta);
				float y = radius * layer * r * sinf(theta);

				float vert[5] = { x, y, c * 0.2f, red_freqs[i] * sensitivity, (float)i / NUM_POINTS };
				vertices.insert(vertices.end(), std::begin(vert), std::end(vert));
			}
		}
	}
	else if (mode == SPHERE_SPIRAL) {
		for (int i = 0; i < NUM_POINTS; i++) {
			float theta = 5.0f * 2.0f * M_PI * float(i) / float(NUM_POINTS) - M_PI;

			float r = red_freqs[i] * sensitivity + 0.5;
			float percent = ((float)i / (float)(NUM_POINTS));
			float layer = sin(percent * M_PI);
			float x = radius * layer * r * cosf(theta);
			float y = radius * layer * r * sinf(theta);

			float vert[5] = { x, y, percent * 2.0f, red_freqs[i] * sensitivity, (float)i / NUM_POINTS };
			vertices.insert(vertices.end(), std::begin(vert), std::end(vert));
		}
	}

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Render() {
	ZoneScoped;
	TracyGpuZone("Render");
	FrameMark;
	glLineWidth(lineWidth);
	glPointSize(10.0);
	if (post_processing_enabled) {
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	}
	glClear(GL_COLOR_BUFFER_BIT);

	double currentTime = glfwGetTime();
	nbFrames++;
	if (currentTime - lastTime >= 1.0) {
		//printf("%f ms/frame\n", 1000.0 / double(nbFrames));
		nbFrames = 0;
		lastTime += 1.0;
	}

	//if((currentTime - lastBeat) - bd->bpm_offset > bd->current_bpm && bd->quality_avg > 200.0) {
	if (((bd->detection[0] && bd->detection[1]))) {
		beat = true;
	} else {
		beat = false;
	}

	if (color_mode == 2)
	{
		int freq_index = std::distance(std::begin(red_freqs), std::max_element(std::begin(red_freqs), std::end(red_freqs)));
		rgb rgbcolor = hsv2rgb(hsv{ ((float)freq_index / NUM_POINTS) * 360.0, 1.0 - ((float)freq_index / NUM_POINTS) , 1.0  });
		color[0] = rgbcolor.r;
		color[1] = rgbcolor.g;
		color[2] = rgbcolor.b;
	}
	else if (color_mode == 3)
	{
		milliseconds current_ms = duration_cast<milliseconds>(
			system_clock::now().time_since_epoch()
		);
		if (red_freqs[0] > 3.0 && (current_ms.count() - last_ms.count()) > 100) {
			if (color_cycle == 0)
			{
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
	float time =  (double)((double)ms.count() - (double)start.count()) / 1000.0f;

	if (mode == TEXT)
	{
		glUseProgram(font_program);
		glUniform1f(glGetUniformLocation(font_program, "time"), (float)time);
		glUniform1f(glGetUniformLocation(font_program, "volume"), (float)red_freqs[0]);
		glUseProgram(0);
		glEnable(GL_BLEND);
		RenderText(font_program, std::to_string(time), -2.0f, -0.5f, 0.005f, glm::vec3(color[0], color[1], color[2]));
		glDisable(GL_BLEND);
	} else {
		CreateVBO();
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
		}
		else if (mode == CIRCLE) {
			glDrawArrays(GL_LINE_STRIP, 0, NUM_POINTS);
			glDrawArrays(GL_POINTS, 0, NUM_POINTS);
		}
		else if (mode == CIRCLE_FLAT) {
			glDrawArrays(GL_TRIANGLES, 0, NUM_POINTS * 3);
		}
		else {
			if (mode == SPHERE) {
				glDrawArrays(GL_LINE_STRIP_ADJACENCY_EXT, 0, SPHERE_LAYERS * NUM_POINTS);
				glDrawArrays(GL_POINTS, 0, SPHERE_LAYERS * NUM_POINTS);
			}
			else {
				glDrawArrays(GL_LINE_STRIP, 0, NUM_POINTS);
			}
			cur += 0.01;
			set_camera(sin(cur), cos(cur), 4.0f, 0.0f);
		}

		// Unbind VAO and shader
		glBindVertexArray(0);
		glUseProgram(0);
	}

	if (post_processing_enabled) {
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		cv::ogl::convertFromGLTexture2D(texture, u1);

		cv::add(u1, effect(u2), u1);

		//background(cv::Rect(150, 150, 50, 50)).copyTo(background(cv::Rect(0, 0, 50, 50)));
		if ((background_mode == 1 || (background_mode == 2 && beat)) && mode != TEXT) {
			cv::add(u1, background, u3);
		} else {
			u3 = u1;
		}
		u3.copyTo(u2);
		cv::ogl::convertToGLTexture2D(u3, texture);

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

	//glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
	window = glfwCreateWindow(screen_width, screen_height, "My Title", NULL, NULL);
	//window = glfwCreateWindow(screen_width, screen_height, "My Title", glfwGetPrimaryMonitor(), nullptr);
	printf("Hi.\n");
	if (!window) {
		printf("Failed to create window.\n");
		return 1;
	}
	
	glfwMakeContextCurrent(window);
	//glfwSwapInterval(0); // Disable vsync
	glfwSetKeyCallback(window, key_callback);
	glfwSetFramebufferSizeCallback(window, resize);
	glewInit();
	TracyGpuContext;

	printf("GL_VENDOR: %s\n", glGetString(GL_VENDOR));
	printf("GL_VERSION: %s\n", glGetString(GL_VERSION));
	printf("GL_RENDERER: %s\n", glGetString(GL_RENDERER));

	if (cv::ocl::haveOpenCL())
	{
		cv::ogl::ocl::initializeContextFromGL();
	}

	if (LoadFontRendering(SHADER_PATH + std::string("fonts/arial.ttf")))
	{
		return -1;
	}
	

	if (!Initialize()) {
		printf("Scene initialization failed.\n");
		return 1;
	}

	while (!glfwWindowShouldClose(window)) {
		Render();
		glfwSwapBuffers(window);
		TracyGpuCollect;
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
