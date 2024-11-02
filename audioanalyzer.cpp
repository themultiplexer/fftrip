#include "audioanalyzer.h"

AudioAnalyzer::AudioAnalyzer(int frequency_bins) {
    fbins = frequency_bins;
    red_freqs = std::vector<float>(fbins);
    cfg = kiss_fft_alloc(FRAMES, 0, NULL, NULL);
    adc = new RtAudio(RtAudio::Api::LINUX_PULSE);
}

void applyHannWindow(float* data, int len) {
    for (int i = 0; i < len; ++i) {
        float windowValue = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (len - 1)));
        data[i] *= windowValue;
    }
}

void AudioAnalyzer::startRecording() {
    RtAudio::StreamParameters parameters;
    parameters.deviceId = adc->getDefaultInputDevice();
    parameters.nChannels = 1;
    parameters.firstChannel = 0;
    unsigned int sampleRate = 48000;
    unsigned int bufferFrames = FRAMES;

#ifdef _WIN32
    bool failure = false;
    try
    {
        adc->openStream(NULL, &parameters, RTAUDIO_FLOAT32, sampleRate, &bufferFrames, &AudioAnalyzer::record);
        adc->startStream();
    }
    catch (const RtAudioError e)
    {
        std::cout << '\n' << e.getMessage() << '\n' << std::endl;
    }
#else
    if (adc->openStream(NULL, &parameters, RTAUDIO_FLOAT32, sampleRate, &bufferFrames, &AudioAnalyzer::static_record, this)) {
        std::cout << '\n' << adc->getErrorText() << '\n' << std::endl;
        exit(0); // problem with device settings
    }
    // Stream is open ... now start it.
    if (adc->startStream()) {
        std::cout << adc->getErrorText() << std::endl;
    }
#endif
}

int AudioAnalyzer::static_record(void* outputBuffer, void* inputBuffer, unsigned int nBufferFrames, double streamTime, unsigned int status, void* userData) {
    return static_cast<AudioAnalyzer*>(userData)->record(outputBuffer, inputBuffer, nBufferFrames, streamTime, status);
}


int AudioAnalyzer::record(void* outputBuffer, void* inputBuffer, unsigned int nBufferFrames, double streamTime, unsigned int status) {
    if (status) {
        std::cout << "Stream overflow detected!" << std::endl;
        return 0;
    }

    streamTimePoint = streamTime;
    int upperLimit = (FRAMES/2) - 50;

    applyHannWindow(((float *)inputBuffer), FRAMES);

    // printf("%d \n", nBufferFrames);
    kiss_fft_cpx in[FRAMES] = {};
    for (unsigned int i = 0; i < nBufferFrames; i++) {
        in[i].r = ((float*)inputBuffer)[i];
        rawdata[i] = ((float*)inputBuffer)[i];
    }

    kiss_fft_cpx out[FRAMES] = {};
    kiss_fft(cfg, in, out);
    for (int i = 0; i < upperLimit; i++) {
        freqs[i] = sqrt(out[i].r * out[i].r + out[i].i * out[i].i);
    }

    int fft_group = upperLimit / fbins;
    for (int i = 0; i < fbins; i++) {
        red_freqs[i] = 0;
        for (int j = 0; j < fft_group; j++) {
            red_freqs[i] += freqs[i * fft_group + j];
        }
        red_freqs[i] /= fft_group;
        red_freqs[i] *= log10(((float)i/fbins) * 10 + 1.01) * 4.0;
    }

    return 0;
}

double AudioAnalyzer::getStreamTime(){
    return streamTimePoint;
}

std::vector<float> AudioAnalyzer::getFrequencies() {
    return red_freqs;
}

std::vector<float> AudioAnalyzer::getFullFrequencies() {
    std::vector<float> v;
    v.assign(freqs, freqs + FRAMES/2);
    return v;
}

void AudioAnalyzer::getdevices() {
    // Get the list of device IDs
#ifdef _WIN32
    std::vector<unsigned int> ids(adc->getDeviceCount());
    std::iota(ids.begin(), ids.end(), 0);
#else
    std::vector<unsigned int> ids = adc->getDeviceIds();
#endif
    if (ids.size() == 0) {
        std::cout << "No devices found." << std::endl;
        exit(0);
    }

    // Scan through devices for various capabilities
    RtAudio::DeviceInfo info;
    for (unsigned int n = 0; n < ids.size(); n++) {

        info = adc->getDeviceInfo(ids[n]);

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
