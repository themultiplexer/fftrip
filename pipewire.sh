pactl load-module module-null-sink media.class=Audio/Duplex sink_name=mytunnel audio.position=FL,FR,RL,RR
pw-link alsa_output.usb-Yamaha_Corporation_Steinberg_CI2_-01.analog-stereo:monitor_FL mytunnel:playback_FL
pw-link alsa_output.usb-Yamaha_Corporation_Steinberg_CI2_-01.analog-stereo:monitor_FR mytunnel:playback_FR
pactl set-default-source mytunnel
