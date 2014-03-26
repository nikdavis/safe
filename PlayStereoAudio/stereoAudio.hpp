#ifndef __STEREO_AUDIO_HPP__
#define __STEREO_AUDIO_HPP__

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>

#include <SDL.h>
#include "SDL_audio.h"
#include <SDL_thread.h>
}

#include <pthread.h>
#include "timer.hpp"
#include <math.h>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdio.h>
#include <unistd.h>

using namespace std;

#define SDL_AUDIO_BUFFER_SIZE 	( 1024 )
#define MAX_AUDIO_FRAME_SIZE 	( 192000 )

typedef struct PacketQueue
{
	AVPacketList 	*first_pkt, *last_pkt;
	int 			nb_packets;
	int 			size;
	SDL_mutex 		*mutex;
	SDL_cond 		*cond;
} PacketQueue;

enum DANGER_LEVEL : int
{
	NONE_DANGER = -1,
	HIGH_DANGER = 0,
	MEDIUM_DANGER = 1,
	LOW_DANGER = 2
};

enum MUTE_CHANNEL : int
{
	MUTE_LEFT,
	MUTE_RIGHT,
	MUTE_BOTH,
	MUTE_NONE
};
enum AMPLIFIER : int
{
	AMP_LEFT,
	AMP_RIGHT,
	AMP_BOTH,
	AMP_NONE
};

void packet_queue_init(PacketQueue *q);

static void packet_queue_flush(PacketQueue *q);

int packet_queue_put(PacketQueue *q, AVPacket *pkt);

static int packet_queue_get(PacketQueue *q, AVPacket *pkt, int block);

int audio_decode_frame(AVCodecContext *aCodecCtx, uint8_t *audio_buf, int buf_size,  PacketQueue* audioq);

int firstAudioStream(AVFormatContext *pFormatCtx);

int loadDataCSV(char* fileNameFormat, int packetNumber, uint8_t* audio_buf );

void writeData2CSV(char* fileNameFormat, int packetNumber, uint8_t* audio_buf, int data_size);

#endif /* __STEREO_AUDIO_HPP__ */
