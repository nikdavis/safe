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

#include "stereoAudio.hpp"

using namespace std;

#define PLAY 			( 1 )
#define STORE_DATA 		( 1 )
#define LOAD_CSV 		( 0 )

#ifdef __MINGW32__
#undef main /* Prevents SDL from overriding main() */
#endif



int dangerLevel;
int muteChannel;


SDL_cond 			*loopCond, *putQueueCond;
SDL_mutex			*loopMutex, *putQueueMutex;

AVFormatContext *pFormatCtx;
AVCodecContext  *pCodecCtx = NULL;
AVCodec         *aCodec = NULL;
SDL_AudioSpec   wanted_spec, spec;
AVDictionary 	*audioOptionsDict   = NULL;
PacketQueue 	audioq;

int             audioStream;
int 			packetNumber = 0;
int 			delayTime = 1000;

int initAudio(char* standardFile);
void audio_callback(void *userdata, Uint8 *stream, int len);



/* This should be set to a function that will be called when the audio device is
 * ready for more data. It is passed a pointer to the audio buffer, and the
 * length in bytes of the audio buffer. This function usually runs in a separate
 * thread, and so you should protect data structures that it accesses by calling
 * SDL_LockAudio and SDL_UnlockAudio in your code.
 * http://dranger.com/ffmpeg/functions.html#SDL_OpenAudio
 * */
void audio_callback(void *userdata, Uint8 *stream, int len) {

	AVCodecContext *pCodecCtx = (AVCodecContext *)userdata;
	int len1, audio_size;

	static uint8_t audio_buf[(MAX_AUDIO_FRAME_SIZE * 3) / 2];
	static unsigned int audio_buf_size = 0;
	static unsigned int audio_buf_index = 0;

	if (muteChannel != MUTE_BOTH)
	{
		while(len > 0)	// loop until all data has been sent out
		{
			// If all data has been sent out to the audio stream, get more data.
			if(audio_buf_index >= audio_buf_size)
			{
				timer laodDataTimer("load:			" );
				laodDataTimer.start();

#if 1
				switch (muteChannel)
				{
				case MUTE_RIGHT:
					audio_size = loadDataCSV("./data/dataLeft", packetNumber++, audio_buf);
					break;
				case MUTE_LEFT:
					audio_size = loadDataCSV("./data/dataRight", packetNumber++, audio_buf);
					break;
				case MUTE_NONE:
					audio_size = loadDataCSV("./data/data", packetNumber++, audio_buf);
					break;
				}
#else
				audio_size = audio_decode_frame(pCodecCtx, audio_buf, audio_buf_size, &audioq);
#endif
				laodDataTimer.stop();
				laodDataTimer.printm();

				if(audio_size < 0) {
					/* If error, output silence */
					audio_buf_size = 1024; // arbitrary?
					memset(audio_buf, 0, audio_buf_size);
				} else {
					audio_buf_size = audio_size;
				}
				audio_buf_index = 0;
			}
			len1 = audio_buf_size - audio_buf_index;

			// saturate the maximum length of output data to the audio stream.
			// The maximum output length is 2048 bytes.
			if(len1 > len)
				len1 = len;

			// Copy data from the audio buffer to stream
			// audio_buf[i]     (8 lower bits) audio_buf[i + 1] (8 higher bits): channel 1 (left)
			// audio_buf[i + 2] (8 lower bits) audio_buf[i + 3] (8 higher bits): channel 2 (right)
			copy(audio_buf + audio_buf_index, audio_buf + audio_buf_index + len1, stream);
			//memcpy(stream, (uint8_t *)audio_buf + audio_buf_index, len1);
			len 	-= len1;
			stream 	+= len1;			// Increase the position of pointer of output stream
			audio_buf_index += len1;	// Increase the position of pointer of audio buffer
		}
		if (packetNumber > 14)
		{
			packetNumber = 0;
			printf("danger level: %d\n", dangerLevel);
			switch (dangerLevel){
			case HIGH_DANGER:
				SDL_Delay(1);
				break;
			case MEDIUM_DANGER:
				SDL_Delay(500);
				break;
			case LOW_DANGER:
				SDL_Delay(1000);
				break;
			case NONE_DANGER:
				SDL_Delay(5000);
				break;
			}
		}
	}
}

int initAudio(char* standardFile)
{
	// Register all formats and codecs
	av_register_all();

	pFormatCtx = avformat_alloc_context();
	if(SDL_Init(SDL_INIT_AUDIO | SDL_INIT_TIMER)) {
		fprintf(stderr, "Could not initialize SDL - %s\n", SDL_GetError());
		exit(1);
	}

	// Open video file
	if(avformat_open_input(&pFormatCtx, standardFile, NULL, NULL)!=0)
		return -1; // Couldn't open file

	// Retrieve stream information
	if(avformat_find_stream_info(pFormatCtx, NULL)<0)
		return -1; // Couldn't find stream information

	// Dump information about file onto standard error
	av_dump_format(pFormatCtx, 0, standardFile, 0);

	// Find the first audio stream
	audioStream = firstAudioStream(pFormatCtx);
	if(audioStream < 0)
		return -1;

	pCodecCtx=pFormatCtx->streams[audioStream]->codec;

	// Set audio settings from codec info
	wanted_spec.freq 		= pCodecCtx->sample_rate;
	wanted_spec.format 		= AUDIO_S16SYS;
	wanted_spec.channels 	= pCodecCtx->channels;
	wanted_spec.silence 	= 0;
	wanted_spec.samples 	= SDL_AUDIO_BUFFER_SIZE;
	wanted_spec.callback 	= audio_callback;
	wanted_spec.userdata 	= pCodecCtx;

	aCodec = avcodec_find_decoder(pCodecCtx->codec_id);
   if(!aCodec) {
	   fprintf(stderr, "Unsupported codec!\n");
	   return -1;
   }

	if(SDL_OpenAudio(&wanted_spec, &spec) < 0)
	{
		fprintf(stderr, "SDL_OpenAudio: %s\n", SDL_GetError());
		return -1;
	}

	avcodec_open2(pCodecCtx, aCodec, &audioOptionsDict);
	packet_queue_init(&audioq);
	SDL_PauseAudio(0);

	return 1;
}

int main(int argc, char *argv[])
{
	timer initTimer("Initial:			" );
	initTimer.start();

	if(argc < 4) {
		fprintf(stderr, "Usage:  <standard file> <Danger level> <Mute Channel> \n");
		exit(1);
	}

	if      (!strcmp(argv[2], "H"))
		dangerLevel = HIGH_DANGER;
	else if (!strcmp(argv[2], "L"))
		dangerLevel = LOW_DANGER;
	else if (!strcmp(argv[2], "M"))
		dangerLevel = MEDIUM_DANGER;
	else if (!strcmp(argv[2], "N"))
		dangerLevel = NONE_DANGER;
	else
		return -1;

	if      (!strcmp(argv[3], "L"))
		muteChannel = MUTE_LEFT;
	else if (!strcmp(argv[3], "R"))
		muteChannel = MUTE_RIGHT;
	else if (!strcmp(argv[3], "B"))
		muteChannel = MUTE_BOTH;
	else if (!strcmp(argv[3], "N"))
		muteChannel = MUTE_NONE;
	else
		return -1;

	printf("danger level: %d\n", dangerLevel);

	initAudio(argv[1]);

 	initTimer.stop();
 	initTimer.printm();

 	timer totalTimer("Total:  	");
 	totalTimer.start();

 	while(1);

 	// Close the codec
 	avcodec_close(pCodecCtx);

 	// Close the video file
 	avformat_close_input(&pFormatCtx);

 	return 0;
}
