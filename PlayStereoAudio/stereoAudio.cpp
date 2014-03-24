#include "stereoAudio.hpp"

using namespace std;

void packet_queue_init(PacketQueue *q)
{
	memset(q, 0, sizeof(PacketQueue));
	q->mutex = SDL_CreateMutex();
	q->cond	 = SDL_CreateCond();
}

void packet_queue_flush(PacketQueue *q)
{
	AVPacketList *pkt, *pkt1;

	SDL_LockMutex(q->mutex);
	for(pkt = q->first_pkt; pkt != NULL; pkt = pkt1) {
		pkt1 = pkt->next;
		av_free_packet(&pkt->pkt);
		av_freep(&pkt);
	}
	q->last_pkt = NULL;
	q->first_pkt = NULL;
	q->nb_packets = 0;
	q->size = 0;
	SDL_UnlockMutex(q->mutex);
}

int packet_queue_put(PacketQueue *q, AVPacket *pkt) {

	AVPacketList *pkt1;
	if(av_dup_packet(pkt) < 0) {
		return -1;
	}

	// allocate memory for new package
	pkt1 = ( AVPacketList *)av_malloc(sizeof(AVPacketList));
	if (!pkt1)
		return -1;


	pkt1->pkt = *pkt;
	pkt1->next = NULL;

	SDL_LockMutex(q->mutex);

	// If last package is NULL, it means that there is only one package left in
	// the queue,
	if (!q->last_pkt)
		q->first_pkt = pkt1;
	else
		q->last_pkt->next = pkt1;

	//
	q->last_pkt = pkt1;

	// Increase the number of packet
	q->nb_packets++;
	q->size += pkt1->pkt.size;
	SDL_CondSignal(q->cond);

	SDL_UnlockMutex(q->mutex);
	return 0;
}

int packet_queue_get(PacketQueue *q, AVPacket *pkt, int block)
{
	AVPacketList *pkt1;
	int ret;

	SDL_LockMutex(q->mutex);

	for(;;)
	{
//		if(quit)
//		{
//			ret = -1;
//			break;
//		}

		// Get the pointer to the first package in the queue
		pkt1 = q->first_pkt;

		// If the pointer to the first package is NOT NULL, it means that there
		// is at least one package in the queue.
		if (pkt1)
		{
			// Assign the first package in the new queue by the second package of
			// the old queue
			q->first_pkt = pkt1->next;

			// If the first package in the new queue is NULL, it means there is
			// no more package left in the new queue.
			if (!q->first_pkt)
			{
				q->last_pkt = NULL;
				//SDL_CondSignal(putQueueCond);
			}

			// reduce the number of packages in the queue
			q->nb_packets--;
			q->size -= pkt1->pkt.size;

			// Return the first packet in the old queue
			*pkt = pkt1->pkt;
			av_free(pkt1);
			ret = 1;
			break;
		}
		else if (!block)
		{
			ret = 0;
			break;
		}
		else
		{
			// If there is no more package in the queue, block this thread and
			// wait for more package
			SDL_CondWait(q->cond, q->mutex);
		}
	}
	SDL_UnlockMutex(q->mutex);
	return ret;
}



int audio_decode_frame(AVCodecContext *aCodecCtx, uint8_t *audio_buf, int buf_size, PacketQueue* audioq) {


	static AVPacket pkt;
	static uint8_t *audio_pkt_data = NULL;
	static int audio_pkt_size = 0;
	static AVFrame frame;

	int len1, data_size = 0;

	for(;;) {
		// Get a new package from the queue
		if(packet_queue_get(audioq, &pkt, 1) < 0)
		{
			return -1;
		}

		// The AVPacket pkt is the input buffer. avpkt->data and avpkt->size
		// should be set for audio decode function (avcodec_decode_audio4)
		audio_pkt_data = pkt.data;
		audio_pkt_size = pkt.size;

		while(audio_pkt_size > 0)
		{
			int got_frame = 0;

			// the number of bytes consumed from the input AVPacket is returned.
			len1 = avcodec_decode_audio4(aCodecCtx, &frame, &got_frame, &pkt);
			if(len1 < 0) {
				/* if error, skip frame */
				audio_pkt_size = 0;
				break;
			}

			audio_pkt_data += len1;			// Increase the data pointer to new position
			audio_pkt_size -= len1;			// Decrease the size of remaining audio package
			if (got_frame)
			{
				// data size of each frame = (frame.nb_samples) X (aCodecCtx->channels)
				data_size = av_samples_get_buffer_size(	NULL,
														aCodecCtx->channels,
														frame.nb_samples,
														aCodecCtx->sample_fmt,
														1);

				// data may be in frame.data[0][4*i + 1]: left and frame.data[0][4*i + 3]: right
				memcpy(audio_buf, frame.data[0], data_size);
				cout << data_size << endl;
#if	STORE_DATA
				for (unsigned int i = audio_buf_index; i < audio_buf_index + len1; i += 4, j++)
				writeData2CSV("dataLeft", packetNumber - 1, audio_buf, audio_size);
#endif
			}
			if(data_size <= 0) {
				/* No data yet, get more frames */
				continue;
			}
			/* We have data, return it and come back for more later */
			return data_size;
		}
		if(pkt.data)
			av_free_packet(&pkt);

//		if(quit) {
//			return -1;
//		}
	}
}

int firstAudioStream(AVFormatContext *pFormatCtx)
{
	int audioStream =-1;
	for(unsigned int i = 0; i < pFormatCtx->nb_streams; i++) {
		if(pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO && audioStream < 0) {
			return i;
		}
	}
	return audioStream;
}

int loadDataCSV(char* fileNameFormat, int packetNumber, uint8_t* audio_buf )
{
	int count = 0;

	stringstream ss;
	ss << setw(3) << setfill('0') << packetNumber;
	string sPackNum = ss.str();

	ifstream dataIn;
	dataIn.open(fileNameFormat + sPackNum + ".csv");
	string tmp = ""; 	// a line in the input file

	while (!dataIn.eof())
	{
		getline(dataIn, tmp, ',');
		audio_buf[count++] = atoi((tmp.c_str()));
	}
	dataIn.close();
	return (count - 1);
}

void writeData2CSV(char* fileNameFormat, int packetNumber, uint8_t* audio_buf, int data_size)
{
	stringstream ss;
	ss << setw(3) << setfill('0') << packetNumber;
	string sPackNum = ss.str();

	ofstream fout(fileNameFormat + sPackNum + ".csv");
	for (int i = 0; i < data_size; i++) {
		//string tmp = NumberToString(audio_buf[i]);
		fout << (int)audio_buf[i] << ',';
	}
	fout.close();
}

