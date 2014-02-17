
// Why is this .h? Is it a stub header for a c library? If so, these functions
// need extern "C", dont they? It's been a long time since I've mixed C and C++

#ifndef _FIREFLYMV_CAMERA_H_
#define _FIREFLYMV_CAMERA_H_

#include "C/FlyCapture2_C.h"


int initCamera(void);

/* When does the data buffer from the camera go away? I'm pretty
 * sure I need to copy the data buffer */
int grabFrameFromCamera(cv::Mat &frame);


int closeCamera(void);

#endif /* ifndef _FIREFLYMV_CAMERA_H_ */

