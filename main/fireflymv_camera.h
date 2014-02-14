

#include "C/FlyCapture2_C.h"


int initCamera(void);

/* When does the data buffer from the camera go away? I'm pretty
 * sure I need to copy the data buffer */
int grabFrameFromCamera(cv::Mat &frame);


int closeCamera(void);
