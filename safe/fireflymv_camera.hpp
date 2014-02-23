/* FireflyMVCamera Class
 *
 * This driver wraps the Point Grey C driver for
 * our camera.
 *
 * version 1.0
 *
 */

#ifndef _FIREFLYMV_CAMERA_HPP_
#define _FIREFLYMV_CAMERA_HPP_

#include "C/FlyCapture2_C.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class FireflyMVCamera {
    public:
        FireflyMVCamera(void);
        ~FireflyMVCamera(void);
        int numCameras(void);
        int ready(void);
        int grabFrame(cv::Mat &frame);
    private:      
        unsigned int camCount;      // initialized once
        int initialized;            // if everything goes ok will be 0
        int frames; //*** Never actually used?
        fc2Image image;
        fc2Context context;
        fc2PGRGuid guid;
};

#endif /* ifndef _FIREFLYMV_CAMERA_HPP_ */



