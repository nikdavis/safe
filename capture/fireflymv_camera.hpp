/* FireflyMVCamera Class
 *
 * This driver wraps the Point Grey C driver for
 * our camera.
 *
 * version 1.1
 *
 * IMPORTANT NOTE ABOUT POINT GREY REGISTERS:
 * Point Grey's registers read out of our device very funky. What we receive
 * is exacly reversed bit-for-bit. It's not just a big vs little endian problem,
 * it's bit-for-bit swapped. E.g. our bit 31, 0 are their 0, 31. This is using
 * their Intel / little-endian (64-bit?) driver. It only gets more confusing though,
 * because multiple-bit values you would set have the bit orientation you would expect:
 * i.e. they are read left to right, even though this is backwards in their scheme.
 * E.g: value spanning 30:31 in their scheme (1:0 in ours) would be read using 30 or 1
 * as the MSB, kind of the opposite of their scheme.
 *
 */

#ifndef _FIREFLYMV_CAMERA_HPP_
#define _FIREFLYMV_CAMERA_HPP_


#include "C/FlyCapture2_C.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>


class FireflyMVCamera {
    public:
        FireflyMVCamera(void);
        ~FireflyMVCamera(void);
        int numCameras(void);
        int ready(void);
        int grabFrame(cv::Mat &frame);
        int enablePID(void);                // Region of interest for exposure is fixed in grabFrame
        int disablePID(void);
    private:
        unsigned int camCount;      // initialized once
        int initialized;            // if everything goes ok will be 1
        int frames;
        int statePID;
        int fps;
        fc2Image image;
        fc2Context context;
        fc2PGRGuid guid;
        /* PID variables
         * Kp -> proportional gain
         * Ki -> integral gain
         * Kd -> derivative gain
         * Ku, Pu come from Ziegler–Nichols method of PID tuning */
        float errorPID[3];      /* History of last three errors (setPoint - muROI) */
        float integError, derivError, dt, Ku, Pu, Kp, Ki, Kd, setPoint;
        /* private functions */
        int readShutter(unsigned int * value);
        int setShutter(unsigned int value);
        /* This PID is pretty basic. It uses the shutter to adjust the exposure, trying to get the MEAN of
         * a region of interest (hard set to about the lower 1/3 of the image for now) to agree with a setpoint
         * that is hard set also (as of writing this setPoint = 90 of {0, 255} )    */
        int processPID(cv::Mat &frame);
};

#endif /* ifndef _FIREFLYMV_CAMERA_HPP_ */

