


#include "fireflymv_camera.hpp"

/* Should make this read or configure camera */
#define  CAMERA_WIDTH   640
#define  CAMERA_HEIGHT  480


/* constructor */
FireflyMVCamera::FireflyMVCamera() {
    fc2Error error;
    fc2EmbeddedImageInfo embeddedInfo;
    initialized = 1;
    frames = 0;
    fps = 30;
    /* PID initialization */
    statePID = 1;
    errorPID[0] = errorPID[1] = errorPID[2] = 0;
    setPoint = 90.0;
    integError = 0;
    derivError = 0;
    dt = 1.0 / 30.0;
    Ku = 0.25;
    Pu = 0.45;
    Kp = 0.45 * Ku;
    Ki = 1.7 * Kp / Pu;
    Kd = Kp * Pu / 8.0;

    error = fc2CreateContext( &context );
    if ( error != FC2_ERROR_OK )
    {
        printf( "Error in fc2CreateContext: %d\n", error );
        initialized = -1;
    }      

    error = fc2GetNumOfCameras( context, &camCount );
    if ( error != FC2_ERROR_OK )
    {
        printf( "Error in fc2GetNumOfCameras: %d\n", error );
        initialized = -1;
    }      

    if ( camCount == 0 )
    {
        printf( "Error no cameras available\n");
        initialized = -1;
    }        

    // Get the 0th camera
    error = fc2GetCameraFromIndex( context, 0, &guid );
    if ( error != FC2_ERROR_OK )
    {
        printf( "Error in fc2GetCameraFromIndex: %d\n", error );
        initialized = -1;
    }       

    error = fc2Connect( context, &guid );
    if ( error != FC2_ERROR_OK )
    {
        printf( "Error in fc2Connect: %d\n", error );
        initialized = -1;
    }       

    error = fc2GetEmbeddedImageInfo( context, &embeddedInfo );
    if ( error != FC2_ERROR_OK )
    {
        printf( "Error in fc2EmbeddedImageInfo: %d\n", error );
        initialized = -1;
    }

    if ( embeddedInfo.timestamp.available != 0 )
    {       
        embeddedInfo.timestamp.onOff = 1;
    }    

    fc2SetEmbeddedImageInfo( context, &embeddedInfo );

    error = fc2StartCapture( context );
    if ( error != FC2_ERROR_OK )
    {
        printf( "Error in fc2StartCapture: %d\n", error );
        initialized = -1;
    }

    error = fc2CreateImage( &image );
    if ( error != FC2_ERROR_OK )
    {
        printf( "Error in fc2CreateImage: %d\n", error );
        initialized = -1;
    }
#if 0
    if(initialized == 1) {
        /* Setup registers for our manual PID shutter control */
        error = fc2WriteRegister(context, 0x81C, 0x820000A8);     // Set shutter value
        error = fc2WriteRegister(context, 0x800, 0x82000001);     // Auto-brightness off
        error = fc2WriteRegister(context, 0x804, 0x82000007);     // Auto-exposure off, set exposure value
        error = fc2WriteRegister(context, 0x820, 0x82000010);     // Auto-gain off
        if( error != FC2_ERROR_OK ) {
            printf("Error in register initialization! %d\n", error);
            initialized = -1;
        }
    }
#endif
}




/* destructor */
FireflyMVCamera::~FireflyMVCamera() {
    fc2Error error;
    error = fc2DestroyImage( &image );
    if ( error != FC2_ERROR_OK )
    {
        printf( "Error in fc2DestroyImage: %d\n", error );
    }
    error = fc2StopCapture( context );
    if ( error != FC2_ERROR_OK )
    {
        printf( "Error in fc2StopCapture: %d\n", error );
    }

    error = fc2DestroyContext( context );
    if ( error != FC2_ERROR_OK )
    {
        printf( "Error in fc2DestroyContext: %d\n", error );
    }
}



int FireflyMVCamera::ready(void) {
    return (int) (initialized >= 0) ? 1 : 0;
}



int FireflyMVCamera::numCameras(void) {
    return (int) camCount;
}


// give frame buffer to user, we still own it
int FireflyMVCamera::grabFrame(cv::Mat &retFrame) {
    fc2Error error;
    error = fc2RetrieveBuffer(context, &image );

    if ( error != FC2_ERROR_OK )
    {
        printf( "Error in fc2RetrieveBuffer: %d\n", error );
        return -1;
    }
    else {
        cv::Mat rawImage(480, 640, CV_8UC1, image.pData, 0);
        retFrame = rawImage.clone();
  //      if(statePID == 1) processPID(retFrame);
        frames++;
        return 0;
    }
}


int FireflyMVCamera::enablePID(void) {
    statePID = 1;
    return 0;
}


int FireflyMVCamera::disablePID(void) {
    statePID = 0;
    return 0;
}


int FireflyMVCamera::readShutter(unsigned int * value) {
    /* Read the value of the shutter register, 0x81C. The value
     * comprises bits 20:31, or as we read it bits 11:0.            */
    fc2Error error;
    error = fc2ReadRegister(context, 0x81C, value);
    *value &= 0xFFF;
    if( error != FC2_ERROR_OK )     return -1;
    else                            return 0;
}

int FireflyMVCamera::setShutter(unsigned int value) {
    /* RMW the value of the shutter register, 0x81C. The value
     * comprises bits 20:31, or as we read it bits 11:0.            */
    fc2Error error;
    unsigned int temp;
    error = fc2ReadRegister(context, 0x81C, &temp);
    if( error != FC2_ERROR_OK )     return -1;
    temp &= 0xFFFFF000;         // Clear old value
    temp |= (value & 0xFFF);    // Set new value
    // Write back
    error = fc2WriteRegister(context, 0x81C, temp);
    return 0;
}


int FireflyMVCamera::processPID(cv::Mat &frame) {
    unsigned int shutter = 0;
    float shutterFl = 0;
    float p, i, d;
    cv::Mat roadFrame = frame(cv::Range(270, CAMERA_HEIGHT), cv::Range(0, CAMERA_WIDTH));
    cv::Scalar roadMu = cv::mean(roadFrame);
    errorPID[2] = errorPID[1];
    errorPID[1] = errorPID[0];
    errorPID[0] = setPoint - roadMu(0);
    /* Use trap method to compute error */
    integError += 0.5 * (errorPID[0] + errorPID[1]) * dt;
    derivError = (errorPID[0] - errorPID[1]) / (dt * 1);     
    p = Kp * errorPID[0];
    i = Ki * integError;
    d = Kd * derivError;
    //i = 0;
    //d = 0;
    /* Note: to make this work I had to disable auto-exposure, -brightness, and -gain.
     * I also had to set ABSOLUTE mode OFF on each */
    readShutter(&shutter);
    /* Update shutter w/ basic PID */
    shutterFl = (float)shutter + p + i + d;
    if(shutterFl > 500) shutterFl = 500;
    if(shutterFl < 0) shutterFl = 0;
    shutter = (unsigned int) shutterFl;
    setShutter(shutter);
    return 0;
}






