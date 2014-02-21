// FILE: fireflymv_camera.cpp

#include "defs.hpp"
#include "fireflymv_camera.hpp"
#include <iostream>


/* constructor */
FireflyMVCamera::FireflyMVCamera() {
    fc2Error error;
    fc2EmbeddedImageInfo embeddedInfo;
    initialized = -1;
    frames = 0;

    error = fc2CreateContext( &context );
    if ( error != FC2_ERROR_OK )
    {
        std::cerr << "Error in fc2CreateContext: " << error << std::endl;
        return;
    }      

    error = fc2GetNumOfCameras( context, &camCount );
    if ( error != FC2_ERROR_OK )
    {
        std::cerr << "Error in fc2GetNumOfCameras: " << error << std::endl;
        return;
    }      

    if ( camCount == 0 )
    {
        std::cerr << "Error no cameras available" << std::endl;
        return;
    }        

    // Get the 0th camera
    error = fc2GetCameraFromIndex( context, 0, &guid );
    if ( error != FC2_ERROR_OK )
    {
        std::cerr << "Error in fc2GetCameraFromIndex: " << error << std::endl;
        return;
    }       

    error = fc2Connect( context, &guid );
    if ( error != FC2_ERROR_OK )
    {
        std::cerr << "Error in fc2Connect: " << error << std::endl;
        return;
    }       

    error = fc2GetEmbeddedImageInfo( context, &embeddedInfo );
    if ( error != FC2_ERROR_OK )
    {
        std::cerr << "Error in fc2EmbeddedImageInfo: " << error << std::endl;
        return;
    }

    if ( embeddedInfo.timestamp.available != 0 )
    {       
        embeddedInfo.timestamp.onOff = 1;
    }    
    fc2SetEmbeddedImageInfo( context, &embeddedInfo );

    error = fc2StartCapture( context );
    if ( error != FC2_ERROR_OK )
    {
        std::cerr << "Error in fc2StartCapture: " << error << std::endl;
        return;
    }

    error = fc2CreateImage( &image );
    if ( error != FC2_ERROR_OK )
    {
        std::cerr << "Error in fc2CreateImage: " << error << std::endl;
        return;
    }

    initialized = 0;
}




/* destructor */
FireflyMVCamera::~FireflyMVCamera() {
    fc2Error error;
    error = fc2DestroyImage( &image );
    if ( error != FC2_ERROR_OK )
    {
        std::cerr << "Error in fc2DestroyImage: " << error << std::endl;
    }
    error = fc2StopCapture( context );
    if ( error != FC2_ERROR_OK )
    {
        std::cerr << "Error in fc2StopCapture: " << error << std::endl;
    }

    error = fc2DestroyContext( context );
    if ( error != FC2_ERROR_OK )
    {
        std::cerr << "Error in fc2DestroyContext: " << error << std::endl;
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
        std::cerr << "Error in fc2RetrieveBuffer: " << error << std::endl;
        return -1;
    }
    else {
        cv::Mat rawImage(480, 640, CV_8UC1, image.pData, 0);
        retFrame = rawImage.clone();
        frames++;
        return 0;
    }
}



