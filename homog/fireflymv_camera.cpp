


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
        frames++;
        return 0;
    }
}





