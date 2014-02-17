


#include "fireflymv_camera.hpp"




// constructor
ffmv_camera::ffmv_camera() {
   fc2Error error;
   fc2EmbeddedImageInfo embeddedInfo;
   initialized = 0;

   error = fc2CreateContext( &context );
   if ( error != FC2_ERROR_OK )
   {
      initialized = -1;
   }        

   error = fc2GetNumOfCameras( context, &numCameras );
   if ( error != FC2_ERROR_OK )
   {
      initialized = -1;
   }        

   if ( numCameras == 0 )
   {
      initialized = -1;
   }        

   // Get the 0th camera
   error = fc2GetCameraFromIndex( context, 0, &guid );
   if ( error != FC2_ERROR_OK )
   {
      initialized = -1;
   }        

   error = fc2Connect( context, &guid );
   if ( error != FC2_ERROR_OK )
   {
      initialized = -1;
   }        

   error = fc2GetEmbeddedImageInfo( context, &embeddedInfo );
   if ( error != FC2_ERROR_OK )
   {
      initialized = -1;
   }  

   if ( embeddedInfo.timestamp.available != 0 )
   {       
      embeddedInfo.timestamp.onOff = enableTimeStamp;
   }    

   fc2SetEmbeddedImageInfo( context, &embeddedInfo );

   error = fc2StartCapture( context );
   if ( error != FC2_ERROR_OK )
   {
      initialized = -1;
   }
}




// destructor
ffmv_camera::~ffmv_camera() {
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




// give frame buffer to user, they own it now!
ffmv_camera::grabFrame(cv::Mat &frame) {

}





fc2SetImageData( 
    fc2Image* pImage,
    const unsigned char* pData,
    unsigned int dataSize);



