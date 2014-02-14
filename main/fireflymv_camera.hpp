

#include "C/FlyCapture2_C.h"



class ffmv_camera {
   public:
      ffmv_camera(void);
      ~ffmv_camera(void);
      int numCameras(void);
      int grabFrame(cv::Mat &frame);
   private:      
      int numCameras;         // initialized once
      int initialized;        // if everything goes ok will be 0
      fc2Context context;
      fc2PGRGuid guid;
};
