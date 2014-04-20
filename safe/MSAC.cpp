// FILE: MSAC.cpp

#include "MSAC.hpp"
#include "errorNIETO.hpp"
#include "lmmin.h"
#include "defs.hpp"
#include <cmath>

using namespace std;
using namespace cv;

MSAC::MSAC(void)
{
    // Auxiliar variables
    __a = Mat(3,1,CV_32F);
    __an = Mat(3,1,CV_32F);
    __b = Mat(3,1,CV_32F);
    __bn = Mat(3,1,CV_32F);
    __li = Mat(3,1,CV_32F);
    __c = Mat(3,1,CV_32F);

    __vp = cv::Mat(3,1,CV_32F);
    __vpAux = cv::Mat(3,1,CV_32F);
}

MSAC::~MSAC(void) {}

void MSAC::init(int mode, cv::Size imSize)
{
    // Arguments
    __width = imSize.width;
    __height = imSize.height;
    __mode = mode;

    // MSAC parameters
    __epsilon = (float)1e-6;
    __P_inlier = (float)0.95;
    __T_noise_squared = (float)0.01623*2;
    __min_iters = 5;
    __max_iters = INT_MAX;
    __update_T_iter = false;

    // Parameters
    __minimal_sample_set_dimension = 2;

    // Minimal Sample Set
    for(int i=0; i<__minimal_sample_set_dimension; ++i) __MSS.push_back(0);

    // (Default) Calibration
    __K = Mat(3,3,CV_32F);
    __K.setTo(0);
    __K.at<float>(0,0) = (float)__width;
    __K.at<float>(0,2) = (float)__width/2;
    __K.at<float>(1,1) = (float)__height;
    __K.at<float>(1,2) = (float)__height/2;
    __K.at<float>(2,2) = (float)1;
}

// COMPUTE VANISHING POINTS
void MSAC::fillDataContainers(std::vector<std::vector<cv::Point> > &lineSegments)
{
    int numLines = lineSegments.size();
    if(VERBOSE_MSAC) printf( "Line segments: %d\n", numLines );

    // Transform all line segments
    // __Li = [l_00 l_01 l_02; l_10 l_11 l_12; l_20 l_21 l_22; ...]; where li=[l_i0;l_i1;l_i2]^T is li=an x bn; 
    __Li = Mat(numLines, 3, CV_32F);
    __Mi = Mat(numLines, 3, CV_32F);
    __Lengths = Mat(numLines, numLines, CV_32F);
    __Lengths.setTo(0);

    // Fill data containers (__Li, __Mi, __Lenghts)
    double sum_lengths = 0;
    for (int i=0; i<numLines; i++)
    {
        // Extract the end-points
        Point p1 = lineSegments[i][0];
        Point p2 = lineSegments[i][1];
        __a.at<float>(0,0) = (float)p1.x;
        __a.at<float>(1,0) = (float)p1.y;
        __a.at<float>(2,0) = 1;
        __b.at<float>(0,0) = (float)p2.x;
        __b.at<float>(1,0) = (float)p2.y;
        __b.at<float>(2,0) = 1;

        if(__mode == MODE_NIETO)
            __c = 0.5*(__a+__b);

        double length = sqrt((__b.at<float>(0,0)-__a.at<float>(0,0))*(__b.at<float>(0,0)-__a.at<float>(0,0))
            + (__b.at<float>(1,0)-__a.at<float>(1,0))*(__b.at<float>(1,0)-__a.at<float>(1,0)));
        sum_lengths += length;
        __Lengths.at<float>(i,i) = (float)length;

        if(__mode == MODE_LS)
        {
            // Normalize into the sphere
            __an = __K.inv()*__a;
            __bn = __K.inv()*__b;
        }
        else // __mode == MODE_NIETO requires not to calibrate into the sphere
        {
            __an = __a;
            __bn = __b;
        }

        // Compute the general form of the line
        __li = __an.cross(__bn);
        cv::normalize(__li,__li);

        // Insert line into appended array
        __Li.at<float>(i,0) = __li.at<float>(0,0);
        __Li.at<float>(i,1) = __li.at<float>(1,0);
        __Li.at<float>(i,2) = __li.at<float>(2,0);

        if( __mode == MODE_NIETO )
        {
            // Store mid-Point too
            __Mi.at<float>(i,0) = __c.at<float>(0,0);
            __Mi.at<float>(i,1) = __c.at<float>(1,0);
            __Mi.at<float>(i,2) = __c.at<float>(2,0);
        }
    }
    __Lengths = __Lengths*((double)1/sum_lengths);
}

bool MSAC::VPEstimation(std::vector<std::vector<cv::Point> > &lineSegments, std::vector<std::vector<std::vector<cv::Point> > > &lineSegmentsClusters, std::vector<int> &numInliers, cv::Mat &vps)
{
    bool vp_found = false;
    // Make a copy of lineSegments because it is modified in the code (it will be restored at the end of this function)
    std::vector<std::vector<cv::Point> > lineSegmentsCopy = lineSegments;

    // Fill data structures
    fillDataContainers( lineSegments );
    int numLines = lineSegments.size();

    if(VERBOSE_MSAC) printf( "----- VP -----\n" );

    // Break if the number of elements is lower than minimal sample set
    if(numLines < 3 || numLines < __minimal_sample_set_dimension)
    {
        if(VERBOSE_MSAC)
            printf("Not enough line segments to compute vanishing point\n");
        return false;
    }

    // Vector containing indexes for current vp
    std::vector<int> ind_CS;

    __N_I_best = __minimal_sample_set_dimension;
    __J_best = FLT_MAX;

    int iter = 0;
    int T_iter = INT_MAX;
    int no_updates = 0;
    int max_no_updates = INT_MAX;

    // Define containers of CS (Consensus set): __CS_best to store the best one, and __CS_idx to evaluate a new candidate
    __CS_best = vector<int>(numLines, 0);
    __CS_idx = vector<int>(numLines, 0);

    // Allocate Error matrix
    vector<float> E = vector<float>(numLines, 0);

    // MSAC
    if(VERBOSE_MSAC)
    {
        if(__mode == MODE_LS)
            printf("Method: Calibrated Least Squares\n");
        if(__mode == MODE_NIETO)
            printf("Method: Nieto\n");

        printf("Start MSAC\n");
    }

    // RANSAC loop
    while ( (iter <= __min_iters) || ((iter<=T_iter) && (iter <=__max_iters) && (no_updates <= max_no_updates)) )
    {
        iter++;

        if(iter >= __max_iters)
            break;

        // Hypothesize ------------------------
        // Select MSS
        if(__Li.rows < (int)__MSS.size())
            break;
        GetMinimalSampleSet(__Li, __Lengths, __Mi, __MSS, __vpAux);     // output __vpAux is calibrated

        // Test --------------------------------
        // Find the consensus set and cost
        int N_I = 0;
        float J = GetConsensusSet(__Li, __Lengths, __Mi, __vpAux, E, &N_I);  // the CS is indexed in CS_idx

        // Update ------------------------------
        // If the new cost is better than the best one, update
        if ( (N_I >= __minimal_sample_set_dimension && (J<__J_best)) || ((J == __J_best) && (N_I > __N_I_best)))
        {
            __notify = true;

            __J_best = J;
            __CS_best = __CS_idx; 

            __vp = __vpAux;     // Store into __vp (current best hypothesis): __vp is therefore calibrated
            if (N_I > __N_I_best)
                __update_T_iter = true;

            __N_I_best = N_I;

            if (__update_T_iter)
            {
                // Update number of iterations
                double q = 0;
                if (__minimal_sample_set_dimension > __N_I_best)
                {
                    // Error!
                    perror("The number of inliers must be higher than minimal sample set");
                }
                if(numLines == __N_I_best)
                {
                    q = 1;
                }
                else
                {
                    q = 1;
                    for (int j=0; j<__minimal_sample_set_dimension; j++)
                        q *= (double)(__N_I_best - j)/(double)(numLines - j);
                }
                // Estimate the number of iterations for RANSAC
                if ((1-q) > 1e-12)
                    T_iter = (int)ceil( log((double)__epsilon) / log((double)(1-q)));
                else
                    T_iter = 0;
            }
        }
        else
            __notify = false;

        // Verbose
        if (VERBOSE_MSAC && __notify)
        {
            int aux = max(T_iter, __min_iters);
            printf("Iteration = %5d/%9d. ", iter, aux);
            printf("Inliers = %6d/%6d (cost is J = %8.4f)\n", __N_I_best, numLines, __J_best);

            if(VERBOSE_MSAC)
                printf("MSS Cal.VP = (%.3f,%.3f,%.3f)\n", __vp.at<float>(0,0), __vp.at<float>(1,0), __vp.at<float>(2,0));
        }

        // Check CS length (for the case all line segments are in the CS)
        if (__N_I_best == numLines)
        {
            if(VERBOSE_MSAC)
                printf("All line segments are inliers. End MSAC at iteration %d.\n", iter);
            break;
        }
    }

    // Reestimate ------------------------------
    if(VERBOSE_MSAC)
    {
        printf("Number of iterations: %d\n", iter);
        printf("Final number of inliers = %d/%d\n", __N_I_best, numLines);
    }

    // Fill ind_CS with __CS_best
    std::vector<std::vector<cv::Point> > lineSegmentsCurrent;
    for(int i=0; i<numLines; i++)
    {
        if(__CS_best[i] == 0)
        {
            int a = i;
            ind_CS.push_back(a);
            lineSegmentsCurrent.push_back(lineSegments[i]);
        }
    }

    // if J==0 maybe its because all line segments are perfectly parallel and the vanishing point is at the infinity
    if(__J_best > 0 && ind_CS.size() > (unsigned int)__minimal_sample_set_dimension) 
    {
        if(VERBOSE_MSAC)
        {
            printf("Reestimating the solution... ");
            fflush(stdout);
        }

        if(__mode == MODE_LS)
            estimateLS(__Li, __Lengths, ind_CS, __N_I_best, __vp);
        else if(__mode == MODE_NIETO)
            estimateNIETO(__Li, __Lengths, __Mi, ind_CS, __N_I_best, __vp); // Output __vp is calibrated
        else
            perror("ERROR: mode not supported, please use {LS, LIEB, NIETO}\n");

        if(VERBOSE_MSAC)
            printf("done!\n");

        // Uncalibrate
        if(VERBOSE_MSAC)
            printf("Cal.VP = (%.3f,%.3f,%.3f)\n", __vp.at<float>(0,0), __vp.at<float>(1,0), __vp.at<float>(2,0));
        __vp = __K*__vp;
        if(__vp.at<float>(2,0) != 0)
        {
            __vp.at<float>(0,0) /= __vp.at<float>(2,0);
            __vp.at<float>(1,0) /= __vp.at<float>(2,0);
            __vp.at<float>(2,0) = 1;
        }
        else
        {
            // Since this is infinite, it is better to leave it calibrated
            __vp = __K.inv()*__vp;
        }
        if(VERBOSE_MSAC)
            printf("VP = (%.3f,%.3f,%.3f)\n", __vp.at<float>(0,0), __vp.at<float>(1,0), __vp.at<float>(2,0));

        // Copy to output vector
        vps = __vp;
        vp_found = true;
    }
    else if(fabs(__J_best - 1) < 0.000001)
    {
        if(VERBOSE_MSAC)
        {
            printf("The cost of the best MSS is 0! No need to reestimate\n");
            printf("Cal. VP = (%.3f,%.3f,%.3f)\n", __vp.at<float>(0,0), __vp.at<float>(1,0), __vp.at<float>(2,0));
        }

        // Uncalibrate
        __vp = __K*__vp;
        if(__vp.at<float>(2,0) != 0)
        {
            __vp.at<float>(0,0) /= __vp.at<float>(2,0);
            __vp.at<float>(1,0) /= __vp.at<float>(2,0);
            __vp.at<float>(2,0) = 1;

            if(VERBOSE_MSAC)
                printf("VP = (%.3f,%.3f,%.3f)\n", __vp.at<float>(0,0), __vp.at<float>(1,0), __vp.at<float>(2,0));
        }
        else
        {
            // Calibrate
            __vp = __K.inv()*__vp;
        }
        // Copy to output vector
        vps = __vp;
        vp_found = true;
    }

    // Fill lineSegmentsClusters containing the indexes of inliers for current vps
    if(__N_I_best > 2)
    {
        while(ind_CS.size())
        {
            lineSegments.erase(lineSegments.begin() + ind_CS[ind_CS.size() - 1]);
            ind_CS.pop_back();
        }
    }

    // Fill current CS
    lineSegmentsClusters.push_back(lineSegmentsCurrent);

    // Fill numInliers
    numInliers.push_back(__N_I_best);

    // Restore lineSegments
    lineSegments = lineSegmentsCopy;

    return vp_found;
}

// RANSAC
void MSAC::GetMinimalSampleSet(cv::Mat &Li, cv::Mat &Lengths, cv::Mat &Mi, std::vector<int> &MSS, cv::Mat &vp)
{
    int N = Li.rows;

    // Generate a pair of samples
    while (N <= (MSS[0] = rand() / (RAND_MAX/(N-1))));
    while (N <= (MSS[1] = rand() / (RAND_MAX/(N-1))));

    // Estimate the vanishing point and the residual error
    if(__mode == MODE_LS)
        estimateLS(Li,Lengths, MSS, 2, vp);
    else if(__mode == MODE_NIETO)
        estimateNIETO(Li, Mi, Lengths, MSS, 2, vp);
    else
        perror("ERROR: mode not supported. Please use {LS, LIEB, NIETO}\n");
}

float MSAC::GetConsensusSet(cv::Mat &Li, cv::Mat &Lengths, cv::Mat &Mi, cv::Mat &vp, std::vector<float> &E, int *CS_counter)
{
    // Compute the error of each line segment of LSS with respect to v_est
    // If it is less than the threshold, add to the CS
    for(unsigned int i=0; i<__CS_idx.size(); i++)
        __CS_idx[i] = -1;

    float J = 0;

    if(__mode == MODE_LS)
        J = errorLS(Li, vp, E, CS_counter);
    else if(__mode == MODE_NIETO)
        J = errorNIETO(Li, Lengths, Mi, vp, E, CS_counter);
    else
        perror("ERROR: mode not supported, please use {LS, LIEB, NIETO}\n");

    return J;
}

// Estimation functions
void MSAC::estimateLS(cv::Mat &Li, cv::Mat &Lengths, std::vector<int> &set, int set_length, cv::Mat &vp)
{
    if (set_length == __minimal_sample_set_dimension)
    {
        // Just the cross product
        // DATA IS CALIBRATED in MODE_LS
        cv::Mat ls0 = Mat(3,1,CV_32F);
        cv::Mat ls1 = Mat(3,1,CV_32F);

        ls0.at<float>(0,0) = Li.at<float>(set[0],0);
        ls0.at<float>(1,0) = Li.at<float>(set[0],1);
        ls0.at<float>(2,0) = Li.at<float>(set[0],2);

        ls1.at<float>(0,0) = Li.at<float>(set[1],0);
        ls1.at<float>(1,0) = Li.at<float>(set[1],1);
        ls1.at<float>(2,0) = Li.at<float>(set[1],2);

        vp = ls0.cross(ls1);

        cv::normalize(vp, vp);

        return;
    }
    else if (set_length<__minimal_sample_set_dimension)
    {
        perror("Error: at least 2 line-segments are required\n");
        return;
    }

    // Extract the line segments corresponding to the indexes contained in the set
    cv::Mat li_set = Mat(3, set_length, CV_32F);
    cv::Mat Lengths_set = Mat(set_length, set_length, CV_32F);
    Lengths_set.setTo(0);

    // Fill line segments info
    for (int i=0; i<set_length; i++)
    {
        li_set.at<float>(0,i) = Li.at<float>(set[i], 0);
        li_set.at<float>(1,i) = Li.at<float>(set[i], 1);
        li_set.at<float>(2,i) = Li.at<float>(set[i], 2);

        Lengths_set.at<float>(i,i) = Lengths.at<float>(set[i],set[i]);
    }

    // Least squares solution
    // Generate the matrix ATA (a partir de LSS_set=A)
    cv::Mat L = li_set.t();
    cv::Mat Tau = Lengths_set;
    cv::Mat ATA = Mat(3,3,CV_32F);
    ATA = L.t()*Tau.t()*Tau*L;

    // Obtain eigendecomposition
    cv::Mat w, v, vt;
    cv::SVD::compute(ATA, w, v, vt);

    // Check eigenvecs after SVDecomp
    if(v.rows < 3)
        return;

    // print v, w, vt...
    //std::cout << "w=" << w << endl;
    //std::cout << "v=" << v << endl;
    //std::cout << "vt" << vt << endl;

    // Assign the result (the last column of v, corresponding to the eigenvector with lowest eigenvalue)
    vp.at<float>(0,0) = v.at<float>(0,2);
    vp.at<float>(1,0) = v.at<float>(1,2);
    vp.at<float>(2,0) = v.at<float>(2,2);

    cv::normalize(vp,vp);

    return;
}

void MSAC::estimateNIETO(cv::Mat &Li, cv::Mat &Lengths, cv::Mat &Mi, std::vector<int> &set, int set_length, cv::Mat &vp)
{
    if (set_length == __minimal_sample_set_dimension)
    {
        // Just the cross product
        // DATA IS NOT CALIBRATED for MODE_NIETO
        cv::Mat ls0 = Mat(3,1,CV_32F);
        cv::Mat ls1 = Mat(3,1,CV_32F);

        ls0.at<float>(0,0) = Li.at<float>(set[0],0);
        ls0.at<float>(1,0) = Li.at<float>(set[0],1);
        ls0.at<float>(2,0) = Li.at<float>(set[0],2);

        ls1.at<float>(0,0) = Li.at<float>(set[1],0);
        ls1.at<float>(1,0) = Li.at<float>(set[1],1);
        ls1.at<float>(2,0) = Li.at<float>(set[1],2);

        vp = ls0.cross(ls1);

        // Calibrate (and normalize) vp
        vp = __K.inv()*vp;
        cv::normalize(vp, vp);
        return;
    }
    else if (set_length<__minimal_sample_set_dimension)
    {
        perror("Error: at least 2 line-segments are required\n");
        return;
    }

    // Extract the line segments corresponding to the indexes contained in the set
    cv::Mat li_set = Mat(3, set_length, CV_32F);
    cv::Mat Lengths_set = Mat(set_length, set_length, CV_32F);
    cv::Mat mi_set = Mat(3, set_length, CV_32F);
    Lengths_set.setTo(0);

    // Fill line segments info
    for (int i=0; i<set_length; i++)
    {
        li_set.at<float>(0,i) = Li.at<float>(set[i], 0);
        li_set.at<float>(1,i) = Li.at<float>(set[i], 1);
        li_set.at<float>(2,i) = Li.at<float>(set[i], 2);

        Lengths_set.at<float>(i,i) = Lengths.at<float>(set[i],set[i]);

        mi_set.at<float>(0,i) = Mi.at<float>(set[i], 0);
        mi_set.at<float>(1,i) = Mi.at<float>(set[i], 1);
        mi_set.at<float>(2,i) = Mi.at<float>(set[i], 2);
    }

    // Lev.-Marq. solution
    int m_dat = set_length;
    //int num_par = 3;
    int num_par = 2;

    // The starting point is the provided vp which is already calibrated
    if(VERBOSE_MSAC)
    {
        printf("\nInitial Cal.VP = (%.3f,%.3f,%.3f)\n", vp.at<float>(0,0), vp.at<float>(1,0), vp.at<float>(2,0));
        cv::Mat vpUnc = Mat(3,1,CV_32F);
        vpUnc = __K*vp;
        if(vpUnc.at<float>(2,0) != 0)
        {
            vpUnc.at<float>(0,0) /= vpUnc.at<float>(2,0);
            vpUnc.at<float>(1,0) /= vpUnc.at<float>(2,0);
            vpUnc.at<float>(2,0) = 1;
        }
        printf("Initial VP = (%.3f,%.3f,%.3f)\n", vpUnc.at<float>(0,0), vpUnc.at<float>(1,0), vpUnc.at<float>(2,0));
    }

    // Convert to spherical coordinates to move on the sphere surface (restricted to r=1)
    double x = (double)vp.at<float>(0,0);
    double y = (double)vp.at<float>(1,0);
    double z = (double)vp.at<float>(2,0);
    double r = cv::norm(vp);
    double theta = acos(z/r);
    double phi = atan2(y,x);

    if(VERBOSE_MSAC)
        printf("Initial Cal.VP (Spherical) = (%.3f,%.3f,%.3f)\n", theta, phi, r);

    //double par[] = {(double)vp.at<float>(0,0), (double)vp.at<float>(1,0), (double)vp.at<float>(2,0)};
    double par[] = {theta, phi};

    lm_control_struct control = lm_control_double;
    control.epsilon = 1e-5; // less than 1\BA
#if 0
    if(VERBOSE_MSAC)
        control.printflags = 2; //monitor status (+1) and parameters (+2), (4): residues at end of fit, (8): residuals at each step
    else
        control.printflags = 0;
#endif
    lm_status_struct status;
    data_struct data(li_set, Lengths_set, mi_set, __K);

    //lmmin(num_par, par, m_dat, &data, evaluateNieto, &control, &status, lm_printout_std);
    lmmin(num_par, par, m_dat, &data, evaluateNieto, &control, &status);

    if(VERBOSE_MSAC)
        printf("Converged Cal.VP (Spherical) = (%.3f,%.3f,%.3f)\n", par[0], par[1], r);

    // Store into vp
    // 1) From spherical to cartesian
    theta = par[0];
    phi = par[1];
    x = r*cos(phi)*sin(theta);
    y = r*sin(phi)*sin(theta);
    z = r*cos(theta);

    vp.at<float>(0,0) = (float)x;
    vp.at<float>(1,0) = (float)y;
    vp.at<float>(2,0) = (float)z;

}

// Error functions
float MSAC::errorLS(cv::Mat &Li, cv::Mat &vp, std::vector<float> &E, int *CS_counter)
{
    cv::Mat vn = vp;
    double vn_norm = cv::norm(vn);

    cv::Mat li;
    li = Mat(3,1,CV_32F);
    double li_norm = 0;
    float di = 0;

    float J = 0;
    for(int i=0; i<Li.rows; i++)
    {
        li.at<float>(0,0) = Li.at<float>(i,0);
        li.at<float>(1,0) = Li.at<float>(i,1);
        li.at<float>(2,0) = Li.at<float>(i,2);

        li_norm = cv::norm(li); // esto lo podria precalcular antes
        di = (float)vn.dot(li);
        di /= (float)(vn_norm*li_norm);

        E[i] = di*di;

        /* Add to CS if error is less than expected noise */
        if (E[i] <= __T_noise_squared)
        {
            __CS_idx[i] = 0;    // set index to 1
            (*CS_counter)++;

            // Torr method
            J += E[i];
        }
        else
        {
            J += __T_noise_squared;
        }
    }

    J /= (*CS_counter);

    return J;
}

float MSAC::errorNIETO(cv::Mat &Li, cv::Mat &Lengths, cv::Mat &Mi, cv::Mat &vp, std::vector<float> &E, int *CS_counter)
{
    float J = 0;
    float di = 0;

    cv::Mat lineSegment = Mat(3,1,CV_32F);
    float lengthLineSegment = 0;
    cv::Mat midPoint = Mat(3,1,CV_32F);

    // The vp arrives here calibrated, need to uncalibrate (check it anyway)
    cv::Mat vn = cv::Mat(3,1,CV_32F);
    double vpNorm = cv::norm(vp);
    if(fabs(vpNorm - 1) < 0.001)
    {
        // Calibrated -> uncalibrate
        vn = __K*vp;
        if(vn.at<float>(2,0) != 0)
        {
            vn.at<float>(0,0) /= vn.at<float>(2,0);
            vn.at<float>(1,0) /= vn.at<float>(2,0);
            vn.at<float>(2,0) = 1;
        }
    }

    for(int i=0; i<Li.rows; i++)
    {
        lineSegment.at<float>(0,0) = Li.at<float>(i,0);
        lineSegment.at<float>(1,0) = Li.at<float>(i,1);
        lineSegment.at<float>(2,0) = Li.at<float>(i,1);

        lengthLineSegment = Lengths.at<float>(i,i);

        midPoint.at<float>(0,0) = Mi.at<float>(i,0);
        midPoint.at<float>(1,0) = Mi.at<float>(i,1);
        midPoint.at<float>(2,0) = Mi.at<float>(i,2);

        di = distanceNieto(vn, lineSegment, lengthLineSegment, midPoint);

        E[i] = di*di;

        /* Add to CS if error is less than expected noise */
        if (E[i] <= __T_noise_squared)
        {
            __CS_idx[i] = 0;    // set index to 1
            (*CS_counter)++;

            // Torr method
            J += E[i];
        }
        else
        {
            J += __T_noise_squared;
        }

        J += E[i];
    }

    J /= (*CS_counter);

    return J;
}

void MSAC::drawCS(cv::Mat &im, std::vector<std::vector<std::vector<cv::Point> > > &lineSegmentsClusters, cv::Mat &vps)
{
    vector<cv::Scalar> colors;
    colors.push_back(cv::Scalar(0,0,255)); // First is RED
    colors.push_back(cv::Scalar(0,255,0)); // Second is GREEN 
    colors.push_back(cv::Scalar(255,0,0)); // Third is BLUE

    // Paint vp
    if( vps.at<float>(2,0) != 0 )
    {
        Point2f vp(vps.at<float>(0,0), vps.at<float>(1,0));

        // Paint vp if inside the image
        if(vp.x >=0 && vp.x < im.cols && vp.y >=0 && vp.y <im.rows)
        {
            circle(im, vp, 4, colors[0], 2);
            circle(im, vp, 3, CV_RGB(0,0,0), -1);
        }
    }
    // Paint line segments
    for(unsigned int c=0; c<lineSegmentsClusters.size(); c++)
    {
        for(unsigned int i=0; i<lineSegmentsClusters[c].size(); i++)
        {
            Point pt1 = lineSegmentsClusters[c][i][0];
            Point pt2 = lineSegmentsClusters[c][i][1];

            line(im, pt1, pt2, colors[c], 1);
        }
    }
}



