/*
 * kalman.hpp
 *
 *  Created on: Mar 4, 2014
 *      Author: nik
 */

#ifndef KALMAN_HPP_
#define KALMAN_HPP_

class Kalman1D {
    public:
        Kalman1D(float initial, float measVar,  float procVar);
        ~Kalman1D();
        void addMeas(float meas);
        void skipMeas(void);
        float xHat;
    private:
        float xHatMinus, P, PMinus, K, KMinus;
        float R;	/* Measurement variance */
        float Q;	/* Process variance */
        void updateTime(void);
};

#endif /* KALMAN_HPP_ */
