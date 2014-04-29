/*
 * kalman.cpp
 *
 *  Created on: Mar 4, 2014
 *      Author: nik
 */

#include "kalman.hpp"

Kalman1D::Kalman1D(float initial, float measVar,  float procVar) {
    xHat = initial;
    R = measVar;
    Q = procVar;
    P = 0;
}

Kalman1D::~Kalman1D() {

}

void Kalman1D::updateTime(void) {
    xHatMinus = xHat;
    PMinus = P + Q;
}

void Kalman1D::addMeas(float meas) {
    updateTime();
    K = PMinus / (PMinus + R);
    /* Multiply error by Kalman gain K */
    xHat = xHatMinus + (K * (meas - xHatMinus));
    P = (1.0 - K) * PMinus;
}

void Kalman1D::skipMeas(void) {
    updateTime();
    /* Let xhat, P, and K remain. If we had a more specific state update we would
     * do it here.	*/
}
