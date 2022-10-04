#pragma once
#ifndef _CLOSED_POINTS_PLANE_H_
#define _CLOSED_POINTS_PLANE_H_

#include "utility.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>

typedef Eigen::Vector3f CPp;

class CPplane {
public:
    CPplane();
    CPplane(float d = 0.0, Eigen::Vector3f normal = Eigen::Vector3f(0.0,0.0,1.0));
    CPplane(CPp CPp);
    float getnorm(){ return abs(d_); }
    Eigen::Matrix<float, 1, 3> Jplane(Eigen::Vector3f p);

    // CPplane operator+(Vector3f& dCPp){
    //     this->CPplane_ += dCPp;
    //     if(this->CPplane_.norm()!=0){
    //     this->d_ = sqrt(this->CPplane_.norm());
    //     this->n_ = this->CPplane_ / this->d_;
    //     }
    // }
    // CPplane operator-(Vector3f& dCPp){
    //     this->CPplane_ -= dCPp;
    //     if(this->CPplane_.norm()!=0){
    //     this->d_ = sqrt(this->CPplane_.norm());
    //     this->n_ = this->CPplane_ / this->d_;
    //     }
    // }

    CPp CPplane_;
    Eigen::Vector3f n_;
    float d_;
};

class CPplaneResidualCeres : public ceres::SizedCostFunction<1, 3>{
public:
    CPplaneResidualCeres(const PointType pt);
    virtual ~CPplaneResidualCeres(){}
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

public:
    PointType pt_;
};

#endif