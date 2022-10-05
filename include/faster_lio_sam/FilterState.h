#ifndef _INCLUDE_FILTER_STATE_H_
#define _INCLUDE_FILTER_STATE_H_
#include <eigen3/Eigen/Dense>
#include <Eigen/Core>
#include <sophus/so3.hpp>
#include "faster_lio_sam/S2.hpp"
#define POS_ 0
#define ROT_ 3
#define VEL_ 6
#define BIA_ 9
#define BIG_ 12
#define GW_ 15
enum class GravityType{Normal, S2M};

class FilterState
{
public:
    Eigen::Vector3f rn_;
    Eigen::Vector3f vn_;   // velocity in n-frame
    Eigen::Quaternionf qbn_;  // rotation from b-frame to n-frame
    Eigen::Vector3f ba_;   // acceleartion bias
    Eigen::Vector3f bw_;   // gyroscope bias

    GravityType GT;
    Eigen::Vector3f gn_;   // gravity
    S2<float> Gs2; 

public:
    FilterState()
    {
        GT = GravityType::Normal;
        setIdentity();
    };
    FilterState(const GravityType& gt) : GT(gt) 
    {
        setIdentity();
    }
    FilterState(const Eigen::Vector3f& rn, const Eigen::Vector3f& vn,
                const Eigen::Quaternionf& qbn, const Eigen::Vector3f& ba,
                const Eigen::Vector3f& bw, const Eigen::Vector3f& gn) : 
                rn_(rn), vn_(vn), qbn_(qbn), ba_(ba), bw_(bw), gn_(gn){};

    FilterState& operator=(const FilterState& state);
    FilterState& operator+(const Eigen::MatrixXf& deltaX);
    FilterState& operator+=(const Eigen::MatrixXf& deltaX);
    Eigen::MatrixXf operator-(const FilterState& state);

    ~FilterState(){}

    void setIdentity();

    void updateVelocity(float dt);
};


FilterState& FilterState::operator+(const Eigen::MatrixXf& deltaX)
{
    Eigen::Vector3f delta_angle_axis(deltaX.block<3, 1>(3, 0));
    Eigen::Quaternionf dq(Eigen::AngleAxisf(delta_angle_axis.norm(), delta_angle_axis.normalized()).toRotationMatrix());
    qbn_ *= dq;
    qbn_.normalize();
    rn_ += deltaX.block<3, 1>(POS_, 0);
    vn_ += deltaX.block<3, 1>(VEL_, 0);
    ba_ += deltaX.block<3, 1>(BIA_, 0);
    bw_ += deltaX.block<3, 1>(BIG_, 0);
    if(GT == GravityType::Normal)
        gn_ += deltaX.block<3, 1>(GW_, 0);
    else
    {
        Gs2.boxplus(deltaX.block<2, 1>(GW_, 0));
    }
    
    return *this;
}

FilterState& FilterState::operator+=(const Eigen::MatrixXf& deltaX)
{
    Eigen::Vector3f delta_angle_axis(deltaX.block<3, 1>(3, 0));
    Eigen::Quaternionf dq(Eigen::AngleAxisf(delta_angle_axis.norm(), delta_angle_axis.normalized()).toRotationMatrix());
    qbn_ *= dq;
    rn_ += deltaX.block<3, 1>(POS_, 0);
    vn_ += deltaX.block<3, 1>(VEL_, 0);
    ba_ += deltaX.block<3, 1>(BIA_, 0);
    bw_ += deltaX.block<3, 1>(BIG_, 0);
    if(GT == GravityType::Normal)
        gn_ += deltaX.block<3, 1>(GW_, 0);
    else
    {
        Gs2.boxplus(deltaX.block<2, 1>(GW_, 0));
    }
    
    return *this;
}

Eigen::MatrixXf FilterState::operator-(const FilterState& state)
{
    Eigen::MatrixXf ret;
    if(GT == GravityType::Normal) 
        ret.resize(18, 1);
    else
        ret.resize(17, 1);

    // Eigen::AngleAxisf deltaAa = Eigen::AngleAxisf(state.qbn_.inverse() * qbn_);
    Eigen::AngleAxisf deltaAa = Eigen::AngleAxisf(qbn_.inverse() * state.qbn_);

    ret.block<3, 1>(POS_, 0) = state.rn_ - rn_; 
    ret.block<3, 1>(ROT_, 0) = deltaAa.axis() * deltaAa.angle();
    ret.block<3, 1>(VEL_, 0) = state.vn_ - vn_;
    ret.block<3, 1>(BIA_, 0) = state.ba_ - ba_;
    ret.block<3, 1>(BIG_, 0) = state.bw_ - bw_;
    if(GT == GravityType::Normal)
        ret.block<3, 1>(GW_, 0)  = state.gn_ - gn_;
    else
    {
        Eigen::Matrix<float, 2, 1> tem;
        state.Gs2.boxminus(tem, Gs2);
        ret.block<2, 1>(GW_, 0) = tem;
    }

    return ret;
}

FilterState& FilterState::operator=(const FilterState& state)
{
    qbn_    = state.qbn_;
    rn_     = state.rn_;
    vn_ = state.vn_;
    ba_ = state.ba_;
    bw_ = state.bw_;
    
    if(state.GT == GravityType::Normal)
    {
        GT = GravityType::Normal;
        gn_ = state.gn_;
    }
    else
    {
        GT = GravityType::S2M;
        Gs2 = state.Gs2;
    }

    return *this;
}

void FilterState::setIdentity()
{
    rn_.setZero();
    vn_.setZero();
    qbn_.setIdentity();
    ba_.setZero();
    bw_.setZero();
    gn_ << 0.0, 0.0, -9.805;
}

void FilterState::updateVelocity(float dt)
{
    vn_ = rn_ / dt;
}



Eigen::Quaternionf axis2Quat(const Eigen::Vector3f &vec)
{
    Eigen::Quaternionf q;
    float alpha = vec.norm();
    Eigen::AngleAxisf rotation_vector(alpha, vec.normalized());
    q = rotation_vector;
    return q;
}
#endif