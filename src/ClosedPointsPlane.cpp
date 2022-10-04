#include "ClosedPointsPlane.h"

CPplane::CPplane(float d, Eigen::Vector3f normal){ 
    CPplane_ = normal * d;
    if(CPplane_.norm()==0){
        d_ = d;
        n_ = normal;
    }
    else{
        d_ = CPplane_.norm();
        n_ = CPplane_ / d_;
    }
}

CPplane::CPplane(CPp CPp) : CPplane_(CPp){
    if(CPplane_.norm()!=0){
        d_ = CPplane_.norm();
        n_ = CPplane_ / d_;
    }
    else{
        d_ = 0;
        n_ = Eigen::Vector3f(0,0,0);
    }

}

Eigen::Matrix<float, 1, 3> CPplane::Jplane(Eigen::Vector3f p){
    return (p.transpose()/getnorm() - (p.transpose() * CPplane_)*(CPplane_.transpose() / pow(getnorm(), 3)) - CPplane_.transpose()/getnorm());
}

CPplane TransformedPlane(CPplane plane, Eigen::Affine3f T){
    Eigen::Matrix3f R = T.rotation().matrix();
    Eigen::Vector3f t = T.translation().matrix();
    Eigen::Vector3f n = R * plane.n_;
    float d = - t.transpose() * plane.n_ + plane.d_;
    return CPplane(d, n);
}

CPplaneResidualCeres::CPplaneResidualCeres(const PointType pt) : pt_(pt){}

bool CPplaneResidualCeres::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const{
    CPplane CP(CPp(parameters[0][0], parameters[0][1], parameters[0][2]));
    // cout<<"parameters are: "<<parameters[0][0]<<" "<<parameters[0][1]<<" "<<parameters[0][2]<<endl;
    // cout<<CP.CPplane_<<endl<<endl;
    // cout<<CP.n_<<endl<<endl;
    // cout<<CP.d_<<endl<<endl;
    // cout<<pt_cov<<endl<<endl;

    // cout<<"infor"<<" "<<information<<endl;
    residuals[0] = ((CP.CPplane_.transpose()/CP.getnorm() * Eigen::Vector3f(pt_.x, pt_.y, pt_.z))(0,0) - CP.getnorm());

    if(jacobians){
        if(jacobians[0]){
            Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jac(jacobians[0]);
            jac.setZero();
            // cout<<"information is "<<information<<endl;
            jac = CP.Jplane(Eigen::Vector3f(pt_.x, pt_.y, pt_.z)).cast<double>();
            // cout<<"J is: "<<jac<<endl;
        }
    }

    return true;
}



