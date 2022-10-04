// This is a NEW implementation of the algorithm described in the
// following paper:
//    C. Hertzberg,  R.  Wagner,  U.  Frese,  and  L.  Schroder.  Integratinggeneric   sensor   fusion   algorithms   with   sound   state   representationsthrough  encapsulation  of  manifolds.
//    CoRR,  vol.  abs/1107.1119,  2011.[Online]. Available: http://arxiv.org/abs/1107.1119
/**
 * @file mtk/types/S2.hpp
 * @brief Unit vectors on the sphere, or directions in 3D.
 */
#ifndef S2_H_
#define S2_H_


#include "utility.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include "../sophus/so3.hpp"
#include <iostream>
using Sophus::SO3;

template<class scalar> inline scalar tolerance();

template<> inline float  tolerance<float >() { return 1e-5f; }
template<> inline double tolerance<double>() { return 1e-11; }

/**
 * Manifold representation of @f$ S^2 @f$. 
 * Used for unit vectors on the sphere or directions in 3D.
 * 
 * @todo add conversions from/to polar angles?
 */
template<class _scalar = double, int S2_typ = 3>
struct S2 {
	
	typedef _scalar scalar;
	typedef Eigen::Matrix<scalar, 3, 1> vec3;
	enum {DOF=2, TYP = 1, DIM = 3};
	
//private:
	/**
	 * Unit vector on the sphere, or vector pointing in a direction
	 */
	vec3 vec; 
	scalar length;
	
public:
	S2() {
		length = scalar(1);
		if(S2_typ == 3) vec=length * vec3(0, 0, std::sqrt(1));
		if(S2_typ == 2) vec=length * vec3(0, std::sqrt(1), 0);
		if(S2_typ == 1) vec=length * vec3(std::sqrt(1), 0, 0);
	} 
	S2(const scalar &x, const scalar &y, const scalar &z) : vec(vec3(x, y, z)) {
		length = vec.norm();
	}
	
	S2(const vec3 &_vec) : vec(_vec) {
		length = vec.norm();
	}

	S2(const S2<scalar, S2_typ>& other) 
	{
		vec = other.vec;
		length = other.length;
	}

	void oplus(const vec3& v)
	{
		Eigen::Matrix<scalar, 3, 3> rot = Sophus::SO3<scalar>::exp(v).matrix();
		vec = rot * vec;
	}
	
	void boxplus(Eigen::Matrix<scalar, 2, 1> delta) {
		Eigen::Matrix<scalar, 3, 2> Bx;
		S2_Bx(Bx);
		vec3 Bu = Bx*delta;
		Eigen::Matrix<scalar, 3, 3> rot = Sophus::SO3<scalar>::exp(Bu).matrix();
		vec = rot * vec;
	} 
	
	//self - other
	void boxminus(Eigen::Matrix<scalar, 2, 1>& res, const S2<scalar, S2_typ>& other) const {
		//check!!
		// scalar v_sin = (GetSkewMatrix(vec)*other.vec).norm();
		scalar v_sin = (GetSkewMatrix(other.vec)*vec).norm();
		scalar v_cos = vec.transpose() * other.vec;
		scalar theta = std::atan2(v_sin, v_cos);
		if(v_sin < tolerance<scalar>())
		{
			if(std::fabs(theta) > tolerance<scalar>() )
			{
				res[0] = 3.1415926;
				res[1] = 0;
			}
			else{
				res[0] = 0;
				res[1] = 0;
			}
		}
		else
		{
			S2<scalar, S2_typ> other_copy = other;
			Eigen::Matrix<scalar, 3, 2>Bx;
			other_copy.S2_Bx(Bx);
			res = theta/v_sin * Bx.transpose() * GetSkewMatrix(other.vec)*vec;
		}
	}
	
	void S2_hat(Eigen::Matrix<scalar, 3, 3> &res)
	{
		Eigen::Matrix<scalar, 3, 3> skew_vec;
		skew_vec << scalar(0), -vec[2], vec[1],
								vec[2], scalar(0), -vec[0],
								-vec[1], vec[0], scalar(0);
		res = skew_vec;
	}


	void S2_Bx(Eigen::Matrix<scalar, 3, 2> &res)
	{
		if(S2_typ == 3)
		{
		if(vec[2] + length > tolerance<scalar>())
		{
			
			res << length - vec[0]*vec[0]/(length+vec[2]), -vec[0]*vec[1]/(length+vec[2]),
					 -vec[0]*vec[1]/(length+vec[2]), length-vec[1]*vec[1]/(length+vec[2]),
					 -vec[0], -vec[1];
			res /= length;
		}
		else
		{
			res = Eigen::Matrix<scalar, 3, 2>::Zero();
			res(1, 1) = -1;
			res(0, 0) = 1;
		}
		}
		else if(S2_typ == 2)
		{
		if(vec[1] + length > tolerance<scalar>())
		{
			
			res << length - vec[0]*vec[0]/(length+vec[1]), -vec[0]*vec[2]/(length+vec[1]),
					 -vec[0], -vec[2],
					 -vec[0]*vec[2]/(length+vec[1]), length-vec[2]*vec[2]/(length+vec[1]);
			res /= length;
		}
		else
		{
			res = Eigen::Matrix<scalar, 3, 2>::Zero();
			res(1, 1) = -1;
			res(2, 0) = 1;
		}
		}
		else
		{
		if(vec[0] + length > tolerance<scalar>())
		{
			
			res << -vec[1], -vec[2],
					 length - vec[1]*vec[1]/(length+vec[0]), -vec[2]*vec[1]/(length+vec[0]),
					 -vec[2]*vec[1]/(length+vec[0]), length-vec[2]*vec[2]/(length+vec[0]);
			res /= length;
		}
		else
		{
			res = Eigen::Matrix<scalar, 3, 2>::Zero();
			res(1, 1) = -1;
			res(2, 0) = 1;
		}
		}
	}

	//a(other - self) / a(other)
	void S2_Nx(Eigen::Matrix<scalar, 2, 3> &res, S2<scalar, S2_typ>& subtrahend)
	{
		if((vec+subtrahend.vec).norm() > tolerance<scalar>())
		{
			Eigen::Matrix<scalar, 3, 2> Bx;
			S2_Bx(Bx);
			if((vec-subtrahend.vec).norm() > tolerance<scalar>())
			{
				scalar v_sin = (GetSkewMatrix(vec)*subtrahend.vec).norm();
				scalar v_cos = vec.transpose() * subtrahend.vec;
				
				res = Bx.transpose() * 
				(std::atan2(v_sin, v_cos)/v_sin*GetSkewMatrix(vec)+GetSkewMatrix(vec)*subtrahend.vec*((-v_cos/v_sin/v_sin/length/length/length/length+std::atan2(v_sin, v_cos)/v_sin/v_sin/v_sin)*subtrahend.vec.transpose()*GetSkewMatrix(vec)*GetSkewMatrix(vec)-vec.transpose()/length/length/length/length));
			}
			else
			{
				res = 1/length/length*Bx.transpose()*GetSkewMatrix(vec);
			}
		}
		else
		{
			std::cerr << "No N(x, y) for x=-y" << std::endl;
			std::exit(100);
		}
	}

	void S2_Nx_yy(Eigen::Matrix<scalar, 2, 3> &res)
	{
		Eigen::Matrix<scalar, 3, 2> Bx;
		S2_Bx(Bx);
		res = 1/length/length*Bx.transpose()*GetSkewMatrix(vec);
	}

	void S2_Mx(Eigen::Matrix<scalar, 3, 2> &res, Eigen::Matrix<scalar, 2, 1> delta)
	{
		Eigen::Matrix<scalar, 3, 2> Bx;
		S2_Bx(Bx);
		if(delta.norm() < tolerance<scalar>())
		{
			res = -GetSkewMatrix(vec)*Bx;
		}
		else{
			vec3 Bu = Bx*delta;
			Eigen::Matrix<scalar, 3, 3> rot = Sophus::SO3<scalar>::exp(Bu).matrix();
			res = -rot*GetSkewMatrix(vec)*Amatrix(Bu).transpose()*Bx;
		}
	}

	operator const vec3&() const{
		return vec;
	}
	
	const vec3& get_vect() const {
		return vec;
	}
	
	friend S2<scalar, S2_typ> operator*(const SO3<scalar>& rot, const S2<scalar, S2_typ>& dir)
	{
		S2<scalar, S2_typ> ret;
		ret.vec = rot * dir.vec;
		return ret;
	}
	
	scalar operator[](int idx) const {return vec[idx]; }
	
	friend std::ostream& operator<<(std::ostream &os, const S2<scalar, S2_typ>& vec){
		return os << vec.vec.transpose() << " ";
	}
	friend std::istream& operator>>(std::istream &is, S2<scalar, S2_typ>& vec){
		for(int i=0; i<3; ++i)
			is >> vec.vec[i];
		vec.vec.normalize();
		vec.vec = vec.vec * vec.length;
		return is;
	
	}
};


#endif /*S2_H_*/
