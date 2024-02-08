#pragma once

#include "marchee.cuh"

#define SCORE_Y_COEF_BRUIT 0.0

#define P_S      2.0
#define P_somme  1.0
#define P_coef   0.0

#define sng(x)	((x>=0) ? 1.0 : -1.0)

#define S(y,w)  (powf(y-w, P_S)/P_S)
#define dS(y,w) (powf(y-w, P_S-1))
#define K(p1,p0) (powf(fabs(100*(p1/p0 - 1)),P_coef))

#define __SCORE(y,p1,p0)  (K(p1,p0) * S(y, sng(p1/p0 - 1)) )
#define __dSCORE(y,p1,p0) (K(p1,p0) * dS(y, sng(p1/p0 - 1)) )

//	----

static float SCORE(float y, float p1, float p0) {
	return __SCORE(y,p1,p0);
};

static float APRES_SCORE(float somme) {
	return powf(somme, P_somme) / P_somme;
};

static float dAPRES_SCORE(float somme) {
	return powf(somme, P_somme - 1);
};

static float dSCORE(float y, float p1, float p0) {
	return __dSCORE(y,p1,p0);
};

//	----

static __device__ float cuda_SCORE(float y, float p1, float p0) {
	return __SCORE(y,p1,p0);
};

static __device__ float cuda_dSCORE(float y, float p1, float p0) {
	return __dSCORE(y,p1,p0);
};

//	S(x) --- Score ---

float  intel_somme_score(float * y, uint depart, uint T);
float nvidia_somme_score(float * y, uint depart, uint T);

float  intel_score_finale(float somme, uint T);
float nvidia_score_finale(float somme, uint T);

//	dx

float d_intel_score_finale(float somme, uint T);
float d_nvidia_score_finale(float somme, uint T);

void  d_intel_somme_score(float d_somme, float * y, float * dy, uint depart, uint T);
void d_nvidia_somme_score(float d_somme, float * y, float * dy, uint depart, uint T);

//	%% Prediction

float* intel_prediction(float * y, uint depart, uint T);
float* nvidia_prediction(float * y, uint depart, uint T);