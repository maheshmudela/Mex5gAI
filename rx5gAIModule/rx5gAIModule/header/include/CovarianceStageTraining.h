/*This header has  functionality of channel estimation based covraince thechniqe so that the final estimated output is cascaded with
other stage and final estimated out is more accurate.

*/
#ifndef COVARIANCESTAGETRAINING_H
#define COVARIANCESTAGETRAINING_H
#include "stdafx.h"
#include <iostream>
using namespace std;

#define TRAIL_WINDOW_MS  10
#define MAX_STAGES    10 //100  // training for MAX stage possiblities, each stage has one scenrios to be tranined for.
#define FFT_LEN       1024

class CovarianceStageTraining
{
   public:
	     CovarianceStageTraining();
	     ~CovarianceStageTraining();
		void ComputeChannelClassifier(unsigned int *f_puIiqRecvedSignal);

		// this will make variance/pca techniw to estimate the recved signal what has been transmiteed
		// corrupted with noise.
  		void PredictTxData(unsigned int *f_puIiqRxDmrs,
						  unsigned int *f_puIPredictTxDmrs );
        
		void GetInfo(void *f_uiData, int type, int length); // type is phase of mag.

		void TrainSignalForChannelEstimationBins(unsigned int *f_puIlocaldmrs,
										         unsigned int *f_uIrxDmrsSig );

private:


#ifndef DISABLE_DEBUG
      FILE  *FpIq;
#endif

	    // K stages and each stage is train with same in put dmrs with M  trails
	    float GainweightCoeff[MAX_STAGES][FFT_LEN];
		float Phaseweightcoeff[MAX_STAGES][FFT_LEN];
		float GainVariance[TRAIL_WINDOW_MS][FFT_LEN];
		float PhaseVariance[TRAIL_WINDOW_MS][FFT_LEN];
		float drmsLocalPhase[FFT_LEN];
        float drmsLocalGain[FFT_LEN];
		unsigned int  uIPredictVector[FFT_LEN];
		float CovPhaseMatrix[TRAIL_WINDOW_MS][TRAIL_WINDOW_MS];
		float CovGainMatrix[TRAIL_WINDOW_MS][TRAIL_WINDOW_MS];
                
        int    StageCount; // track the stage
		int    timeCountMs; // Cunt no of time, sample trained vector is used with diffrent sceneraio.
		                   // reset once system is trained for that vector.
		bool   AlphaClassificationWeight[MAX_STAGES]; // it has either 0 or 1

	    // MAX_DIM = 4 is max dimensionality, M is vector.
        //double  phaseEigenVector[MAX_STAGES][MAX_DIM][M];// M is time window.
		// double GainEigenVector[MAX_STAGES][MAX_DIM][M];// M is time window.
		void   ComputeFinalVector(unsigned int *f_puItempPredVector,
	                              unsigned int *f_uIPredictVector,
						          int           f_uIalpha);

		void   GetPhase(void *f_uipPhase);
		void   GetMag  (void  *f_uipMag);

       int Init(void *args) ;

	   // Computing varinace matrix of gain and phase variations until
	   // time window TRAIL_WINDOW_MS.
	   int VarianceGainPhaseKXM(unsigned int *f_puIlocaldmrs,
								 unsigned int *f_puIrxDmrsSig,
								 int  f_timeCounterMs);


	   //Computing covraince matrix of gain and phase matrix after
	   // TRAIL_WINDOW_MS
	   void CovMatrixGainPhaseKXM(unsigned int *f_puIlocaldmrs,
			  			          unsigned int *f_puIrxDmrsSig) ;

	   float DotProductXY(float *x, float *y) ;
	   void   ComputeSVD(void);
	   void   computePCAFeatureVect(void) ;
	   

};

#endif