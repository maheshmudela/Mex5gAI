

#ifndef TRAINLMS_H
#define TRAINLMS_H

#include "stdafx.h"
#include <iostream>
using namespace std;
#define MAX_STAGES    100  // training for MAX stage possiblities, each stage has one scenrios to be tranined for.
#define FFT_LEN       1024

/*This header has  functionality of channel estimation based lms thechniqe so that the final estimated output is cascaded with
other stage and final estimated out is more accurate.

*/
class  TrainLmsFilter
{
public: 
	 TrainLmsFilter();
	 ~TrainLmsFilter();
    void ComputeChannelClassifier(unsigned int *f_puIiqRecvedSignal);

		// this will make variance/pca techniw to estimate the recved signal what has been transmiteed
		// corrupted with noise.
    void PredictTxData(unsigned int *f_puIiqRecvedSignal,
			            unsigned int *f_puIPredictVector ); 
        
	void TrainSignalForChannelEstimationBins(unsigned int *f_puIlocaldmrs,
										     unsigned int *f_uIrxDmrsSig );

private:

	//FILE *fp = NULL;

	// some varaibles and provate funtion
	void   ComputeFinalVector(unsigned int *f_puItempPredVector,
	                          unsigned int *f_uIPredictVector,
						      int           f_uIalpha);
	
   

	

};


#endif