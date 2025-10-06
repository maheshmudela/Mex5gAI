
//include "stdafx.h"
#include "lmsStageTraining.h"


/*

	// impact of one fft compmenete to its nebhouring compmenets + diffent paths?? this has to be stidied..

	phase0fdmrs[i] =  phase0fdmrs[i-1] +  phase0fdmrs[i] + phase0fdmsr [i+1]
	gain0fdmrs[i]  =  gain0fdmrs[i-1] +  gain0fdmrs[i] + gain0fdmsr [i+1]
*/
 TrainLmsFilter::TrainLmsFilter() 
{
   // consttructor or init the class vraibales;

   

}
 TrainLmsFilter:: ~TrainLmsFilter() 
{
  
}

/* rx dmrs classification as at run time, once we get rx dmrs, lets calssify that what
kind of channel behaiour it has impacted with?? */
void TrainLmsFilter::ComputeChannelClassifier(unsigned int *f_puIiqRecvedSignal) 
{
   // Its dummy implementation for now , need implment lms  gradient decent or
	// define

     unsigned int *puIiqRecvedSignal = f_puIiqRecvedSignal;

	 for (int i = 0; i < FFT_LEN; ++i) 
	 {
           *puIiqRecvedSignal = *puIiqRecvedSignal*0.01;
		   ++puIiqRecvedSignal;
		   

	 }


}

void TrainLmsFilter::PredictTxData(unsigned int *f_puIiqRxDmrs,
								   unsigned int *f_puIPredictTxDmrs ) 
{
   //define;
}


void TrainLmsFilter::ComputeFinalVector(unsigned int *uItempPredVector,
                                          unsigned int *uIPredictVector,
										  int alpha) 
{

}

void TrainLmsFilter::TrainSignalForChannelEstimationBins(unsigned int *f_puIlocaldmrs,
												         unsigned int *f_puIrxDmrsSig)
{
   //trains
}