

#ifndef AI5GBASIC_H
#define AI5GBASIC_H

#include "stdafx.h"
#include "iostream"
using namespace std;
#include "lmsStageTraining.h"
#include "CovarianceStageTraining.h"

class TrainLmsFilter;
class CovarianceStageTraining;

#define MAX_PARITY_CHANNEL_MATRIX_ROW  5  // this no of parity bits
#define MAX_PARITY_CHANNEL_MATRIX_COL  10  // info bits 50-20, is info bits
#define THRESHOLD                      0.0

#define MAX_TIME_WINDOW  4  // this is trials, a same scenrio is repeated .
#define MAX_STAGES   100
#define FFT_LEN     1024
#define uIMaxRB     12
#define MODE_TYPE    2  // for qpsk, 

#define K_RANGE    100

#define Z_FACTOR    5

/*Implements basic 5g with AI   , for now no inheritance  channel estimation classes 
only as private call object*/
class Basic5gWithAIExp
{

public:
	  Basic5gWithAIExp();
	 ~Basic5gWithAIExp();
     void computeChannelResponse(unsigned int  *f_puiIQSigBuffer, 
		                         unsigned char *f_puCbitsStream);
	 void configureInitpdcch(unsigned int f_uIdmrsType,
						   unsigned int f_uIscramblingNID0,
						   unsigned int f_uIscramblingNID1,
						   unsigned int f_uIrnti,
						   unsigned int f_uIdci_formate,
						   unsigned int f_uIMaxRB,
						   unsigned int f_uIBetaDmrs,
						   unsigned int f_uISlotCnt,
						   unsigned int f_uIsymbCnt
						  );
	     void GetInfo(void *f_plotData, 
			          int    type, 
					  int    length); // type is phase of mag.

	 


private:

#ifndef DISABLE_DEBUG
      FILE  *Fp;
#endif

   // private variable and structrs
   int uICounterK; //
   int rxSeq[uIMaxRB*12];// complex dmrs seq extrcated , i and q 
   unsigned int uIstageBitErrCounter;
   unsigned int uIstageChannelCounter;
   unsigned int channelMatrix[MAX_PARITY_CHANNEL_MATRIX_ROW][MAX_PARITY_CHANNEL_MATRIX_COL];
   char         llr[MAX_PARITY_CHANNEL_MATRIX_COL*Z_FACTOR];
   bool  bitCorrectionAlpha[MAX_STAGES];
   bool  channelAlpha[MAX_STAGES];
   CovarianceStageTraining *CovarianceChEstimationObj;
   TrainLmsFilter          *lmsChEstimationObj;


   // Every scenario, there will bit coeffients that indicates probabilities of correctness.
   double BaysianProbCoeff[MAX_STAGES][MODE_TYPE*FFT_LEN];  
   double ChannelCoeff[MAX_STAGES][FFT_LEN];
   double phaseVariance[MAX_TIME_WINDOW][FFT_LEN];
   double GainVariance[MAX_TIME_WINDOW][FFT_LEN]; // MAXTIMEWINDOW = M slots
   double CovPhaseMatrix[MAX_TIME_WINDOW][MAX_TIME_WINDOW];
   double CovGainMatrix[MAX_TIME_WINDOW][MAX_TIME_WINDOW];
   unsigned char recvedBits[FFT_LEN*MODE_TYPE];
   unsigned char dmrsRefbits[FFT_LEN*MODE_TYPE];
   unsigned char CorrectedBitsInfo[FFT_LEN*MODE_TYPE];
   unsigned int uIDmrsEstimated[FFT_LEN];
   unsigned int uIlocaldmrs[FFT_LEN];
   unsigned int uIrxSig[FFT_LEN];
   void initMmembers(void) ;
   
   void ComputeChannelEstimate(unsigned int *f_puIiqRecvedSignal);
   void computeChannelRspOut  (unsigned int  *f_puiIQSigBuffer,
							   unsigned char *f_puCbitsStream);

   void ComputeFinalVector(unsigned int *f_puItempPredVector,
 	                       unsigned int *f_uIPredictVector,
						   int           f_uIalpha);
   void PredictChannel(unsigned int *f_puIiqRecvedSignal, 
	                   unsigned int *f_puIPredictVector ); 
   
   void ExtractDmrs(unsigned int *f_puIiqRecvedSignal,
	                int           dmrsType,
					int           f_uIBetaDmrs);


   
#if 1
   void computeQPSKBitErrorCoeffient(unsigned int *f_puIModLocalDmrsSignal, 
	                                 unsigned int *f_puIRxDemodSignal);
   void BayesianProbEstimationQpsk  (unsigned int *f_puIlocaldmrs,
									 unsigned int *f_puIrxDmrsSig );
#endif

   void BaysianProbBitErrClassification(unsigned char *f_pUdecodedBitsInfo,
									    unsigned char *f_dmrsRefbits,
									   unsigned char *f_puCorrectedBitsInfo);

   void BayesianProbEstimation(unsigned int *f_puIlocaldmrs,
							   unsigned int *f_puIrxDmrsSig );

   void BaysianProbBitErrCorrection(unsigned char *pEstimatedBit,
							  	   unsigned char *f_puCorrectedBitsInfo,
							       int            uIstageBitErrCounter) ;

   void TrainSignalForChannelEstimationBins(unsigned int *f_puIlocaldmrs,
										    unsigned int *f_uIrxDmrsSig );

   void TrainSignalForBitCorrectionBins(unsigned int *f_puIModLocalDmrsSignal,
										unsigned int *f_puIRxDemodSignal);

   // k IS FFT BIN, M is time window for which gain and phase has to be tracked
   // for making currnet estimation.
   void MatrixGainPhaseKXM(unsigned int *f_puIlocaldmrs,
						   unsigned int *f_puIrxDmrsSig);
   void CovMatrixGainPhaseKXM(unsigned int *f_puIlocaldmrs,
							   unsigned int *f_puIrxDmrsSig );

   void PhaseEstimation(unsigned int *f_puIRxIQSignal);
   void GainEstimation(unsigned int *f_puIRxIQSignal);
   void DelayEstimation(unsigned int *f_puIRxIQSignal);
   // private function:

   void channelCodingLdpc(int iterationMax );

};


#endif
