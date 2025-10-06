

//#include "stdafx.h"
#include "CovarianceStageTraining.h"
#include "math.h"

#define MODE(x)    (x) > 0 ? (x): -(x)  


//   slot[0][14] // oth time all 14 symbols
//   slot[1][14] // 1 ms time
//   slot[2][14]  // 2 ms , all symbol...

CovarianceStageTraining::CovarianceStageTraining() 
{
   // init an dconstrcuter
	Init(0); 
	
#ifndef DISABLE_DEBUG
     
	FpIq =  fopen("InputIQ.txt", "w+");
#endif

	
}

CovarianceStageTraining:: ~CovarianceStageTraining() 
{
  // destructor
}

int CovarianceStageTraining::Init(void *args) 
{
   // generate local dmrs and compute angle and gain 
   // Possible phase qpsk.. see constelation of qpsk
   // map it bits..

	return 0;

}






void CovarianceStageTraining::GetInfo(void *f_plotData, int type, int length) 
{

	float *pdata = (float *)f_plotData;

	for (int i = 0; i < length; ++i) 
	{
          *pdata = drmsLocalPhase[i];
		  ++pdata;
		   
	}
    


}




#if 0
/*At every call, k will increase , ensuring all other  or revoius k as been trained 
  while traing first tries with all prevoius and error is zero , it means this is trained 
  for this vector trialCount, is same pn seq repeaet for 10 times one slot window, tital count start from 0 to M*/
void CovarianceStageTraining::ComputeChannelClassifier(unsigned int *f_puIiqRecvedSignal, int trialCount) 
{

	  // locally generate dmrs
	  m = trialCount;

	  // extract dmrs from given reved  signal in freq domain
      dmrs_recved[bin] =  f_puIiqRecvedSignal;
	  
		  for all k = MAX_STAGES; //GainEigenVector
		  {

              for all bins  i
			  {
				  
					I = dmrs_recved[m][i];
					Q = dmrs_recved[m][I];
					RxphaserX = atan[I/Q];
					gain = 10log10(I*I +Q*Q );

				  for ( all dimenaion dim=0 to 4) 
						{
							
								// Update Phase estimation, later couldl be phase varinace can be think of instead of this as 
								// covarinace coeff..
								estPhaseRx = RxphaserX*PhaseEigenVector[k][dim][m];



								RxphaserX = estPhaseRx;
								
								estimatedGain = gain*GainEigenVector[k][dim][m];
								gain = estimatedGain;
								
							
							
				       }
                    localPhase;
                    localGain;
                    
				    // error sum of all vector , eclidian sum of all vector , such EVM 
                   SumErrorPhase += mod(localPhase - estPhaseRx);
				   sumErrorGain  += mod(localGain - estimatedGain);
				 
			  }	  
			  
               SumErrorPhase = SumErrorPhase/ALL_BIN_FFT_LENGTH
			
               // complex alpha and complex error
			   // alpha[k], IS classification flag indecating which posiibilities of dmrs corrputed with k's difrenet dyanmic of radio channels
			   // is best matched to currnet rx dmrs pattren.. 
		
			   channelAlpha[k] = 1/ SumErrorPhase;

		  }

}


#endif


/* rx dmrs classification as at run time, once we get rx dmrs, lets calssify that what
kind of channel behaiour it has impacted with?? */
void CovarianceStageTraining::ComputeChannelClassifier(unsigned int *f_puIiqRecvedSignal) 
{
    float RxphaserX = 0;
	float rateFactor = 0.0;
	float SumErrorPhase = 0.0f;
	float SumErrorGain = 0.0f;
	float estPhaseRx = 0;
	float estimatedGain = 0;
	float localPhase = 0;
	float localGain  = 0;
	float gain = 0;
    float I = 0;
	float Q = 0;
	unsigned int *puIDmrsRx;

#ifndef LOCAL_COMPILATION_FIXED
	  // locally generate dmrs
	  
	  // extract dmrs from given reved  signal in freq domain
      // dmrs_recved[bin] =  f_puIiqRecvedSignal;


	  puIDmrsRx = f_puIiqRecvedSignal;
	  for (int k = 0; k < MAX_STAGES; ++k)  //GainEigenVector
	  {
		   //printf(" training for stage %d \n", k);

           for ( int fft_bin = 0; fft_bin < FFT_LEN; ++fft_bin) 
		  {
				  
				I = (short)(puIDmrsRx[fft_bin] >> 16);
				Q = (short) puIDmrsRx[fft_bin];

				RxphaserX = atan((float)I/Q);
				gain      = 10*log10((float) (I*I +Q*Q) );
			
				// Update Phase estimation, later couldl be phase varinace can be
				//think of instead of this as covarinace coeff..
				 estPhaseRx    = RxphaserX*GainweightCoeff[k][fft_bin];
  				 estimatedGain = gain*Phaseweightcoeff[k][fft_bin];
				 						
                 localPhase = drmsLocalPhase[fft_bin];
                 localGain  = drmsLocalGain[fft_bin];
            

#ifndef DISABLE_DEBUG
		        //fprintf(FpIq, "bin index %d I %f  Q %f  \n", fft_bin, I, Q);	
#endif

				 // error sum of all vector , eclidian sum of all vector , such EVM 
                 SumErrorPhase += MODE((localPhase - estPhaseRx));
		   	     SumErrorGain  += MODE((localGain  - estimatedGain));
	
#ifndef DISABLE_DEBUG
				 fprintf(FpIq, "%d th stage: bin index:%d I: %f  Q: %f localPhase %f estPhaseRx %f  SumErrorPhase %f  localGain %f estimatedGain %f SumErrorGain %f \n",k, fft_bin, I, Q, localPhase,estPhaseRx , SumErrorPhase, localGain, estimatedGain, SumErrorGain);	
#endif

			  }	  
			  
               SumErrorPhase = SumErrorPhase/FFT_LEN;
			   SumErrorGain  = SumErrorGain/FFT_LEN;
			   rateFactor    = exp(SumErrorPhase*SumErrorGain);
			
               // complex alpha and complex error
			   // alpha[k], IS classification flag indecating which posiibilities of
			   // dmrs corrputed with k's difrenet dyanmic of radio channels
			   // is best matched to currnet rx dmrs pattren.. 
		
			   // Make sure this is boolean, if summ error  > some threshold, 
			   // then it is considered as 0 else 1
			   // hOW GAIN is added still need to work out, for any possibilities of K , one 
			   // has to t be true , rest false..
			   AlphaClassificationWeight[k] = 1.0f / rateFactor;
               printf(" training for stage %d  AlphaClassificationWeight %d \n", k, AlphaClassificationWeight[k]);

		  }
#endif
}

#if 0
void CovarianceStageTraining::dmrsExtraction(unsigned int *f_puIiqRxDmrs,
											 unsigned int *f_puIPredictTxDmrs ) 
{

      // Based on slot config, get location or dmrs type and extract dmrs
      // and save.

	  // Interpolation of dmrs over full grid



}
#endif

/*
for all k possibities of dmrs impacted k possiblites channel variations, need to chose which possiblities has least error.
            {   output of  k= 0   possiblites of channel variations
                
				error[k] = y_r(i) - weight[k][i]*x(i)( just aply stage 0 filter see  estmation errors
				alpha[k] = 1/error(k) // if error is low , alpha is 1 and else 0, this is classifire
                   
                      
               }

			  // now we have k possible classifications
              but only one callsifire is 1, based on trained wehits for that input, so say if y_r(i) is same as what we trained for stage 0, then 
			      alpha[0]= 1 , rest all are 0.

              
inputVector: this is input vector which has data and convlving this vector with channel estimate , give the output and get llrs
and for im cores..and pass the same to om and get crc..
*/
void CovarianceStageTraining::PredictTxData(unsigned int *f_puIiqRxDmrs,
									        unsigned int *f_puIPredictTxDmrs ) 
{
	float estimatedGain = 0.0;
	float RxphaserX = 0;
	float estPhaseRx = 0;
	float localPhase = 0;
	float localGain  = 0;
	float gain = 0;
    unsigned short I = 0;
	unsigned short Q = 0;
	unsigned int *puIiqRecvedSignal = f_puIiqRxDmrs;

	// K possiblites of feature vectors.
	for (int k = 0; k < MAX_STAGES ;  ++k) 
	{		  
		for (int i = 0; i < FFT_LEN; ++i)
		{				  
		      I = puIiqRecvedSignal[i];
			  Q = puIiqRecvedSignal[i];
			  RxphaserX = atan((float)I/Q);
			  gain = 10*log10((float)(I*I +Q*Q) );

			  //for ( all dimenaion dim=0 to 4) 
			{
							
			  // Update Phase estimation, later couldl be phase varinace can be think of instead of this as 
			  // covarinace coeff.
			  estPhaseRx = RxphaserX*Phaseweightcoeff[k][i];

			   //RxphaserX  = estPhaseRx;
								
			  estimatedGain = gain*GainweightCoeff[k][i];
			  //gain          = estimatedGain;
				
               I = (unsigned short)estimatedGain*cos(atan(estPhaseRx));
			   Q = (unsigned short)estimatedGain*sin(atan(estPhaseRx));
			   uIPredictVector[i] = I << 16 | Q; 
                    
		     }

            // this will sum all posbilities of tx dmrs and the one with max al
            ComputeFinalVector(&uIPredictVector[0],&uIPredictVector[0], AlphaClassificationWeight[k]);
 	}
}

}

void CovarianceStageTraining::ComputeFinalVector(unsigned int *uItempPredVector,
                                                 unsigned int *uIPredictVector,
										         int           alpha) 
{
   unsigned int *puItempPredV = uItempPredVector;
   unsigned int *puIPredictV  = uIPredictVector;

   // Only classsified alpha is 1 , rest is 0
	for (int i = 0; i < FFT_LEN; ++i) 
	{
	   *puIPredictV += (unsigned int)(*puItempPredV*alpha);
	   ++puIPredictV;
	   ++puItempPredV;

 
	}

}




// in time domain.. aftre down conversion and dwonsampling to bas band signal
// channelCoeff[k][i] = dmrs_recved_bin[i]/dmrs_local_bin[i]
// probablity dmrs_recved_bin[i] is same as  dmrs_local_bin[i]
// 

/*  */
void CovarianceStageTraining::TrainSignalForChannelEstimationBins(unsigned int *f_puIlocaldmrs,
														          unsigned int *f_puIrxDmrsSig)
{

    // for p 0 to 100 ranges 
   
   // we have dmrs signal in freq domain  with each RE
   // WE hAVE LOCAL REFRENCE dmrs RE   , 
	// we dmrs receved bin , with channel artifact such as delay/phase shft.. and other multipath effect
	//dmrs_local_bin[i] *channelCoeff[k][i] = dmrs_recved_bin[i]
    //k IS IN INCREMENTAL ORDER.

	// Note down chnages in phase at each bin, and chnages in chnagse in magnintude..
	// how is the chnages its linear or its forming some  equation.. thats is system behaiuor... and predict
	// and estimate the accuarcy of such behuor... and use such prediction when when the recved data is not mdrs to correct the
	// the phase and mag.. of any kind of noise in particlur brust of bins...and suppose every time u getting loss or noisy
	// data in some bands of bins.. then in case of real data.. we need to check how much is probalaity that particylar 
	// set of bpsk or qpsk/m-ary.. signal is close to actaul in dmrs loss study.. and hence prodict the losss??

	// m is time window.. pdcch symbol..1ms 
	// dimension is kXm, this covriance matrix is same as filter coeffienct
	 // This will compute variance in phase over m trails uptO M slots.
     VarianceGainPhaseKXM (f_puIlocaldmrs,f_puIrxDmrsSig, timeCountMs );  // 
     ++timeCountMs;
     
	 // At mth trail now compute covraince matrix, and from covrainve matrix compute svd and filter weights
	 if ( timeCountMs == TRAIL_WINDOW_MS) 
	 {
		 CovMatrixGainPhaseKXM(f_puIlocaldmrs,f_puIrxDmrsSig);

		// Once for any kth possibilitues of channel characteristsic
		// , m trail got completed , we have svd coeff computed for this currnt stage.
        ComputeSVD();

		++StageCount;
		timeCountMs = 0;
	 }


}


/*

fid = fopen('iqdump.bin', 'r');
%% Read the data to a variable
data = fread(fid, '*int16');
%% close the file after reading the data
fclose(fid);
cplxdata= complex(data(1:2:end),data(2:2:end));
figure;plot(abs(cplxdata));
% PCA1: Perform PCA using covariance.
% data - MxN matrix of input data
% (M dimensions, N trials)
% signals - MxN matrix of projected data
% PC - each column is a PC
% V - Mx1 matrix of variances
cplxdata = cplxdata(1:14*2192);
cplxdata_re = reshape(cplxdata,[],14);
cplxdata_re = cplxdata_re(145:end, :);

data_fft=fft(cplxdata_re);

[M,N] = size(data_fft);

% subtract off the mean for each dimension
mn = mean(data_fft,2);

% repeat mean all mxn matrix , at every element of mxn 
% data_fft value - mean
data_fft = data_fft - repmat(mn,1,N);
% calculate the covariance matrix
covariance = 1 / (N-1) * data_fft' * data_fft;
% find the eigenvectors and eigenvalues
[PC, V] = eig(covariance);
 surf([1:14], [1:14], abs(covariance))
% extract diagonal of matrix as vector
V = diag(V);
% sort the variances in decreasing order
[junk, rindices] = sort(-1*V);
V = V(rindices);
PC = PC(:,rindices);
PC = PC(:,end);
% project the original data set
%signals = PC' * data_fft';
signals = PC' * data_fft';


*/
int CovarianceStageTraining::VarianceGainPhaseKXM(unsigned int *f_puIlocaldmrs,
									              unsigned int *f_puIrxDmrsSig,
												  int           timeCountMs) 
{
	 short I = 0;
	 short Q = 0;
	 short Ir = 0;
	 short Qr = 0;
	 float RxGain = 0.0;
	 float localArgs = 0.0;
     float rxArgs;
	 float localGain;
	 float gaindiffDb[FFT_LEN];
	 float PhaseShiftDiff[FFT_LEN];
	 float MeanGain = 0.0;
	 float meanPhaseShiftDiff = 0.0;


#ifndef LOCAL_COMPILATION_FIXED
	for (int fft_k = 0; fft_k < FFT_LEN; ++fft_k) 
	{
		localArgs          = atan((float)I/Q);
		rxArgs             = atan((float)Ir/Qr);
		localGain          = 10*log10((float)(I*I   + Q*Q));
		RxGain             = 10*log10((float)(Ir*Ir + Qr*Qr));
		gaindiffDb[fft_k]  = localGain - RxGain;  // gain variation

		MeanGain               += gaindiffDb[fft_k];
		PhaseShiftDiff[fft_k]   = localArgs - rxArgs; 
        meanPhaseShiftDiff     += PhaseShiftDiff[fft_k];

	}

     MeanGain = MeanGain/FFT_LEN;

     meanPhaseShiftDiff = meanPhaseShiftDiff/FFT_LEN;

	for (int fft_k = 0; fft_k < FFT_LEN; ++fft_k)
	{

		// td is clss varivale provate memeor that track time window
	   // complete memmove
		GainVariance[timeCountMs][fft_k] =  gaindiffDb[fft_k] - MeanGain;
        PhaseVariance[timeCountMs][fft_k] = PhaseShiftDiff[fft_k] - meanPhaseShiftDiff;

	}

	// Moving average 
   
  
#endif
	return 0;

}


// In practical implementations, especially with high dimensional data (large p)
// p is M, its, or same dmrs is repeated for M time .. to capture impact of time
// on all RE
void CovarianceStageTraining::CovMatrixGainPhaseKXM(unsigned int *f_puIlocaldmrs,
											        unsigned int *f_puIrxDmrsSig ) 
{

#ifndef LOCAL_COMPILATION_FIXED
   //MXN , covraince.. 


	// For all trail 
	for (int TimeCnt1 = 0; TimeCnt1 <TRAIL_WINDOW_MS; TimeCnt1++ ) 
	{

		for ( int TimeCnt2 = 0; TimeCnt2 < TRAIL_WINDOW_MS; TimeCnt2++)
		{
			     // Compute corrleation of vraince at TimeCnt  with all varince pattren at TimeCnt2 
	        	//CovPhaseMatrix[TimeCnt1][TimeCnt2] = CorrelationXY(&phaseVariance[TimeCnt1][0],&phaseVariance[TimeCnt2][0]);
				//CovGainMatrix[TimeCnt1][TimeCnt2] =  CorrelationXY(&GainVariance[TimeCnt1][0],&GainVariance[TimeCnt2][0]);


                // Compute corrleation of vraince at TimeCnt  with all varince pattren at TimeCnt2 
	        	CovPhaseMatrix[TimeCnt1][TimeCnt2] = DotProductXY(&PhaseVariance[TimeCnt1][0],&PhaseVariance[TimeCnt2][0]);
				CovGainMatrix[TimeCnt1][TimeCnt2] =  DotProductXY(&GainVariance[TimeCnt1][0],&GainVariance[TimeCnt2][0]);



		}
	}

   // U have M*M coravraince Matrix, which is diagonal elemnt is around unity..or max and rest are less < 1
	// take the colum with maximun eigen value and 

   // so prodction of dmrs from local is estimation of rx dmrs
#endif

}



float CovarianceStageTraining::DotProductXY(float *x, float *y) 
{
	float *pX = x;
	float *pY = y;
	double prodSum=0.0f;

	for( int i= 0; i < FFT_LEN; ++i) 
	{
		 // dot product of 2 series, something comparing two series as enery
         prodSum += *pX * *pY;
		 ++pX;
		 ++pY;
	}

   return (float)prodSum;

}


// Y=XA, X is tranmited dmrs and Y is recved dmrs
// This is not svd but its something digonal matrix of covraince matrix,
// which characterizes the channel capturing a particular scenrios of env.
void CovarianceStageTraining::ComputeSVD(void)
{


#ifndef LOCAL_COMPILATION_FIXED
	// for now diagonal element of covraince matrix svd
	// 1/svd is weight coeffiencet 

	for ( int diag_i = 0;  diag_i < TRAIL_WINDOW_MS; ++diag_i) 
	{
		// why we taking 1/svd, is something like invesring the imact in rx dmrs to get trasnmited 
		// 
	     Phaseweightcoeff[StageCount][diag_i] = 1.0/CovPhaseMatrix[diag_i][diag_i];
		 GainweightCoeff[StageCount][diag_i]  = 1.0/CovGainMatrix[diag_i][diag_i];
	}

#endif
}


// need to check lator
void CovarianceStageTraining::computePCAFeatureVect(void) 
{

#ifdef LOCAL_COMPILATION_FIXED
	// get diagonal vector.
	/*
    V = diag(V);
% sort the variances in decreasing order
[junk, rindices] = sort(-1*V);
V = V(rindices);
PC = PC(:,rindices);
PC = PC(:,end);

% project the original data set
%signals = PC' * data_fft';
signals = PC' * data_fft';
*/

#endif

}








