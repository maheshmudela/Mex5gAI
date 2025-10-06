/* Add comments */
//#include "stdafx.h"
#include "AI5gBasic.h"
#include <math.h>
//#define DISABLE_DEBUG 1010

 Basic5gWithAIExp::Basic5gWithAIExp() 
{
   // init and constructor

	initMmembers();

#ifndef DISABLE_DEBUG
     
	Fp =  fopen("Analysys.txt", "w+");
    fprintf(Fp, "Basic5gWithAIExp contsructor \n");

#endif

}


 void Basic5gWithAIExp::initMmembers(void)  
 {

     lmsChEstimationObj          = (TrainLmsFilter *)new TrainLmsFilter;
	 CovarianceChEstimationObj   = (CovarianceStageTraining *) new CovarianceStageTraining ;
	
 }

 Basic5gWithAIExp::~Basic5gWithAIExp() 
{
   // deinit and destructor
#ifndef DISABLE_DEBUG
	fprintf(Fp, "Basic5gWithAIExp destructor \n");
	fclose(Fp);
	Fp = NULL;

#endif
	delete lmsChEstimationObj;
	delete CovarianceChEstimationObj;

}


 void Basic5gWithAIExp::GetInfo(void *f_uiData, int type, int length) 
 {
      // this function can access private data and memeober functions.
      CovarianceChEstimationObj->GetInfo(f_uiData , type, length);


 }





/* It reads any signal , idenety which channel and extract the information bits */
void Basic5gWithAIExp::computeChannelResponse(unsigned int  *f_puiIQSigBuffer,
											  unsigned char *f_puCbitsStream) 
{
	computeChannelRspOut(f_puiIQSigBuffer,f_puCbitsStream);
}


void Basic5gWithAIExp::computeChannelRspOut  (unsigned int  *f_puiIQSigBuffer,
											  unsigned char *f_puCbitsStream) 
{  
        unsigned int *puiIQSigBuffer = f_puiIQSigBuffer;
		unsigned int *puiDmrsEstimatBuffer = &uIDmrsEstimated[0];

	   // fft

	   // extract dmrs

	   // We can have channel estimate in two stage in casacde way
	   // for example, channel estimate due to lms alogiruth and what evever is final estimated out
	    // of lms , gaian will be go to covariance and whoch output is furrther move to next state
	   // so training for particular is in differnet stages, same scenerion input is used to train 
	   // muliple casecade stage estimation coefeinet so that this will give more accuracy..
	   
	   // Channel estimate and classify the channel..
	   //lmsChEstimationObj->ComputeChannelClassifier(puiIQSigBuffer);

	   printf(" computig channe rsp out \n");
	   //ComputeChannelEstimate(uiIQSigBuffer);

	   //it will predict smrs which is transmited witout any noise and bit error
	   //lmsChEstimationObj->PredictTxData(puiIQSigBuffer, puiDmrsEstimatBuffer) ;
       
	   //This 2nd stage will enhance the first estimation.
	   puiIQSigBuffer = &uIDmrsEstimated[0];
	   puiIQSigBuffer = f_puiIQSigBuffer;

     
	   //cascaded with cov, 
	   // this will estimate and does channel classification.
	   CovarianceChEstimationObj->ComputeChannelClassifier(puiIQSigBuffer);

	   //ComputeChannelEstimate(uiIQSigBuffer);

	   //it will predict smrs which is transmited witout any noise and bit error
	   CovarianceChEstimationObj->PredictTxData(puiIQSigBuffer, puiDmrsEstimatBuffer) ;


	   // extract RE out AS LLRS and see om crc passes
       // CovarianceChEstimationObj->computeBits(puiDmrsEstimatBuffer, &recvedBits[0]);
	    
	   // This will classify bits error pattren.
	   BaysianProbBitErrClassification(&recvedBits[0],
		                               &dmrsRefbits[0], 
									   &CorrectedBitsInfo[0] );

	   BaysianProbBitErrCorrection(&recvedBits[0],
		                           &recvedBits[0],
								   0);
	   
       printf("ldpc caling \n");
       channelCodingLdpc(2);

	   //As already its trained for biterror detection.

	   int crc = 0;
	   if( crc ) 
	   {


	   } else 
	   {
            // complete traing is based on dmrs...if crc fails
		   //Once failed or initial  training phase is such that we can repeated the same dmrs M times ( time windwo)
		   // such covraince matrix capture the dimensionaity...
       
		   // train the system o ai for delay/doppler fading and channel estimation is computed.
		   lmsChEstimationObj->TrainSignalForChannelEstimationBins(&uIlocaldmrs[0], &uIrxSig[0] ) ;

		   CovarianceChEstimationObj->TrainSignalForChannelEstimationBins(&uIlocaldmrs[0], &uIrxSig[0]) ;
            
		   //Train Bit correction  .
		  // TrainSignalForBitCorrectionBins(dmrs_Receved_featureVect);
        
	   }


}




/*At every call, k will increase , ensuring all other  or revoius k as been trained 
  while traing first tries with all prevoius and error is zero , it means this is trained 
  for this vector*/
void Basic5gWithAIExp::ComputeChannelEstimate(unsigned int *f_puIiqRecvedSignal) 
{

#ifdef LOCAL_COMPILATION_FIXED
	  // locally generate dmrs

	  // extract dmrs from given reved  signal in freq domain
      dmrs_recved[bin] =  f_puIiqRecvedSignal;
	  
		  for all k 
		  {

              for all bins  i
			  {
                  I = dmrs_recved[i]

				  Q = dmrs_recved[I]
				  RxphaserX = atan[I/Q]
 
				  // Update Phase estimation, later couldl be phase varinace can be think of instead of this as 
				  // covarinace coeff..
                   estPhaseRx = RxphaserX*phaseCoef[k];

				   gain = 10log10(I*I +Q*Q );
                   estimatedGain = gain*GainCoeff[k];
                    
                    localPhase;
                    localGain;
                    
				    // error sum of all vector , eclidian sum of all vector , such EVM 
                   errorPhase[k][i] = mod(localPhase - estPhaseRx);
				   errorGain[k][i] = mod(localGain - estimatedGain);

				  
			  }
               error[k][i] = error[k][i]/ALL_BIN_FFT_LENGTH
			
               // complex alpha and complex error
			   alpha[k] = 1/ error[k][i]


		  }
#endif

}


void Basic5gWithAIExp::ExtractDmrs(unsigned int *f_puIiqRecvedSignal,
								   int           dmrsType ,
								   int           f_uIBetaDmrs
								   ) 
{

#ifdef LOCAL_COMPILATION_FIXED
    // EACH RB has 12 RE
	for (RBCount = 0; RBCount < uIMaxRB; ++RBCount) 
	{
	   // genereate  dmrs
	   // t38.211: 7.4.1.3 Demodulation reference signals for PDCCH

	   // Re 0 of coreset. for every RB of 12 RE, 3 are dmrs
       ReIndexK0 = RBCount*12 + 4*0 + 1; 
	   ReIndexK1 = RBCount*12 + 4*1 + 1;
	   ReIndexK2 = RBCount*12 + 4*2 + 1;

       rxSeq[3*RBCount + 0] = f_puIiqRecvedSignal[ReIndexK0]/f_uIBetaDmrs;
	   rxSeq[3*RBCount + 1] = f_puIiqRecvedSignal[ReIndexK1]/f_uIBetaDmrs;
	   rxSeq[3*RBCount + 2] = f_puIiqRecvedSignal[ReIndexK2]/f_uIBetaDmrs;
 	}


#endif

}

/* The reference point for k is - subcarrier 0 of the lowest-numbered 
resource block in the CORESET if the CORESET is configured by the 
PBCH or SIB1, 
- subcarrier 0 in common resource block 0 otherwise */
void Basic5gWithAIExp::configureInitpdcch(unsigned int f_uIdmrsType,
										  unsigned int f_uIscramblingNID0,
						                  unsigned int f_uIscramblingNID1,
						                  unsigned int f_uIrnti,
						                  unsigned int f_uIdci_formate,
										  unsigned int f_uIMaxRB,
										  unsigned int f_uIBetaDmrs,
										  unsigned int f_uISlotCnt,
										  unsigned int f_uIsymbCnt
										  ) 
{

#ifdef LOCAL_COMPILATION_FIXED
	c_init = power(2,17)*(14*f_uISlotCnt+f_uIsymbCnt+1)(2*f_uIscramblingNID1 +1)+ 2*f_uIscramblingNID1);
	c_init = MOD(c_init,power(2,31);

	// genereate  dmrs
	for {n= 0; n <  ; +n} 
	{
      // genereate  dmrs
      refSeq[n] = (1/2)*

	}

	// EACH RB has 12 RE
	for (RBCount = 0; RBCount < uIMaxRB; ++RBCount) 
	{
	   // genereate  dmrs
	   // t38.211: 7.4.1.3 Demodulation reference signals for PDCCH

	   // Re 0 of coreset. for every RB of 12 RE, 3 are dmrs
       ReIndexK0 = RBCount*12 + 4*0 + 1; 
	   ReIndexK1 = RBCount*12 + 4*1 + 1;
	   ReIndexK2 = RBCount*12 + 4*2 + 1;

       localdmrsalpha[ReIndexK0] = f_uIBetaDmrs*refSeq[3*RBCount + 0];
	   localdmrsalpha[ReIndexK1] = f_uIBetaDmrs*refSeq[3*RBCount + 1];
	   localdmrsalpha[ReIndexK2] = f_uIBetaDmrs*refSeq[3*RBCount + 2];

	}


#endif

}

#if 1

/*
for all k stages, need to chose which stage has least error.
            {   output of  k= 0   stage
                
				         error[k] = y_r(i) - weight[k][i]*x(i)( just aply stage 0 filter see  estmation errors
						 alpha[k] = 1/error(k) // if error is low , alpha is 1 and else 0, this is classifire
                         
                      
               }

			  // now we have k possible classifications
              but only one callsifire is 1, based on trained wehits for that input, so say if y_r(i) is same as what we trained for stage 0, then 
			      alpha[0]= 1 , rest all are 0.

              
inputVector: this is input vector which has data and convlving this vector with channel estimate , give the output and get llrs
and for im cores..and pass the same to om and get crc..
*/
void Basic5gWithAIExp::PredictChannel(unsigned int *f_puIiqRecvedSignal,
									  unsigned int *f_puIPredictVector ) 
{

    unsigned int tempPredVector[FFT_LEN];

	for (int k=0;  k < K_RANGE ;  ++k) 
	{
		 // Convolution(ChannelCoeff[k][0], &f_puIiqRecvedSignal[0], &tempPredVector[0]);
          ComputeFinalVector(&tempPredVector[0],&f_puIPredictVector[0], channelAlpha[k]);
 	}
}


void Basic5gWithAIExp::ComputeFinalVector(unsigned int *uItempPredVector,
                                          unsigned int *uIPredictVector,
										  int alpha) 
{

	for (int i = 0; i < FFT_LEN; ++i) 
	{
	   uIPredictVector[0] += uItempPredVector[0]*alpha;
 
	}

}


/*
  00,01,10,11, :
  , if estimaton of probabalit of getting corect bit > threshols, means we getting correct
  bit and alpha[k] = 0, not need to invert bits , and what ever we receved is correct
  and if estimation goes less then threshols then alpha[k] = 1 and means bit is not correct
  and invert the bits. lost bits are in category of wrong bits

  in qpsk model
  B: is event of getting correct bit
  V: is event that observed bit or recved bit is correct
  F: is event of getting wrong bit

  p(B/V),this estimated 
  p(B/V) = p(V/B)*P(B)/( p(V/B)*P(B) + p(V/F)*P(F)

  qpsk and if 10 RE are there so alph[k] where 0 k < 20
  This is designed for only qpsk
  Called during training for bit error.
*/
void Basic5gWithAIExp::computeQPSKBitErrorCoeffient(unsigned int *f_uIModLocalDmrsSignal,
													unsigned int *f_uIRxDemodSignal) 
{

#ifdef LOCAL_COMPILATION_FIXED

	 // The Re are i, q signal are demoluated based on some mmse tecegniw
     qpskDemod(puImodulatedSignal, PuOutBitPattren) ;

	 BayesianProbEstimationQpsk(puIlocaldmrs,puIrxDmrsSig); 

#endif


}


//
void Basic5gWithAIExp::BaysianProbBitErrClassification(unsigned char *f_pUdecodedBitsInfo,
												       unsigned char *f_dmrsRefbits,
												       unsigned char *f_puCorrectedBitsInfo) 
{


/* Sems the probalaity density function of releige models define the probalaity
of bits 

instead 1/2, it prbalaity densitfiy funtion will define its pprbalaity and in case of awgn it 1/2
to get corect or not corect but with religh model , p robalaity is pdf(x=1) ..relifgh probalaity ensity
functions..

*/



#ifdef LOCAL_COMPILATION_FIXED
	rx_bit = decodedBitsInfo;

	for all stages uIstageBitErrCounter
	{

		for all bin 
		{
		
		 
			  for every bit
			  {
				  
				  if (puBitProbCoeff > .90) 
				  {
                    *pEstimatedBit = *prx_bit ;


				  } else 
				  {
                     *pEstimatedBit = 1- *prx_bit ;
				  }

                  errorsum += (refBit - pEstimatedBit)


			  }  


					++pCorrectbit;
				 ++prx_bit;

                 ++bitCount;
			  
		
		      } 

		 bitCorrectionAlpha[uIstageBitErrCounter] = 1/errorsum;
		
         BaysianProbBitErrCorrection(pEstimatedBit,
								     f_pUCorrectedBitsInfo,
								     uIstageBitErrCounter);
		  ++uIstageBitErrCounter;
         

	}

#endif

}

// this is same as prdication signal..
void Basic5gWithAIExp::BaysianProbBitErrCorrection(unsigned char *f_pEstimatedBit,
												   unsigned char *f_puCorrectedBitsInfo,
												   int            f_uIstageBitErrCounter) 
{

#ifdef LOCAL_COMPILATION_FIXED
	// for the stagse for which this is called , get callsifation factor
	for all bins {

		   for all mode bits 
		   {
				  // summ all staged bits for that bins and mode bit location
                  pCorrectbit += *pEstimatedBit *bitCorrectionAlpha[uIstageBitErrCounter]

		   }
			 


	}
#endif
   
}


/* Part of bit error training , here its same as getting charateridtic or feature vector of channel
 i mean every stages has its own feature vector as bayesioan coeffent based on traing corepeoding specific
  channel charatistic an dfinal during run time based on diff of revd and refrence.. , the stages which
  has least error , that stages is calissified as currnet cahhnel caharetistc,*/
void Basic5gWithAIExp::BayesianProbEstimationQpsk(unsigned int *f_puIlocaldmrs,
										   	      unsigned int *f_puIrxDmrsSig ) 
{

#ifdef LOCAL_COMPILATION_FIXED
    // update P(V)
	/* 00,01,10,11, :
  , if estimaton of probabalit of getting corect bit > threshols, means we getting correct
  bit and alpha[k] = 0, not need to invert bits , and what ever we receved is correct
  and if estimation goes less then threshols then alpha[k] = 1 and means bit is not correct
  and invert the bits. lost bits are in category of wrong bits

  in qpsk model
  B: is event of getting correct bit
  V: is event that observed bit or recved bit is correct,
     ( aftre many trils we get once correct bit.. its computtaions depends on channel models
	 P(v) =  function(channel characteristic) annd measerment.. so if channel is very lossy then what is its value is mathetmaticall derived. 
	 so its all or even in signal trails its all depends on how good is channel charaterisiic . this part still need to be chaceked

  F: is event of getting wrong bit which is 1- P(v)
  P(V/B), is something like ,  we need to updated  based P(B/V)?

  p(B/V),this estimated 
  p(B/V) = p(V/B)*P(B)/( p(V/B)*P(B) + p(V/F)*P(F)

  qpsk and if 10 RE are there so alph[k] where 0 k < 20
  This is designed for only qpsk
  P(B) is probability of getting current bit 1/2 in qpsk, 1/4 4psk
  P(V)  
  p(B/v) =  1/2 *1/2 /(  1/2*1/2 + ( 1-(1/2))*)(1-1/2)

  EstimationpB_V = 1/2*1/2

 */
   
   // for for every stage this basyen coeffient ...as part of transing but calssificatio has to be at runtime , means
	// when we rx signal.. not at traning period.
	// using all stages coeffients 
	unsigned int* puIdemodbitsStageCoeff = &(BaysianProbCoeff + uIstageBitErrCounter);
    
	// This estimation has to increase to 1 when the same iq data is apppreaed.
	for every re bin = 0 bin < len 
	{
		qpskBitValLocal =
        qpskBitValRx
        bitCount = 0;
		while (bitCount < MODE_TYPE) 
		{
			// Prob of V, given B= pV_B, 
          pB_V = pV_B*pB /((pV_B*pB) + (pV_F*(1- pB));


		  bitLocal = (qpskBitValLocal >> bitCount)&0x1;
		  bitRx    = (qpskBitValRx    >> bitCount)&0x1;

		  if( bitLocal ==  bitRx) 
		  {
               // pV_B, THIS prb is something define lossy or erranroius channel charecteritics.
			   // which we want to know.. once this is known then we can compte , given pV, what is PB_V, PRBA OF GETTING CORRECT bit
			   // given the probabality of that receved bit/what we recved is correct one ..considering given channel characteristsic.
			   // so improve in case every time we gett correct bit..during trainng we can have mutiple trail of seding same info bits
			   // dmrs pattren and see every time how does it goes , and if every time its goes well means this channel is cliaasifed as godd one
			   //  so every time propry get updated based on some measure ment results.., here getting bit or measuring for corrcet bit is measurement reports

               pV_B += pV_B*exp(?); // this has to exponenetialy increased or deceres
               pV_F -= pV_F*10/100;


		  } else 
		  {
               //EstimationpB_V += EstimationpB_V*10/100;

			    pV_B -= pV_B*10/100;
                pV_F += pV_F*10/100;

		  }

		 // pEstimatedB_V = pB_V;

		  // for that stgae.. for which its getting trained..
           *puBitProbCoeff = *pB_V;

		   ++puBitProbCoeff;
		}

#endif

}



#endif

#if 0
/* Part of bit error training */
void Basic5gWithAIExp::BayesianProbEstimation4psk(unsigned int *f_puIlocaldmrs,
										   	      unsigned int *f_puIrxDmrsSig ) 
{

    // update P(V)
	/* 00,01,10,11, :
  , if estimaton of probabalit of getting corect bit > threshols, means we getting correct
  bit and alpha[k] = 0, not need to invert bits , and what ever we receved is correct
  and if estimation goes less then threshols then alpha[k] = 1 and means bit is not correct
  and invert the bits. lost bits are in category of wrong bits

  in qpsk model
  B: is event of getting correct bit
  V: is event that observed bit or recved bit is correct
  F: is event of getting wrong bit

  p(B/V),this estimated 
  p(B/V) = p(V/B)*P(B)/( p(V/B)*P(B) + p(V/F)*P(F)

  qpsk and if 10 RE are there so alph[k] where 0 k < 20
  This is designed for only qpsk
  P(B) is probability of getting current bit 1/2
  P(V)  
  p(B/v) =  1/2 *1/2 /(  1/2*1/2 + ( 1-(1/2))*)(1-1/2)

  

 */
       
	for every re bin = 0 bin < len 
	{

       pCoeff




	}



}

#endif




// in time domain.. aftre down conversion and dwonsampling to bas band signal
// channelCoeff[k][i] = dmrs_recved_bin[i]/dmrs_local_bin[i]
// probablity dmrs_recved_bin[i] is same as  dmrs_local_bin[i]
// 

/*  */
#if 0
void 5gBasicWithAIExp::TrainSignalForChannelEstimationBins(unsigned int *f_puIlocaldmrs,
														   unsigned int *f_puIrxDmrsSig )
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
	
     CovMatrixGainPhaseKXM(f_puIlocaldmrs,f_puIrxDmrsSig )  // 
    	
	ChainCoeff_gain[p];
	phaseCoef[p] // so we p phase Coef and p gain coeff

}

void 5gBasicWithAIExp:VarianceGainPhaseKXM(unsigned int *f_puIlocaldmrs,
									          unsigned int *f_puIrxDmrsSig) 
{

	for all bin k = 0 to fft_len 
	{
		localArgs = atan(I/Q);
		rxArgs    = atan(Ir/Qr);
		localGain = 10log10(I*I + Q*Q);
		RxGain    = 10log10(Ir*Ir + Qr*Qr);
		gaindiffDb[k] = localGain - RxGain;  // gain variation

		MeanGain += gaindiffDb[k];
		PhaseShiftDiff[k] = localArgs-rxArgs; 
        meanPhaseShiftDiff += PhaseShiftDiff[k];

	}

     MeanGain = MeanGain/K;

     meanPhaseShiftDiff = meanPhaseShiftDiff/k

	for all bin k = 0 to fft_len 
	{

		GainVar[k] =  gaindiffDb[k] - MeanGain;
        phaseVariance[k] = PhaseShiftDiff[k] - meanPhaseShiftDiff

	}

	// Moving average 
   

	// td is clss varivale provate memeor that track time window
	// complete memmove
    memcpy(&GainVar[td][0], GainVar);

	memcpy(&phaseVariance[td][0], phaseVariance);


}


void 5gBasicWithAIExp::CovMatrixGainPhaseKXM(unsigned int *f_puIlocaldmrs,
											 unsigned int *f_puIrxDmrsSig ) 
{

	for all n < M
	{

		for ( i < M)
		{
	        	CovPhaseMatrix[n][i] = AutoCorelation( &phaseVariance[n][0], &phaseVariance[i][0]);
				CovGainMatrix[n][i] =  AutoCorelation( &GainVariance[n][0], &GainVariance[i][0])

		}
	}

   // U have M*M coravraince Matrix, which is diagonal elemnt is around unity..
  // for Now avagerage all the colums to 1*M and avegere over row and signale coeff.

   // so prodction of dmrs from local is estimation of rx dmrs
   

}






void 5gBasicWithAIExp::TrainSignalForBitCorrectionBins(unsigned int *f_puIModLocalDmrsSignal,
													   unsigned int *f_puIRxDemodSignal) 
{
     // 00 is reved as 00, 10,11,01
	computeQPSKBitErrorCoeffient(uIModLocalDmrsSignal,uIRxDemodSignal) ;

}

#endif

#ifdef BACKUPCODE
void 5gBasicWithAIExp:: RxSignalEstimation() 
{

    // local phase and magnitude

	// rx phase and mag estimation
   
	for all bins 
	{
        // Get IcompLocal;
		// QcompLocal
		// compute phase =  atan(IcompLocal/QcompLocal);
		// compte mag    =  sqrt(IcompLocal*IcompLocal + QcompLocal*QcompLocal);
        // Get IcompRx;
		// QcompRx
		// compute phase =  atan(IcompRx/QcompRx);
		// compte mag    =  sqrt(IcompRx*IcompRx + QcompRx*QcompRx);

        // estimate gain, estimate delay, estimate phase,
          
		// local dmrs, apply above to local and compute final dmrs and say is it matches to recved one.
		ComputeFinalEstimatedSignal(gain, phase, delay, localDmrsSignal, OutDmrsSignal); 
        

		// if this do not matches , update the again the gain estimate, other estimate to make local dmrs same as recbved dmrs
		// once we make the localms same rved dmrs.. , such delay and gain estimates are appliied to reved signal
		UpdateChannelParameter( localDmrsSignal, OutDmrsSignal);

		// and this will revresed to get or extract transmitted signal from recved one,
		// now transmited data could be loossy or corrputed bits,  now..
		// once u get this corrputed bits... , now obtain the bits from each bits.. these bits are obtained with demoduated
		// gets bits of each re, these bits may be corrputed and lost, so how you will make these correct  ..  ?? and what 
		// ...?

		computeBitErrorCoeffient();


        // now we may..accurate mathed this will locally generated dmrs, and we can easily find which re and which bits 
		// have been corrputed and lost.. and apply some compuation to retrievde the lost bits..

        // qpsk modulation







	}
   



}




// re[k][t], so last 1ms  , t is slot or symbol of pdcch or pdsch, which..which is not evrey slot..,if its pdcch,
// then track this Re for next 2 to 3 slot.. only dmrs...
// total power of each RE of the symbol has to be unity.. so every symbol  is translitted with unity gain
// but how this gain/enery is ditributed among RE, is something estimated...PAR to maintain PAR.

// pHASE OF EACH RE must be less then 1/2*subcarrier.., how much , asuming this is tragted for particular psch channel or psch channel.. 
// delay dispersion.. means at any the at any re, phase varies wrt time..
// in freq selcetive, diffrent freq com.. fade difrently, cohernet bw, all freq comp in this bw  have same fade.. 
// estimation and modeling of fade, phase, and again, is bsed on refrence signl dmrsm but the same model is applied to recved signal
// to get signal back from rx coreputed paatren , what has been transmited tx.

void 5gBasicWithAIExp::ChannelEstimation(input iq, estimated_phase)  
{

	for all Re 
	{

       // estimating each RE.. wrt time..
		re[k][t], so estimating I, q, phase and ( tone 2pift, f is subcarrier 15khz)




	}





}

#endif

//the phase roataion at nth subcarrier is phase(m,k,n)  = -1 * cos(2*pi*n*tdk*/Nc) 
//+ j sin(2*pi*n/Nc), so this phase rotation is  directly propotional  to subcarraier index .. so 
// gain is applied considreing Pmax power as sum of power all bin has  < Pmax.
void Basic5gWithAIExp::PhaseEstimation(unsigned int *f_puIRxIQSignal)
{

#ifdef LOCAL_COMPILATION_FIXED
	// phase comliety deend on delay 
	phase =  exp(jf*td)// so for every tone subcarrier idenx
		phas is linear

	// S(0) =    sx(0) + j sy(o)  , where    phase(0) =    tan -1( sy(0)/sx(0)  ,  is transmiited..  
	//, all phase( 0) =  2pi*m/M   ( toal M ary psk element) 		//so all symbol at phase offset of 2pi/M , and this fixed phas offset between symbol has to be fixed 

   // R(0) =  Ri  +J Rq     =     new phase (0) =  tan -1 (Rq/Ri)..    =  phase(0)  + some_estimated phase_noise(0)
   // …   this will impact coherency and inrerfence  …so if we estimate this noise correctly  we can correct this noise or subtract this correct phase noise (0) 
   // from receved signal phase and get exact;y same as transmitted phase as phse(0).



   for each bin 
   {

     localdmrsPhase[k] = 
     rxdrmsphase[k];
	 phasediff[t][k];
     phasediff[t][k];
	 , gain[t][k]

   }


   // compute varinace and mean of rx dmrs .

   // find ratio..    this vairance pattren wont chnages through put symbol period and estimate the same pattren and use to recover 
   // actal signal//.
   
   // wrt time for 1ms, get the varinace patren of phase change, gain, covrance 
   phasediff[t][k] , gain[t][k]

   // cross corelation of varinace pattre at t1, with all other then t1 and cross corelation of variance pattren at t2 with other then t2
   // let supose t1, t2 to tm-1, window length M of vraince tracking.. now withing this window, this covraince pattren of gain, phase  
   // is some sort feature.. of channel behaiour...iT MEANS THIS MUCH PHASE IS GETTING UPDATED TO DMRS signal and same is applicable to other signal other then dmrs
   // and dmrs vector row i*K matrix is mulipled with each  col K*M 
   //corabrane AND FINAL DMRS vector is outdmrs 1*K but each elemet is dived and avrged over M, 

   // this is channel estimation and 


#endif


}


void Basic5gWithAIExp::GainEstimation (unsigned int *f_puIRxIQSignal)  
{

    // auto corrleation of local dmrs,
	// cross relation  delayed local dmrs and recved dmrs 

	//find the ratio of these coefeient , is gain wrt origin signal, idelayy this has to be unitit and in some case
	// it go negative means low signal level and in some cas eit has to be positive means high gain





}





void Basic5gWithAIExp::DelayEstimation(unsigned int *f_puIRxIQSignal)  
{

#ifdef LOCAL_COMPILATION_FIXED
      // get all time domain smaples of recved dmrs.. of currnet symbol at any time t,
      rx_dmrs[n];
	  localdmrs[n]

	  for ( all delay 0 to td) 
	  {

         corrCoef = correlation(rx_dmrs,localdmrs[-td]);
		 if (corrCoef > max_coeff) 
		 {
              max_delay = td;
              
		 }

			
	  }

	  // max delay is the delay and this delay has to less cp length, as impulse reposle
#endif

}

void Basic5gWithAIExp::channelCodingLdpc(int iterationMax )
{
     float lq[MAX_PARITY_CHANNEL_MATRIX_COL*Z_FACTOR];
	 float lr[MAX_PARITY_CHANNEL_MATRIX_COL*Z_FACTOR];
     float DeNormllr[MAX_PARITY_CHANNEL_MATRIX_COL*Z_FACTOR];
	 float sumRow[MAX_PARITY_CHANNEL_MATRIX_COL*Z_FACTOR];
	 char SoftDecsion[MAX_PARITY_CHANNEL_MATRIX_COL*Z_FACTOR];

     for ( int col_i = 0; col_i < MAX_PARITY_CHANNEL_MATRIX_COL*Z_FACTOR; ++col_i) 
	 {
             llr[col_i] =  rand() %128; 
     }


	 for ( int col_i = 0; col_i < MAX_PARITY_CHANNEL_MATRIX_COL*Z_FACTOR; ++col_i) 
	 {
         // denormalize
         DeNormllr[col_i] = llr[col_i];// >> 7;    
#ifndef DISABLE_DEBUG

		 fprintf(Fp,"%d normal llr %d  denormal llr %f \n", col_i, llr[col_i],  DeNormllr[col_i]);
#endif

	 }
	//for all iteration
	for ( int it = 0; it < iterationMax; ++it)
	{

        // get updated llrs
	    // for all col i
		for ( int col_i = 0; col_i < MAX_PARITY_CHANNEL_MATRIX_COL*Z_FACTOR; ++col_i)
	    { 
			// h[j][i]
			//if ( channelMatrix[(row_j%Z_FACTOR)][(col_i%Z_FACTOR)] !=0 && (col_i%Z_FACTOR) !=0) 
			{
				lq[col_i] =   DeNormllr[col_i];
				printf("lq[%d]  %f \n", col_i,lq[col_i]);   
	   
			}
	    }

		float prod;
        double atanhVal;

		
    	//for all j
		for ( int row_j = 0; row_j < MAX_PARITY_CHANNEL_MATRIX_ROW*Z_FACTOR; ++row_j)
		{
		    // for all  colm i
			for ( int col_i = 0; col_i < MAX_PARITY_CHANNEL_MATRIX_ROW*Z_FACTOR; ++col_i)
		    {

               prod = 1.0;
			   if ( channelMatrix[(row_j%Z_FACTOR)][(col_i%Z_FACTOR)] !=0 && (col_i%Z_FACTOR) !=0) 
			   {
               
                   prod =  prod * tanh(0.5*lq[col_i]);

			    }
		   }
          atanhVal = 1.0/(tanh((double)prod)); 
          lr[row_j] =  atanhVal ;
	
		  printf(" %d lr[row_j]  %f \n", row_j, lr[row_j]);
		}

		// update the llr based on this iteration
        float sum = 0.0;       
		//for all j
		for ( int row_j = 0; row_j < MAX_PARITY_CHANNEL_MATRIX_ROW*Z_FACTOR; ++row_j)
		{           
            sum += lr[row_j];
		}


		//for all i 
		for (int col_i = 0; col_i < MAX_PARITY_CHANNEL_MATRIX_COL*Z_FACTOR; ++col_i)
		{
			 DeNormllr[col_i] =  DeNormllr[col_i] + sum; 
			 printf(" updated llrs index %d  %f  \n",col_i,DeNormllr[col_i]);  
			if ( DeNormllr[col_i] < THRESHOLD)
			{
				SoftDecsion[col_i] = 1;
			} else {

				SoftDecsion[col_i] = 0;
			}

			printf(" coli %d SoftDecsion[col_i] %d \n", col_i, SoftDecsion[col_i] );
		}
#if 1
	    sum = 0.0;
		// check the again  pairty....
		//for all j {
		int colmi = 0;
		for ( int row_j = 0; row_j < MAX_PARITY_CHANNEL_MATRIX_ROW*Z_FACTOR; ++row_j) 
		{
			//for all i {
			colmi = 0;
			
			while(colmi < MAX_PARITY_CHANNEL_MATRIX_COL*Z_FACTOR)
			{
				 // parity = 
				 //sum =  c'(i]*h[i][j]; transpose
				 sum  +=  SoftDecsion[colmi]*channelMatrix[colmi][row_j];
				// printf(" col i %d and row %d  soft dec %d channel matrix[%d][%d] = %d  \n ", colmi, row_j, SoftDecsion[colmi], colmi, row_j, channelMatrix[colmi][row_j]);

			     ++colmi;
			}
            sumRow[row_j] = sum;
			sum = 0;
		    printf(" %d   %f \n ", row_j,  sumRow[row_j]);
		}
#endif
		sum = 0.0;
		//for all j {
		for ( int row_j = 0; row_j < MAX_PARITY_CHANNEL_MATRIX_ROW*Z_FACTOR; ++row_j) 
		{
             sum += sumRow[row_j];
		}

 		if ( sum == 0)
		{

			printf("its done , all parity matched to 0 and now  exit\n");
           // its done , all parity matched to 0 and now  exit

		} else 
		{

			printf("go for next iterations \n");
			//  go for next iterations
		}
		 
   }
}










#ifdef NOT_NEEDED
void 5gBasicWithAIExp::ComputeFinalEstimatedSignal(unsigned int *f_pf_puIRxIQSignal,
												   unsigned int *f_puIRxEstimatIQSignal) 
{
	// apply gain to local dmrs signal
	// apply delay to local dmrs signal
	// apply delay to local dmrs signal
    // compute final signal OutDmrsSignal;
	// this outdmrs isgnal is more like recved signal  or estimated recved signal


}


void 5gBasicWithAIExp::UpdateChannelParameter( unsigned int *f_puIRxLocalDmrsIQSignal,
											  unsigned int *f_puIRxDmsrEstimatIQSignal) 
{
	  // compute diff of 
	  error[k] = localDmrsSignal[]-OutDmrsSignal

     // update channel paramters such gain matrix, delay matrix and pashe shift that will be used again in gain, phase and delay estimation block


}

#endif

#ifdef BACKUPCODE
void 5gBasicWithAIExp::ComputeIQ() 
{
     // for given bits
	// real(d(i))=(1/sqrt(2))*(1-2*b(2i) =  0.70710*(1-2*b(2i)  = 0.707 - 1.414*b(2i)
	 //  img(di)   = 1/sqrt(2))*(1-2*b(2i+1)                    = 0.707 - 1.414*b(2i+1)

    
      // say 00, 01, 10, 11, four possibity

	  // so based on these four  4 i and 4 q POSSIBLITIES 

	   // so recevievd I and Q

	  // what is probalaity , THAT recved I is corespdong bits 00, given Probality 00, the probal of 00 IS 1/4, as all are equaly likely

     // find diff Irx - I00, Irx - I01, Irx - I10, Irx - I11,    
     //  which has diff max , will upgrade the probale other will have downgrad

     // say estimate..so for 00,  we have min diff, means its closer to local..so estimate probalaities will increase for these rest 
	 // of other gets 0, so 
	// which means this re HAS INFO BITS 00, with max probalaities and same for other Re
	 
     // Its something like  each element RE, has four weights.. which are probalaites  actually

	 // which one probalaitie above threshols  is considred as 1 weight and rest possiblities have 0 weights.

	// this is bit pattren or info what we reved, but that may not be same is local dmrs pattren..as this reved pattren is 
	// corrupted, delay.. and chnanel impairements..

	// that stardn pattren and add each artificat and find min.. between Rxced and local...    
	// in delay axis, we got SOME pattern dmrs patrent that closly matches the rx pattren
	// in doppler dimention we got some pattren
	// gain axis  .. we got some other apttren..  

    // get final magintute .. of the final pattrens , this will be final output.. 
	// this all is applicable at RE elemnt level ...




}

#endif

#ifdef BACKUPTECHNIQ



void 5gBasicWithAIExp::computeChannelResponseOut(rx_dmrs) 
{
       // fft

	   // extract dmrs

	   ComputeChannelEstimate(rx_dmrs);

	   //it will predict smrs which is transmited witout any noise and bit error
	   PredictChannel(double *rx_dmrs, double *estimatedDmrs) ;

	   // extract RE out AS LLRS and see om crc passes
       compute_llrs( estimatedDmrs, llr);

	   // pass llr to om cores and extract info bits
       computeQPSKBitErrorCoeffient(dmrsModulatedSignal,rx_dmrs) 
	   
	   //As already its trained for bitCorrection

	   if( crc pasees ) 
	   {


	   } else 
	   {

		   // train the system o ai for delay/doppler fading and channel estimation is computed.
		   TrainSignalForChannelEstimationBins(dmrs_Receved_featureVect) ;

		   //Train Bit correction  .
		   TrainSignalForBitCorrectionBins(dmrs_Receved_featureVect);
        
	   }








}

#endif