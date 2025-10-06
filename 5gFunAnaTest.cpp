// 5gFunAnaTest.cpp : Defines the entry point for the console application.
//
/* MyMEXFunction
 * c = MyMEXFunction(a,b);
 * Adds offset argument a to each element of double array b and
 * returns the modified array c.
 https://help.octave.narkive.com/LYHNgZlB/call-c-dlls-in-octave
*/

//#include "mex.hpp"
//#include "mexAdapter.hpp"
#if 0
#ifndef MATLAB_CPLUS_CODE
using namespace data;
using mex::ArgumentList;

class Basic5gWithAIExp;

class MexFunction : public mex::Function
{

private:

	 Basic5gWithAIExp  *Ai5gObjPdcch ;
	 Basic5gWithAIExp  *Ai5gObjPdsch ;

	unsigned int uIRxSignal[FFT_LEN];
    unsigned int *puIRxSignal;
    unsigned char uCInfoBits[FFT_LEN*MODE_TYPE];
    unsigned char *puCInfoBits;


public:

	MexFunction()
	{
    int signal = 0;
    const char *myPath = 'D:\EXP_5G_RX_TX_AI\rx5gAIModule\Debug';
    const char *hfile =  'D:\EXP_5G_RX_TX_AI\rx5gAIModule\rx5gAIModule\header\include\AI5gBasic.h';
    loadlibrary([myPath '\rx5gAIModule'],hfile);
    signal=calllib('rx5gAIModule');


    Ai5gObjPdcch = (Basic5gWithAIExp *)new(Basic5gWithAIExp);
	  Ai5gObjPdsch = (Basic5gWithAIExp *)new(Basic5gWithAIExp);

	    puIRxSignal = &uIRxSignal[0];

	    puCInfoBits = &uCInfoBits[0];

	}

	~MexFunction()
	{

      delete Ai5gObjPdcch;
	  delete Ai5gObjPdsch;
	}

    void operator()(ArgumentList outputs, ArgumentList inputs)
	{

		checkArguments(outputs, inputs);
        const double offSet = inputs[0][0];
        TypedArray<double> doubleArray = std::move(inputs[1]);
        for (auto& elem : doubleArray) {
            elem += offSet;
        }
        outputs[0] = doubleArray;

        // test
        Ai5gObjPdcch->computeChannelResponse(puIRxSignal,puCInfoBits);

    }

    void checkArguments(ArgumentList outputs, ArgumentList inputs)
	{
        // Get pointer to engine
        std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

        // Get array factory
        ArrayFactory factory;

        // Check offset argument: First input must be scalar double
        if (inputs[0].getType() != ArrayType::DOUBLE ||
            inputs[0].getType() == ArrayType::COMPLEX_DOUBLE ||
            inputs[0].getNumberOfElements() != 1)
        {
            matlabPtr->feval(u"error",
                0,
                std::vector<Array>({ factory.createScalar("First input must be scalar double") }));
        }

        // Check array argument: Second input must be double array
        if (inputs[1].getType() != ArrayType::DOUBLE ||
            inputs[1].getType() == ArrayType::COMPLEX_DOUBLE)
        {
            matlabPtr->feval(u"error",
                0,
                std::vector<Array>({ factory.createScalar("Input must be double array") }));
        }
        // Check number of outputs
        if (outputs.size() > 1) {
            matlabPtr->feval(u"error",
                0,
                std::vector<Array>({ factory.createScalar("Only one output is returned") }));
        }



    }




};





#endif


#include <octave/oct.h>
DEFUN_DLD (rx5gMexInterfaceClass, args, nargout, "Hello World Help String")
{
    octave_stdout << "Hello World has "
    << args.length () << " input arguments and "
    << nargout << " output arguments.\n";
    // Return empty matrices for any outputs
    octave_value_list retval (nargout);
    for (int i = 0; i < nargout; i++)
    retval(i) = octave_value (Matrix ());
    return retval;
}

#endif


#ifndef ENABLE_MEX_C_INTERFACE



#ifdef ENABLE_MEX_INTERFACE

#include "mex.hpp"
#include "mexAdapter.hpp"

using namespace matlab::data;
using matlab::mex::ArgumentList;
#include "math.h"
#endif

#include "stdafx.h"
#include "AI5gBasic.h"

#include "mex.h"
typedef  unsigned int      UINT32IOT;
typedef  signed int        INT32IOT;
typedef  short             INT16IOT;
typedef  unsigned char     UINT8IOT;
typedef  char              INT8IOT;
typedef  unsigned short    UINT16IOT;
typedef  float             UFLOAT32IOT;


typedef struct __COMPLEX__
{
	short iReal;
	short iImg;
} tcomplex;

typedef struct __COMPLEXF__
{
	float iReal;
	float iImg;
} tcomplexF;


static tcomplexF ModIn[2048]; // 20mhz, fft
static int rxllrCompution(void *CmplxSimbolBuff,   UINT32IOT InLen, UINT8IOT *llrBuffer) ;


static UINT32IOT Qm64BitDeModulationLLRMinMax(tcomplexF *f_pModIn, 
									          UINT32IOT InLen, 
									          UINT8IOT *f_pOutScrambledBit) ;
//#include "mex.h"
#if 1
#define FFT_LEN    2048
#define MODE_TYPE   1
  // create class.
//static Basic5gWithAIExp *Ai5gObjPdcch;
//static Basic5gWithAIExp *Ai5gObjPdsch;
static unsigned int uIRxSignal[FFT_LEN];
static unsigned int *puIRxSignal;
static unsigned char uCInfoBits[FFT_LEN*MODE_TYPE];
static unsigned char *puCInfoBits;


#endif


// https://arma.sourceforge.net/
// C++ library for linear algebra & scientific computing


#include <iostream>
//#include <octave/oct.h>
//#include <octave/octave.h>
//#include <octave/parse.h>
//#include <octave/toplev.h>

using namespace std;

void MexC_CallAbleConfigInit(void)
{

	// test
	printf(" this is 5g functional test in octave/matlab \n");


	//Ai5gObjPdcch = (Basic5gWithAIExp *)new(Basic5gWithAIExp);
	//Ai5gObjPdsch = (Basic5gWithAIExp *)new(Basic5gWithAIExp);

	puIRxSignal = &uIRxSignal[0];

	puCInfoBits = &uCInfoBits[0];



}

// Call from matlab..paasing everytime input.. call per tti
// matrix f_uIRxSignal =  1x
void MexC_CallAbleRunTime(int           f_uISampleLen,
						  unsigned int  *f_uIRxSignal ,
						  unsigned char *f_uCInfoBits,
						  int            f_uIbitCount )
{



   MexC_CallAbleConfigInit();

   // test
   //Ai5gObjPdcch->computeChannelResponse(puIRxSignal,puCInfoBits);

   rxllrCompution(void *CmplxSimbolBuff,   UINT32IOT InLen, UINT8IOT *llrBuffer)
   printf(" computing info bits done \n");
}




// radio init , only once to configure radio parametrs, such carrer gain and noise parametrs

// Matl lab function will call configInit and runtime , something like running under 1ms
// call back that read radio samples of 1ms at parricular carrier
// init will be called once per slot and run time will be call every time 1 ms


/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
	/* variable declarations here */
	int uISampleLen;      /* input scalar */
	unsigned int *uIRxSignal;       /* 1xN input matrix */

	// No of right hand side as input varibale.

#if 0
  if(nrhs != 2) {
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
			"Two inputs required.");
	}


	if(nlhs != 1) {
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs",
			"One output required.");
	}
#endif

#if 0
	float *A = (float *)mxGetData(A IN); /* Get single data */
    signed char *A = (signed char *)mxGetData(A IN); /* Get int8 data */
    short int *A = (short int *)mxGetData(A IN); /* Get int16 data */
    int *A = (int *)mxGetData(A IN); /* Get int32 data */
    int64 T *A = (int64 T *)mxGetData(A IN); /* Get int64 data */
#endif
	/* make sure the first input argument is scalar */

  /*
  if( !mxIsDouble(prhs[0]) ||
		mxIsComplex(prhs[0]) ||
		mxGetNumberOfElements(prhs[0]) != 1 ) {
			mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notScalar",
				"Input multiplier must be a scalar.");
	}
*/
	//int  uISampleLen;           /* size of matrix */

	//unsigned char *uCinfoBits;      /* output matrix */

	/* get the value of the scalar input  */
	//argCIn = mxGetScalar(prhs[0]);

	/* create a pointer to the real data in the input matrix  */
	uIRxSignal = (unsigned int *)mxGetData(prhs[0]);
	//int *A = (int *)mxGetData(A IN); /* Get int32 data */

	/* get dimensions of the input matrix , sample len IS FFT_LEN*/
	uISampleLen = mxGetN(prhs[0]);

	// NCOLS is sample length;


	/* create the output matrix */
	//plhs[0] = mxCreateDoubleMatrix(1,ncols,mxREAL);
    //plhs[0] = mxCreateCharArray(K, (const mwSize *)N)
   // plhs[0] = mxCreateCharArray(1, (const mwSize *)FFT_LEN) ;

	/* get a pointer to the real data in the output matrix
	  M OUT = mxCreateCharArray(2, (const int*)Size);
      Data = mxGetData(M OUT);

	*/
	/* uCinfoBits = (unsigned char *)mxGetData(plhs[0]); */

 // if(nrhs != 2) {
		//mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
		//	"Two inputs required.");
	//}

	/* call the computational routine */
	//arrayProduct(multiplier,inMatrix,outMatrix,ncols);

   unsigned char uCinfoBits[FFT_LEN];
   MexC_CallAbleRunTime(uISampleLen,uIRxSignal , &uCinfoBits[0],FFT_LEN );

	/* code here */
}



#if 1

UINT32IOT NewIFFT(UINT32IOT N, UINT32IOT *x, tcomplexF *DFT)
{
    UINT32IOT  stageCount = 0;
	UINT32IOT  bsep  = 0;
	UINT32IOT  bwidth = 0;
	UINT32IOT  botval = 0;
	UINT32IOT  topval = 0;

	tcomplexF *pDFT = DFT;

	double weighting_factor = 0;
	double rFactor = 0;
	double theta = 0;
	tcomplexF WN ;

	tcomplexF X[2048];

	tcomplexF temp;
       
    stageCount = 10;//

    //m , evluate stages 

    for (int stage = 1; stage <= stageCount; ++stage) // to stage = m ) 	 
	 {         
           bsep = (UINT32IOT)pow(2.0,(double)stage);    // for currnet stage

		   weighting_factor = (double)N/bsep;
		   bwidth = (double)bsep/2;

		   for (int j = 0; j < bwidth - 1; ++j) 
		   {

                 rFactor = weighting_factor *j;
 				 theta = -1*(2 * PI *rFactor)/N;
				 //WN =  CMPLX(cos(theta),-sin(theta));
                  WN.iReal = cos(theta);
				  WN.iImg  = -1.0*sin(theta);

				 for (topval = j; topval < N/2; topval+= bsep) 
				 {

                       botval = topval + bwidth;

					   //X[botval].iImg  = 0;
					  (pDFT + topval)->iImg = 0;
					  (pDFT + topval)->iReal = *(x + topval);
                       //temp = X(botval)x WN;


					   temp.iReal = *(x + botval)* WN.iReal;
					   temp.iImg  = *(x + botval)* WN.iImg;


					   //X(botval) = X(topval)- temp;
                       (pDFT + botval)->iImg  = (pDFT + topval)->iImg  - temp.iImg;
					   (pDFT + botval)->iReal = (pDFT + topval)->iReal - temp.iReal;


					   //X(topval) = X(topval)+temp;

					   (pDFT + topval)->iImg  = (pDFT + topval)->iImg  + temp.iImg;
					   (pDFT + topval)->iReal = (pDFT + topval)->iReal + temp.iReal;
 
                       //X(botval) = X(botval)/N;
					   (pDFT + botval)->iImg  = (pDFT + botval)->iImg/N;
					   (pDFT + botval)->iReal = (pDFT + botval)->iReal/N;

     				    //X(topval) = X(topval)/N;
					   (pDFT + topval)->iImg  =  (pDFT + topval)->iImg/N;
					   (pDFT + topval)->iReal =  (pDFT + topval)->iReal/N;



			     }


			 }

	}

     


   return 0;

}


// stage m = 10
UINT32IOT bitReverse(UINT32IOT N, UINT32IOT *x,UINT32IOT *reverse_x, UINT32IOT m) 
{

    UINT32IOT naddr;
	UINT32IOT iaddr;
	UINT32IOT rmdr;


     for (UINT32IOT k = 0;  k < N-1; ++k) 
	 {

		 naddr  = 0;
		 iaddr = k;
		 for (UINT32IOT i = 0; i < m-1; ++i) 
		 {
			rmdr = MOD(iaddr,2);
			naddr = naddr + rmdr *((double)pow(2.0,(double)(m-1-i)));
			iaddr = iaddr/2;

		 }

		 reverse_x[naddr] = x[k];
	 }

	 return 0;


}




#endif

#if 1

// Add any function to be test......

int rxllrCompution(void *CmplxSimbolBuff,   UINT32IOT InLen, UINT8IOT *llrBuffer) 
{
    UINT32IOT  outLen;
	UINT32IOT i = 0;

	// InLen no of modulates samples  per Symbol in slot.
	while( i < InLen) 
	{
       ModIn[i].iReal = -0.707;
	   ModIn[i].iImg  = 0.707;
	   ++i;

	   ModIn[i].iReal = 0.707;
	   ModIn[i].iImg  = 0.707;
	   ++i;

	   ModIn[i].iReal = 0.707;
	   ModIn[i].iImg  = -0.707;
	   ++i;

	   ModIn[i].iReal = -0.707;
	   ModIn[i].iImg  = -0.707;
	   ++i;


	}
    
    outLen =  Qm64BitDeModulationLLRMinMax(&ModIn[0], 
									        InLen, 
					  			         llrBuffer) ;




    return outLen;
}



// min max llr , ITS SOMETHING FININDING CLOSET DEMOD BITS FOR THAT RECVED SYMBOL,

UINT32IOT Qm64BitDeModulationLLRMinMax(tcomplexF *f_pModIn, 
									   UINT32IOT InLen, 
									   UINT8IOT *f_pOutScrambledBit) 
{

     unsigned char a0[1][6];
	 unsigned char a1[1][6];

	 unsigned char InPhase_a;
	 unsigned char QadPhase_a;

    

	// 16 qam, m = 4, p = 2
       
	//un_i[p] = 
    // m = 2p,
	// m =  6 bits, 2*p = 6, p= 3 bits 

	// a + jb, inphase a 
    // inphase
    //int p = 3;
	// 64 qam, 6 bits
	a0[0][0]  = 0;
	a0[0][1]  = 0;
	a0[0][2]   = 0;
	a0[0][3]   = 0;
	a0[0][4]   = 0;
	a0[0][5]   = 0;

	a1[0][0]  = 1;
	a1[0][1]  = 1;
	a1[0][2]   = 1;
	a1[0][3]   = 1;
	a1[0][4]   = 1;
	a1[0][5]   = 1;

	//sigma = 1; // max variance is (+-1), sqr of (+-1) = 1;


	int p = 0;
	short diff_0;
	short diff_1;
	short variance_0;
	short variance_1;
	char sigma  = 1;
	unsigned int sumVar0 = 0;
	unsigned int sumVar1 = 0;
	float llr[8];

	for ( int n = 0; n < InLen; ++n) 
	{
		InPhase_a = f_pModIn[n].iReal;
		QadPhase_a = f_pModIn[n].iImg;

		p = 3;
		// p = 3 , say inpahse_a 101
		for ( int i = 0; i < p; ++i) 
		{
			// could be 0 or 1 bits
			for ( int j = 0; j < 2*p; ++j) 
			{

				//diff_0 = InPhase_a - a0[i][j];
                diff_0 = InPhase_a - a0[0][j]; // for now i = 0;, all have 0 only
				diff_0 = diff_0*diff_0;
				variance_0 = diff_0/(2*sigma);
				sumVar0 += variance_0;

				diff_1 = InPhase_a - a1[i][j];
				diff_1 = diff_1*diff_1;
				variance_1 = diff_1/2*sigma;
				sumVar1 += variance_1;


			}

			llr[i] = (float)log10((float)sumVar0/sumVar1);

			if ( i%20==0) 
			{
               printf("\n");
			}
			printf(" llr %d   ", llr[i]);

			sumVar0 = 0;
			sumVar1 = 0;

		}

		sumVar0 = 0;
		sumVar1 = 0;

		p = 3;
		// p = 3 , say Qadphase_a 101
		for (int i = 0; i < p; ++i) 
		{
			//un[i] // could be 0 or 1 bits
			for (int  j = 0; j < 2*p; ++j) 
			{

				diff_0 = QadPhase_a - InPhase_a*a0[i][j];
				diff_0 = diff_0*diff_0;
				variance_0 = diff_0/2*sigma;
				sumVar0 += variance_0;

				diff_1 = QadPhase_a - InPhase_a*a1[i][j];
				diff_1 = diff_1*diff_1;
				variance_1 = diff_1/2*sigma;
				sumVar1 += variance_1;


			}

			llr[i] = (float)log10((float)sumVar0/sumVar1);

			sumVar0 = 0;
			sumVar1 = 0;

		}

	}

	return 0;
}

#endif










// mkoctfile -I D:\EXP_5G_RX_TX_AI\rx5gAIModule\rx5gAIModule\header\include -L D:\EXP_5G_RX_TX_AI\rx5gAIModule\Debug  -lrx5gAIModule --mex -v file rx5gMexInterfaceClass.cpp


