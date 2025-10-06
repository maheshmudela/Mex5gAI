// Receiver5GWithAI.cpp : Defines the entry point for the console application.
//

#ifdef ENABLE_MEX_INTERFACE

#include "mex.hpp"
#include "mexAdapter.hpp"

using namespace matlab::data;
using matlab::mex::ArgumentList;
#include "math.h"
#endif

#include "stdafx.h"
#include "AI5gBasic.h"


#if 1

  // create class.
static Basic5gWithAIExp *Ai5gObjPdcch;
static Basic5gWithAIExp *Ai5gObjPdsch;
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
	printf(" this is pdcch/slot configuration \n");

   
	Ai5gObjPdcch = (Basic5gWithAIExp *)new(Basic5gWithAIExp);
	Ai5gObjPdsch = (Basic5gWithAIExp *)new(Basic5gWithAIExp);

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

   // test
   Ai5gObjPdcch->computeChannelResponse(puIRxSignal,puCInfoBits);

}


int _tmain(int argc, _TCHAR* argv[])
{

#if 0

  string_vector argv (2);

argv(0) = "embedded";
argv(1) = "-q";
octave_main (2, argv.c_str_vec (), 1);
octave_idx_type n = 2;
octave_value_list in;

for (octave_idx_type i = 0; i < n; i++)
in(i) = octave_value (5 * (i + 2));
octave_value_list out = feval ("gcd", in, 1);
if (! error_state && out.length () > 0)
std::cout << "GCD of ["
<< in(0).int_value ()
<< ", "
<< in(1).int_value ()
<< "] is " << out(0).int_value ()
<< std::endl;
else
std::cout << "invalid\n";
clean_up_and_exit (0);

#endif

    // create class.
	
	MexC_CallAbleConfigInit();
	

	// From dl rfi we get Symbol
	// fft 
    // call matlab funtion of fft


	// for every slot, configure  slot for pdcch/pdsch and srs..
	while (1) 
	{

         // Any slot, first symbol is pdcch , decoding pdcch , indicates dci formate and 
		// read file data of slot 1  and sleep for 1 ms, as slot interval in 1 ms.
		

		//Ai5gObjPdcch->configureInitpdcch(dmrsType,scramblingNID0, scramblingNID1,rnti,dci_formate );

		Ai5gObjPdcch->computeChannelResponse(puIRxSignal,puCInfoBits);

        // pdcch decode will payload bts which deicde abous pdsch and other in slot



	}
	return 0;
}











#ifdef ENABLE_MEX_C_INTERFACE

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
	if(nrhs != 2) {
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
			"Two inputs required.");
	}
	if(nlhs != 1) {
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs",
			"One output required.");
	}
#if 0
	float *A = (float *)mxGetData(A IN); /* Get single data */
    signed char *A = (signed char *)mxGetData(A IN); /* Get int8 data */
    short int *A = (short int *)mxGetData(A IN); /* Get int16 data */
    int *A = (int *)mxGetData(A IN); /* Get int32 data */
    int64 T *A = (int64 T *)mxGetData(A IN); /* Get int64 data */
#endif
	/* make sure the first input argument is scalar */
	if( !mxIsDouble(prhs[0]) ||
		mxIsComplex(prhs[0]) ||
		mxGetNumberOfElements(prhs[0]) != 1 ) {
			mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notScalar",
				"Input multiplier must be a scalar.");
	}

	mwSize uISampleLen;           /* size of matrix */

	unsigned char *uCinfoBits;      /* output matrix */

	/* get the value of the scalar input  */
	//argCIn = mxGetScalar(prhs[0]);

	/* create a pointer to the real data in the input matrix  */
	uIRxSignal = (unsigned int *)mxGet(prhs[0]);
	//int *A = (int *)mxGetData(A IN); /* Get int32 data */

	/* get dimensions of the input matrix , sample len IS FFT_LEN*/
	uISampleLen = mxGetN(prhs[0]);

	// NCOLS is sample length;


	/* create the output matrix */
	//plhs[0] = mxCreateDoubleMatrix(1,ncols,mxREAL);
   // plhs[0] = mxCreateCharArray(K, (const mwSize *)N) 
     plhs[0] = mxCreateCharArray(1, (const mwSize *uISampleLen)) ;

	/* get a pointer to the real data in the output matrix
	  M OUT = mxCreateCharArray(2, (const int*)Size);
      Data = mxGetData(M OUT);
	
	*/
	uCinfoBits = mxGetData(plhs[0]);


	/* call the computational routine */
	//arrayProduct(multiplier,inMatrix,outMatrix,ncols);

	MexC_CallAbleRunTime(uISampleLen,uIRxSignal , uCinfoBits,uISampleLen );

	/* code here */
}


void arrayProduct(double x, double *y, double *z, mwSize n)
{
  mwSize i;

  for (i=0; i<n; i++) {
    z[i] = x * y[i];
  }
}




#endif


#if 0

/*
-------------------------------------
function [resultScalar, resultString, resultMatrix] = exampleOctaveFunction (inScalar, inString, inMatrix)

  resultScalar = (inScalar * pi);
  resultString = strcat ('Good morning Mr. ', inString);
  resultMatrix = (inMatrix + 1);

endfunction
-------------------------------------

I have a file called how-to-call-octave.cpp which is this:

-------------------------------------
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>
#include <octave/toplev.h> /* do_octave_atexit */

int main (const int argc, char ** argv)
{
const char * argvv [] = {"" /* name of program, not relevant */, "--silent"};

  octave_main (2, (char **) argvv, true /* embedded */);

  octave_value_list functionArguments;

  functionArguments (0) = 2;
  functionArguments (1) = "D. Humble";

  Matrix inMatrix (2, 3);

  inMatrix (0, 0) = 10;
  inMatrix (0, 1) = 9;
  inMatrix (0, 2) = 8;
  inMatrix (1, 0) = 7;
  inMatrix (1, 1) = 6;
  functionArguments (2) = inMatrix;

const octave_value_list result = feval ("exampleOctaveFunction", functionArguments, 1);

  std::cout << "resultScalar is " << result (0).scalar_value () << std::endl;
  std::cout << "resultString is " << result (1).string_value () << std::endl;
  std::cout << "resultMatrix is\n" << result (2).matrix_value ();

  do_octave_atexit ();
}

*/

#endif



