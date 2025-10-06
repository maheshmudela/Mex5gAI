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

#if 1
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



   MexC_CallAbleConfigInit();

   // test
   Ai5gObjPdcch->computeChannelResponse(puIRxSignal,puCInfoBits);

   printf(" computing info bits done \n");
}

#endif



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


void arrayProduct(double x, double *y, double *z, mwSize n)
{
  mwSize i;

  for (i=0; i<n; i++) {
    z[i] = x * y[i];
  }
}




#endif





// mkoctfile -I D:\EXP_5G_RX_TX_AI\rx5gAIModule\rx5gAIModule\header\include -L D:\EXP_5G_RX_TX_AI\rx5gAIModule\Debug  -lrx5gAIModule --mex -v file rx5gMexInterfaceClass.cpp

