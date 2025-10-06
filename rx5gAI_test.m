
mkoctfile  -v -I"E:/maheshData/EXP_5G_RX_TX_AI/rx5gAIModule/rx5gAIModule/header/include"  -I"E:/maheshData/EXP_5G_RX_TX_AI/rx5gAIModule/rx5gAIModule/source" -o1 -pg -c E:/maheshData/EXP_5G_RX_TX_AI/rx5gAIModule/rx5gAIModule/source/AI5gBasic.cpp -o1 -pg -c E:/maheshData/EXP_5G_RX_TX_AI/rx5gAIModule/rx5gAIModule/source/CovarianceStageTraining.cpp -o1 -pg -c E:/maheshData/EXP_5G_RX_TX_AI/rx5gAIModule/rx5gAIModule/source/lmsStageTraining.cpp -o1 -pg -c E:/maheshData/EXP_5G_RX_TX_AI/rx5gAIModule/rx5gAIModule/source/stdafx.cpp
%mkoctfile   -v -I"D:/EXP_5G_RX_TX_AI/rx5gAIModule/rx5gAIModule/header/include"  -I"D:/EXP_5G_RX_TX_AI/rx5gAIModule/rx5gAIModule/source"  -o1 -pg -c D:/EXP_5G_RX_TX_AI/rx5gAIModule/rx5gAIModule/source/AI5gBasic.cpp -o1 -pg -c D:/EXP_5G_RX_TX_AI/rx5gAIModule/rx5gAIModule/source/CovarianceStageTraining.cpp -o1 -pg -c D:/EXP_5G_RX_TX_AI/rx5gAIModule/rx5gAIModule/source/lmsStageTraining.cpp -o1 -pg -c D:/EXP_5G_RX_TX_AI/rx5gAIModule/rx5gAIModule/source/stdafx.cpp


%mex  -I"D:/EXP_5G_RX_TX_AI/rx5gAIModule/rx5gAIModule/header/include"  -I"D:/EXP_5G_RX_TX_AI/rx5gAIModule/rx5gAIModule/source" -v -m64 -pg --mex -DMATLAB_MEX_FILE  D:/EXP_5G_RX_TX_AI/rx5gAIModule/Main5gAiApp/source/Receiver5GWithAI.cpp AI5gBasic.o  CovarianceStageTraining.o lmsStageTraining.o  stdafx.o
mex  -I"E:/maheshData/EXP_5G_RX_TX_AI/rx5gAIModule/rx5gAIModule/header/include"  -I"E:/maheshData/EXP_5G_RX_TX_AI/rx5gAIModule/rx5gAIModule/source" -v -m64 -pg --mex -DMATLAB_MEX_FILE  E:/maheshData/EXP_5G_RX_TX_AI/rx5gAIModule/Main5gAiApp/rx5gMexInterfaceClass.cpp AI5gBasic.o  CovarianceStageTraining.o lmsStageTraining.o  stdafx.o


fid = fopen('cs_mxg_dump.bin', 'r');
%% Read the data to a variable
data = fread(fid, '*int16');
%% close the file after reading the data
fclose(fid);

%%pnsesq
%%iff
%% delay, channel artifacts
%%fft pn
%%fft

%%
cplxdata= complex(data(1:2:end),data(2:2:end));
%% fft 2048

%[A B] =
 rx5gMexInterfaceClass(data);
%% it check for mex gateway function in above file

%%rx5gMexInterfaceClass

figure;plot(abs(cplxdata));


