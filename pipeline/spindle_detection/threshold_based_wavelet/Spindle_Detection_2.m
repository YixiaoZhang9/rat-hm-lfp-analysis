function [SpindleAll, SpindleAvg, Parameter,NumSpin] = Spindle_Detection_2(StageData,sub,stage,numepoch,srate,varargin)
% Reference：
% 1. A better way to define and describe Morlet wavelets for time-frequency analysis
% 2. Macro and micro sleep architecture and cognitive performance in older adults
% 3. Characterizing sleep spindles in 11,630 individuals from the National Sleep Research Resource

%% inputs:
%** requered inputs**
%   Data: subjects*numepoch*timepoints
%   sub : number
    %the input data is from which subject
%   %numepoch : number
    %the total number of epochs for the input data
%   %srate: sampling frequency(in HZ) 

%** Optional inputs**
%   'PeakFrequency': The peak frequency of the wavelet (in Hz). Default = 13.5

%   'WaveletBandWidth': Based on the reference article 1, the bandwidth is
%    defined alternatively as the full-width at half-maximum (FWHM),
%    which is the distance in time between 50% gain before the peak to 50%
%    gain after the peak. . Default = 5 (spindle:11-15HZ)

%   'DurationCoreMin': The minimal duration of spindle core. Default = 0.3

%   'DuraionMin'：The minimal duraion of detected spindle(including core and
%    waning region). Default = 0.5

%   'DurationMax': The maximal duration of spindle. Default = 3

%   'DurationInterval': The interval of adjacent spindles which less than
%   this criteria whill be merged

%   'AmplitudeCoreCriteria': Amplification factor the signal core must exceed to be
%    classified as a spindle. Default = 6

%   'AmplitudeCriteria': Amplification factor the signal must exceed to be
%    classified as a spindle. Default = 3

%   'ThresholdType': Use either the 'median' or the 'mean for defining the
%   threshold. Default = 'median'

%   [[[explaination: see the supplymetary of paper "Macro and micro sleep
%   architecture and cognitive performance in older adults"
%    Defining thresholds as a multiplicative function of the individual’s
%    median wavelet power was more robust compared to using the mean, given
%    that wavelet power had a very skewed distribution and apparent
%    outliers (which in large part reflect true spindles)
%    disproportionately influenced the mean but not the median.]]]

%   'Plot':Plot the wavelet and detected spindle. Default = 0

%   'Stage': detect spindle in which sleep stage.you can choose N2 or N3.The input 
%   parameter is the sequence of sleep stage:'Wake','N1','N2','N3','REM'.Default = 3
%% outputs
%** requered outputs**
%   'SpindleALL': the spindle parameters in each epoch
%   'SpindleAve': the average value of spindle parameters

%   Author:ZhangYixiao
%   Finish Date:     2023-04-23  
%%
% parameter setting 
% parameter setting 
PeakFreq = 13.5;
LowFreq = 11;
HighFreq = 15;
BandWidth = 5;
tThresh1 = 0.3;
tThresh2 = 0.5;
tThresh3 = 3;
tThresh4 = 0.5;
AmplThresh1 = 6;
AmplThresh2 = 3;
method = 'median';
fPlot = 0;

if find(strcmpi(varargin, 'PeakFrequency'))
    PeakFreq = varargin{find(strcmpi(varargin, 'PeakFrequency'))+1};
end
% strcmpi:compare strings or character vectors ignoring case
% TF = strcmpi(S1,S2) compares S1 and S2 and returns logical 1 (true)
% if they are the same except for case, and returns logical 0 (false) 
if find(strcmpi(varargin, 'WaveletBandWidth'))
    BandWidth = varargin{find(strcmpi(varargin, 'WaveletBandWidth'))+1};
end
if find(strcmpi(varargin, 'DurationCoreMin'))
    tThresh1 = varargin{find(strcmpi(varargin,'DurationCoreMin'))+1};
end
if find(strcmpi(varargin, 'DuraionMin'))
    tThresh2 = varargin{find(strcmpi(varargin,'DuraionMin'))+1};
end
if find(strcmpi(varargin, 'DurationMax'))
    tThresh3 = varargin{find(strcmpi(varargin,'DurationMax'))+1};
end
if find(strcmpi(varargin, 'DurationInterval'))
    tThresh4 = varargin{find(strcmpi(varargin,'DurationInterval'))+1};
end
if find(strcmpi(varargin, 'AmplitudeCoreCriteria'))
    AmplThresh1 = varargin{find(strcmpi(varargin, 'AmplitudeCoreCriteria'))+1};
end
if find(strcmpi(varargin, 'AmplitudeCriteria'))
    AmplThresh2 = varargin{find(strcmpi(varargin, 'AmplitudeCriteria'))+1};
end
if find(strcmpi(varargin, 'ThresholdType'))
    method = varargin{find(strcmpi(varargin, 'ThresholdType'))+1};
end
if find(strcmpi(varargin, 'Plot'))
    fPlot = varargin{find(strcmpi(varargin, 'Plot'))+1};
end
% print 
fprintf(['Detecting sleep spindles on central channels for the subject',32,num2str(sub),'\n\n'])

%% step 1: Construct wavelet: Form frequency domain
% This method was propsed in reference paper 1 and in it's supplimentary,
% the author also provide corresponding matlab code

npoints = 8001;
% vector of frequencies
hz = linspace(0,srate,npoints);

% Frequency domain Gaussian
s  = BandWidth*(2*pi-1)/(4*pi); % normalized width
x  = hz - PeakFreq; % shifted frequency
fx = exp(-.5*(x/s).^2); % gaussian

% Complex Morlet Wavelet in time domain
Morletwavelet = fftshift(ifft(fx));
% fftshift-Shift zero-frequency component to center of spectrum.
% ifft-Inverse discrete Fourier transform
wavetime = (-floor(npoints /2):floor(npoints /2))/srate;% Time vector

if fPlot==1
    % empirical FWHM in HZ
    idx = dsearchn(hz',PeakFreq);
    empFWHM =  hz(idx-1+dsearchn(fx(idx:end)',.5)) - hz(dsearchn(fx(1:idx)',.5));
    
    figure;
    subplot(311)
    plot(hz,fx,'linew',2)
    set(gca,'XLim',[0 PeakFreq*3])
    xlabel('Frequency (Hz)');ylabel('Amplitude (gain)')
    title(['Bandwidth specified: ' num2str(BandWidth) ' Hz, obtained: ' num2str(empFWHM) ' Hz'])
    
    subplot(312), hold on
    plot(wavetime,real(Morletwavelet),'linew',2)
    plot(wavetime,imag(Morletwavelet),'--','linew',2)
    h = plot(wavetime,abs(Morletwavelet),'linew',2);
    set(h,'color','m')
    set(gca,'xlim',[-1 1])
    legend({'Real part';'Imag part';'Envelope'})
    xlabel('Time (sec.)')
    
end
%% step 2: Convolve and Compute wavelet coefficients
% Prepare convolution parameters
nWavelet = length(wavetime);
nData = 30*srate; % the data is segmented into 30-s epoch
nConv = nWavelet+nData-1;
halfW = ceil(nWavelet/2); % ceil：Round towards plus infinity

% Convolve
for i = 1:numepoch(sub,stage)    %the number of stage N2
    signal = squeeze(StageData{stage}(sub,i,:));
    % Spectrum of data
    dataX = fft(signal',nConv);
    % Spectrum of morlet
    waveX = fft(Morletwavelet,nConv);
    waveX = waveX./max(waveX); % normalize
    % Convolve
    WavCoef = ifft(waveX.*dataX);
    % (Remember: To transpose complex matrices you need to use .' to not conjugate them)
    % Cut the coefficient
    WavCoef = WavCoef(1,halfW:(end-halfW)+1);
%     if fPlot == 1
%         subplot(313)
%         coef2plot = WavCoef; 
%         %% Check here the amplitude response of your wavelet
%         fig.x = (1:1:nData)/srate;
%         fig.xTable = timetable(seconds(fig.x)',coef2plot');
%         [fig.pxx,fig.f] = pspectrum(fig.xTable);
%         % Analyze signals in the frequency and time-frequency domains
%         plot(fig.f,pow2db(fig.pxx))
%         set(gca,'XLim',[0 srate/2])
%         xlabel('Frequency (Hz)')
%         ylabel('Power Spectrum (dB)')
%         title('FFT of wavelet coefficents (EEG*wavelet)')
%     end
    coef_temp = abs(WavCoef).^2;
    Coef(i,:) = coef_temp;
    clear coef_temp
    clear WavCoef
    % this matrix store the wavelet coefficient of each epoch
end
%% step 3: smooth coefficient and calculate the average value for this subject
% smooth the wavelet coefficients using a moving average (window duration 0.1 s).
for j = 1:numepoch(sub,stage)
    coef_temp = Coef(j,:);
    winlength = ceil(srate*0.1);% window duration is 0.1s
    % method 1
    % coef = smoothdata(coef_temp,'movmean',winlength);  
    
    % method 2
     window = ones(winlength,1)/winlength;
    % Take the moving average using the above window
     coef = filtfilt(window,1,coef_temp);
     % filtfilt: perform Zero-phase digital filtering by forward and reverse
     % processing of input data x
     SmoothCoef(j,:) = coef;         
%      figure
%      subplot(211)
%      plot(coef_temp,'linew',1);
%      subplot(212)
%      plot(coef,'linew',1);
end

% Compute Threshold
switch method
    case 'median'
        Threshold = median(SmoothCoef(:)); 
    case 'mean'
        Threshold = sum(SmoothCoef(:))/numepoch(sub,stage)/nData;
end

% Check for threshold
if Threshold < 1e-30
    warning(['The threshold for spindle detection in Subject',num2str(sub),'seems to abnormal'])
end
%% step 4: detect spindles
NumSpin = zeros(numepoch(sub,stage),1);
% the number of spindles in each epoch
for j = 1:numepoch(sub,stage)
    numSpin = 0; 
    currentData = SmoothCoef(j,:);
    Over = currentData>AmplThresh2*Threshold; % Mark all points over AmplThresh2*Threshold as logical '1'
    CoreOver = currentData>AmplThresh1*Threshold; 
    WholeMark = double(Over); % convert the logical variable to double 
    CoreMark = double(CoreOver);
    Mark = WholeMark+CoreMark;
    % data over AmplThresh1*Threshold is marked 2 while data over AmplThresh2*Threshold
    % is marked 1.
    
    % find the location of spindle core 
    Deriv_Core = diff([0 CoreMark 0 ]);
    Index_CoreStrat = find(Deriv_Core==1);
    Index_CoreEnd = (find(Deriv_Core==-1))-1;
    
    % find the location of spindle 
    Deriv_Spin = diff([0 WholeMark 0 ]);
    Index_Strat = find(Deriv_Spin==1); 
    Index_End = (find(Deriv_Spin==-1))-1;
      
    % check the duration of spindle core and spindle
    for k = 1:length(Index_CoreStrat)
        Dura_Core_temp = (Index_CoreEnd(k)-Index_CoreStrat(k))/srate;
        if (Dura_Core_temp>=tThresh1&&Dura_Core_temp<=tThresh3)
            Spin_start_temp = max(Index_Strat(find((Index_CoreStrat(k)-Index_Strat)>=0)));
            %Index_SpinStrat_temp = find(Index_Strat == Spin_start_temp);
            % find the index of start location of spindle that containing this core  
            Spin_end_temp = min(Index_End(find((Index_End-Index_CoreEnd(k)>=0))));
            %Index_SpinEnd_temp = find(Index_End == Spin_end_temp);
            % find the index of end location of spindle that containing this core  
            Dura_Spin_temp = ((Spin_end_temp-Spin_start_temp))/srate;
            if (Dura_Spin_temp>=tThresh2&&Dura_Spin_temp<=tThresh3)
                numSpin = numSpin+1;
                Index_SpinStart(j,numSpin) = Spin_start_temp ;
                Index_SpinEnd(j,numSpin) = Spin_end_temp;
                if numSpin>1
                    if Index_SpinStart(j,numSpin)==Index_SpinStart(j,numSpin-1)
                        Index_SpinStart(j,numSpin) = 0;
                        Index_SpinEnd(j,numSpin) = 0;
                        numSpin = numSpin-1;
                    end
                    % there may be more than one spindle core between one
                    % start and end point that define spindles over AmplThresh2*Threshold
                end
                % check the detected spindle again
                if sum(WholeMark(Index_SpinStart(j,numSpin):Index_SpinEnd(j,numSpin)))~=Spin_end_temp-Spin_start_temp+1
                    warning(['The detedted Spindle in subject',32,num2str(sub),32,'epoch',32,num2str(j),32,'is wrong'])
                end
            end
        end
    end  
    if numSpin ==0 %there is no spindles in this epoch
       Index_SpinStart(j,:) = 0;
       Index_SpinEnd(j,:) = 0;
    end
    NumSpin(j) = numSpin;    
end
%% Step 5: Discard or Merge detected spindles that meet criteria
% Spindles within 0.5 seconds of each other were merged, unless the above 
% mentioned 3.0 second criterion would be violated, in which case they were both excluded.

% merge the detected spindles(may be more than two spindles are nearby)
for j = 1:numepoch(sub,stage)
    k = 1;
    while NumSpin(j)
        if k< NumSpin(j)
            Dura_interval_temp = (Index_SpinStart(j,k+1)-Index_SpinEnd(j,k))/srate;
            if Dura_interval_temp<=tThresh4
                Index_SpinStart(j,k+1) = Index_SpinStart(j,k);
                Index_SpinStart(j,k:end) = circshift(Index_SpinStart(j,k:end),-1,2);
                Index_SpinEnd(j,k:end) = circshift(Index_SpinEnd(j,k:end),-1,2);
                NumSpin(j) = NumSpin(j)-1;
                Index_SpinStart(j,NumSpin(j)+1:end) = 0;
                Index_SpinEnd(j,NumSpin(j)+1:end) = 0;
            else
                k = k+1;
            end
        else 
            break;
        end
    end
end
% delete spindles after merging which the duraion exceed tThresh3 
for j = 1:numepoch(sub,stage)
    k = 1;
    while NumSpin(j)
        if k<NumSpin(j)
            Dura_Merged_temp = (Index_SpinEnd(j,k)-Index_SpinStart(j,k))/srate;
            if Dura_Merged_temp>tThresh3
                Index_SpinStart(j,k:end) = circshift(Index_SpinStart(j,k:end),-1,2);
                Index_SpinEnd(j,k:end) = circshift(Index_SpinEnd(j,k:end),-1,2);
                NumSpin(j) = NumSpin(j)-1;
                Index_SpinStart(j,NumSpin(j)+1:end) = 0;
                Index_SpinEnd(j,NumSpin(j)+1:end) = 0;
            else
                k = k+1;
            end
        else
            break;
        end
    end
end
%% Step 6: Calculate spindle paramters
% Initialize the spindle parameter structure
Parameter = {'Density','Duration','PeakFreq','SigmaPower','PeaktoPeak','Numofcycles','Symmetry','ISA'};
SpindleAll.Start = [];
SpindleAll.End = [];
SpindleAll.Duration = [];
SpindleAll.PeakFreq = [];
SpindleAll.SigmaPower = [];
SpindleAll.PeaktoPeak = [];
SpindleAll.Numofcycles = [];
SpindleAll.Symmetry = [];
SpindleAll.ISA = [];
% parameters
for j = 1:numepoch(sub,stage) %loop through epoch
       SpindleAll.Start{j,1} = Index_SpinStart(j,:);
       SpindleAll.End{j,1} = Index_SpinEnd(j,:);
       SpindleAll.Duration{j,1} = Index_SpinStart(j,:);
       SpindleAll.PeakFreq{j,1} = Index_SpinStart(j,:);
       SpindleAll.SigmaPower{j,1} = Index_SpinStart(j,:);
       SpindleAll.PeaktoPeak{j,1} = Index_SpinStart(j,:);
       SpindleAll.Numofcycles{j,1} = Index_SpinStart(j,:);
       SpindleAll.Symmetry{j,1} = Index_SpinStart(j,:);
       SpindleAll.ISA{j,1} = Index_SpinStart(j,:);
       if NumSpin(j)>0
           for k =1:NumSpin(j)
               Duration_temp = (Index_SpinEnd(j,k)-Index_SpinStart(j,k))/srate;
               signal = squeeze(StageData{stage}(sub,j,:));
               [Parameter,Parameter_Name] = Spindle_Features(signal,srate,LowFreq,HighFreq,Index_SpinStart(j,k),Index_SpinEnd(j,k));
               % the matrix Parameter include 'PeakFrequency','SigmaPower','PeaktoPeak'
               % 'NumberofCycles','Symmetry','Integrated Spindle Activity'
               SpindleAll.Duration{j,1}(k) = Duration_temp;
               SpindleAll.PeakFreq{j,1}(k) = Parameter(1);
               SpindleAll.SigmaPower{j,1}(k) = Parameter(2);
               SpindleAll.PeaktoPeak{j,1}(k) = Parameter(3);
               SpindleAll.Numofcycles{j,1}(k) = Parameter(4);
               SpindleAll.Symmetry{j,1}(k) = Parameter(5);
               SpindleAll.ISA{j,1}(k) = Parameter(6);
           end
       end
end
% averaged parameters
TotalNumofSpin = sum(NumSpin);
SpindleAvg.Density = [];
SpindleAvg.Duration = [];
SpindleAvg.PeakFreq = [];
SpindleAvg.SigmaPower = [];
SpindleAvg.PeaktoPeak = [];
SpindleAvg.Numofcycles = [];
SpindleAvg.Symmetry = [];
SpindleAvg.ISA = [];
field = fieldnames(SpindleAvg); % cell
SpindleAvg.Density = TotalNumofSpin/(numepoch(sub,stage)*0.5);
for m = 2:length(field)
    name_m = field{m};
    value_temp = SpindleAll.(name_m); %the whole value for this parameter
    sumofspin = 0;
    for n = 1:length(value_temp)
        sumofspin = sumofspin+sum(value_temp{n});
    end
    SpindleAvg.(name_m) = sumofspin/TotalNumofSpin;
end

%% step7 plot the detected Spindles
fplot = 0;
if fplot==1
    for i = 1:numepoch(sub,stage)
        signal = squeeze(StageData{stage}(sub,i,:));
        t = (0:1/srate:30-1/srate);
        height_rectangle = max(SpindleAll.PeaktoPeak{i,1})+20;
        % the height of rectangle 
        if NumSpin(i,1)
            figure;
            plot(t,signal)
            xlim=get(gca,'Xlim');
            hold on
            plot(xlim,[0,0],'k-','LineWidth',1,'linestyle','--')
            for r = 1:NumSpin(i,1)
                x1 = SpindleAll.Start{i,1}(r)/srate;
                x2 = SpindleAll.End{i,1}(r)/srate;
                y1 = -1/2*height_rectangle;
                y2 = +1/2*height_rectangle;
                rectx = [x1 x2 x2 x1 x1];
                recty = [y1 y1 y2 y2 y1];
                plot(rectx,recty,'linewidth',2)
            end
            set(gca,'YAxisLocation','origin','Box','off','LineWidth',1.5,'FontSize',10);
            xlabel('Time/s','FontSize',15);
            ylabel('Amplitude/uv','FontSize',15)
            title('Detected Spindles','FontSize',20);
        end
    end
end
return