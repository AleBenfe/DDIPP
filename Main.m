% Matlab script for applying the method described in 
% A Benfenati, A Catozzi and V Ruggiero, Neural blind deconvolution with
% Poisson data, Inverse Problems 39 054003 DOI 10.1088/1361-6420/acc2e0
%
%==========================================================================
%
% Version  : 1.0 (22-07-2022)
% Authors  : A. Benfenati (alessandro.benfenati@unimi.it)
%            A. Catozzi (ambra.catozzi@unife.it)
%            V. Ruggier (valeria.ruggiero@unife.it)
%
%==========================================================================

clearvars
close all
clc

%% Visualization & Save options
savingIter = 2;
visualIter = 3;

%% Choose the image file
imagName = 'rice.mat'; % or 'micro.mat', 'synth001.mat'
% Load data
[obj,gn,psf,bg,estH,scaling,rho] = loadData(imagName,'NOISELEV',1);
% Create saving directory for the metrics and the trained networks
dirName = 'Results';
if ~isfolder(sprintf('%s/%s',dirName,imagName(1:end-4)))
    mkdir(sprintf('%s/%s',dirName,imagName(1:end-4)))
end

%% Setting up the Networks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Random Kernel input size
kernelInputSize  = [200,1];                      
% Realization of the input for the kernel net
kernelInput_init = -1 + 2*rand(kernelInputSize); 
% Size of the estimated kernel
kernelEstDim     = size(estH);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convolution network 
% The convolution for the blurring operator is
% implemented as a neural network in order to trace the gradient
% computation
%
% The random input for the image network has the spatial dimension of the
% given data and a number of channel set by the user
imageInputSize   = [size(gn,1:2),32];
fprintf('Setting convolution network...')
layers = [imageInputLayer(imageInputSize(1:2),...
    'Normalization','none',...
    'Name','Image-Input');
    convolution2dLayer(kernelEstDim(1),1,...
    'stride',[1,1], ...
    'Padding','same',...
    'PaddingValue','symmetric-exclude-edge',...
    'Name','convolution')];

convNet = dlnetwork(layers);
% Sets the weights of the network, i.e. the convolution, with the initial
% estimate
convNet.Learnables(1,3).Value{1} = dlarray(estH,'SSCB');
fprintf('Done.\n')

% Perturbations for the random inputs
sigmaI = 1e-3;
sigmaK = 0;

% Iteration numbers for
% a) kernelNet initialisation
% b) imageNet initialisation
% c) main cycle
TK = 200;
TI = 200;
TT = 1000;

% Early Stopping Settings
W        = 10;
P        = 20;
varVec   = zeros(prod(size(gn,1:2)),W);
varmin   = Inf;
eStopTol = 1e-3;
neteps   = 0.001;

% Initialise Kernel Net
kernelNet = SIREN_NET(200,'K',kernelEstDim(1),'N',500);

estH      = projectDF(zeros(size(estH)),rho*ones(size(estH)), ...
                      1,estH,ones(size(estH)),'VERB',0);
estH      = dlarray(single(estH(:)),'CB');
kernelNet = NetInit(kernelNet,kernelInput_init,sigmaK, ...
                          estH,TK,'Kernel',neteps);
Gh        = forward(kernelNet, dlarray(single(kernelInput_init(:)),'CB'));
V         = projectDF(zeros(size(estH)),rho*ones(size(estH)), ...
                      1,Gh,ones(size(estH)),'VERB',1);

% Initialise Image Net
imageNet = DDIPP_UNET(imageInputSize,'depthE',5,'depthD',5,...
    'encFilts',2.^(3:1:7), ...
    'decFilts',2.^(7:-1:3));

imageInput_init = rand(imageInputSize);
X               = gn;
Y               = TV_dlarrayPGDA(X);
imageNet        = NetInit(imageNet,imageInput_init,sigmaI, ...
                                X,TI,'Image',neteps);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Loss fun
lossFun = @lagrangian_TotVar;

% Dual variables
muy{1} = dlarray(zeros(size(X)),'SSCB');
muy{2} = dlarray(zeros(size(X)),'SSCB');
muv    = dlarray(zeros(size(estH(:))),'CB');

% PGDA parameters
gammay = 1;
gammav = .001;
ap     = 1e-6;
ad     = 1e-5;
delta  = 1e-4;

% Adam Parameters
averageIGrad   = [];
averageISqGrad = [];
averageKGrad   = [];
averageKSqGrad = [];

loss    = zeros(TT,1);
ssimVec = zeros(TT+1,1);
snrVec  = zeros(TT,2);
psnrVec = zeros(TT,2);

ssimVec(1) = ssim(obj,gn);

% Save the parameters on settings.txt file
fid = fopen(sprintf('Results/%s/settings.txt',imagName(1:end-4)),'w');
fprintf(fid,'gamma_y : %4.4e\n',gammay);
fprintf(fid,'gamma_v : %4.4e\n',gammav);
fprintf(fid,'a_p : %4.4e\n',ap);
fprintf(fid,'a_d : %4.4e\n',ad);
fprintf(fid,'TK : %d\n',TK);
fprintf(fid,'TI : %d\n',TI);
fprintf(fid,'TT : %d\n',TT);
fprintf(fid,'sigmaI: %d\n', sigmaI);
fprintf(fid,'sigmaK: %d\n\n', sigmaK);
fprintf(fid,'W: %d\n', W);
fprintf(fid,'P: %d\n', P);
fprintf(fid,'patienceTol: %e\n\n',eStopTol );
[psnrini,snini]=psnr(gn,obj);
fprintf(fid,'psnr(gn)=%g snr(gn)=%g ssim(gn)=%g \n', psnrini,snini,ssim(gn,obj));
fprintf(fid,'rho: %f\n\n',rho );
fclose(fid);

% Creation of the file for the performance measures 
fid = fopen(sprintf('Results/%s/metrics.txt',imagName(1:end-4)),'w');
fprintf(fid,'SSIM,SNRX,PSNRX,SNRK,PSNRK,KL,rho,nnz,LL,maxvar\n');

iter = 1;
flag = 1;
flagWrite = 1;
figure(99)
imagesc(extractdata(gn))
while flag

    % Perturbation of input for image network
    imageInput  = imageInput_init + sigmaI*randn(imageInputSize);
    imageInput = dlarray(imageInput,'SSCB');

    % Perturbation of input for kernel network
    kernelInput = kernelInput_init + sigmaK*randn(kernelInputSize);
    kernelInput = dlarray(kernelInput,'CB');

    % Gradients computation
    [loss(iter),gradientsKernel,gradientsImage,gradientsV,convNet] = dlfeval(lossFun, ...
        imageNet,kernelNet,convNet,...
        imageInput,kernelInput, ...
        V,Y, ...
        gammav,gammay,...
        muv,muy,...
        gn,bg,delta);

    % Kernel net update
    [kernelNet,averageKGrad,averageKSqGrad] = adamupdate(kernelNet,...
        gradientsKernel,...
        averageKGrad,...
        averageKSqGrad,...
        iter, ...
        ap);

    % Image net update
    [imageNet,averageIGrad,averageISqGrad] = adamupdate(imageNet,...
        gradientsImage,...
        averageIGrad,...
        averageISqGrad,...
        iter, ...
        ap);

    Gx = forward(imageNet,imageInput);
    Gh = forward(kernelNet,kernelInput);

    % Early Stopping
    if iter <=W
        varVec(:,iter) = extractdata(Gx(:));
    else
        varVec(:,1:W-1) = varVec(:,2:W);
        varVec(:,W) = extractdata(Gx(:));
    end

    if iter >=W
        temp = varVec - mean(varVec,2);
        VAR(iter-W+1) = 1/W*sum(vecnorm(temp).^2);
        if VAR(iter-W+1)<min(varmin)
            varmin = [varmin, VAR(iter-W+1)];
        end
    end

    if flagWrite && length(varmin)>P
        discr=2*loss(iter)*scaling/numel(gn);
        if discr<2 || (discr<5 && loss(iter)>loss(iter-1)+0.1*numel(gn)/(2*scaling))

            contFlag = any(abs(diff(varmin(end-P+1:end)))>eStopTol);
            if (~contFlag || (discr<5 && loss(iter)>loss(iter-1)+0.1*numel(gn)/(2*scaling)))
                fid2 = fopen(sprintf('Results/%s/settings.txt',imagName(1:end-4)),'a');
                fprintf(fid2,'Stopping iter: %d contFlag=%d\n',iter,contFlag);
                fclose(fid2);
                flagWrite = 0;
            end
        end
    end

    flag = flag & iter<TT;

    % Update for V
    vInput = V - ap*(gradientsV);
    V = projectDF(zeros(size(vInput)),... 
        rho*ones(size(vInput)), ... 
        1, ... 
        vInput, ... 
        ones(size(vInput))); 

    [psnrVec(iter,1),snrVec(iter,1)] = psnr(Gx,obj);

    [psnrVec(iter,2), snrVec(iter,2)]  = psnr(V,psf(:));

    ssimVec(iter) = ssim(Gx,obj);

    % Update for Y
    AX    = TV_dlarrayPGDA(Gx);
    % Adaptive regularization parameter
    betaI = 1/(2*prod(size(gn,1:2)))* loss(iter)./sqrt(AX{1}.^2+AX{2}.^2+delta^2);

    gradientsY{1}   = gammay*( Y{1} - AX{1} )  + muy{1};
    gradientsY{2}   = gammay*( Y{2} - AX{2} )  + muy{2};

    Y = softThresh(Y{1}-ap*gradientsY{1}, ...
        Y{2}-ap*gradientsY{2}, ...
        ap*betaI);

    muy{1} = muy{1} + ad*(Y{1}-AX{1});
    muy{2} = muy{2} + ad*(Y{2}-AX{2});
    muv    = muv    + ad*(V-Gh);

    fprintf(['%04d) ssim: %2.2f - SNRX: %2.2f - PSNR(X): %2.2f - SNRK: %2.2f -  PSNR(K): %2.2f' ...
        ' - KL: %f - rho: %f - nz: %d\n'], ...
        iter,ssimVec(iter),snrVec(iter,1),psnrVec(iter,1), snrVec(iter,2),psnrVec(iter,2), ...
        2*loss(iter)*scaling/numel(gn), ...
        max(V(:)),nnz(extractdata(V)));

    fprintf(fid,'%2.2f,%2.2f,%2.2f,%2.2f,%2.2f, %f,%f,%d\n', ...
        ssimVec(iter),snrVec(iter,1),psnrVec(iter,1),snrVec(iter,2),psnrVec(iter,2), ...
        2*loss(iter)*scaling/numel(gn), ...
        max(V(:)),nnz(extractdata(V)));

    if mod(iter,visualIter)==0
        onlineVisualization(Gx,V.reshape(kernelEstDim),...
            gn,psf)
    end

    if mod(iter,savingIter)==0
        save(sprintf('Results/%s/image_%d',...
            imagName(1:end-4), iter),'Gx');
        temp = V.reshape(kernelEstDim);
        save(sprintf('Results/%s/ker_%d',...
            imagName(1:end-4), iter),'temp');

        imwrite(extractdata(Gx),sprintf('Results/%s/image_%04d.png',...
            imagName(1:end-4), iter));
        imwrite(uint8(255*(extractdata(V.reshape(kernelEstDim)))), ...
            sprintf('Results/%s/ker_%04d.png',...
            imagName(1:end-4), iter));
    end

    iter = iter +1;
end

psnrVec(iter:end,:) = [];
loss(iter:end)      = [];

% Saving results.
fprintf('Saving inputs...')
save(sprintf('Results/%s/randomInputs',imagName(1:end-4)),'kernelInput_init','imageInput_init');
fprintf('\tDone.\n')
fprintf('Saving nets...')
save(sprintf('Results/%s/imageNet',imagName(1:end-4)),'imageNet');
save(sprintf('Results/%s/kernelNet',imagName(1:end-4)),'kernelNet');
fprintf('\tDone.\n')
fclose(fid);
