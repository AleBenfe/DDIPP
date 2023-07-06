function lgraph = DDIPP_UNET(inputSize,varargin)
%
% Matlab function for creating an Unet (AE) for image reconstruction.
%
% lgraph = asUnet(inputSize,varargin)
%
% This function creates a Deep Image Prior network:
% structure from Ulyanov paper
%
% MANDATORY INPUT
%
% inputSize (double array) : size of the input images.
%
% OPTIONAL INPUT
%
% depthE (integer)          : Number of Encoder's depth. Default: 4;
%
% innerDepthE (integer)     : number of (convolutional + ReLU) blocks in
% each
%                           encoder level
%
% depthD (integer)          : Number of Dencoder's depth. Default: 4;
%
% innerDepthD (integer)     : number of (convolutional + ReLU) blocks in
%                             each decoder level. Default: 2
%
% nrfilt (integer)          : number of filters in the first block. It will
%                             be doubled in subsequent levels. Deafult: 8
%
% custom (bool)             : flag for custom training. Default: 1
%
% OUTPUT
%
% lgraph                    : Neural network ready for being trained. If
%                             custom is TRUE, then it is a dlnetwork
%                             object, otherwise it is a layergraph object.
%
%==========================================================================
%
% Version : 1.0 (22-06-2023)
% Authors : A. Benfenati (alessandro.benfenati@unimi.it)
%           A. Catozzi (ambra.catozzi@unife.it), 
%
%==========================================================================

% Default Values
depthE      = 5;
depthD      = 5;
encFilts    = 2.^(3:7);
decFilts    = 2.^(7:-1:3);
custom      = 1;

if (nargin-length(varargin)) ~= 1
    error('Wrong number of required parameters');
end

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'DEPTHE'
                depthE = varargin{i+1};
            case 'DEPTHD'
                depthD = varargin{i+1};
            case 'ENCFILTS'
                encFilts = varargin{i+1};
            case 'DECFILTS'
                decFilts = varargin{i+1};
            case 'CUSTOM'
                custom = varargin{i+1};
        end
    end
end

[depthE,depthD] = errorManagement(depthE,depthD,encFilts,decFilts);

layers = imageInputLayer(inputSize,'Normalization','none','Name','Image-Input');

%% Encoder
fprintf('Creating Encoder...')

for i = 1:depthE
    % Conv - BN - LeakyRelu
    layers = [layers;
        convolution2dLayer(3,encFilts(i),...
        'Stride',[1 1],...
        'Padding','same',...
        'Name',sprintf('EncStage-%d-Conv-1',i));

        convolution2dLayer(3,encFilts(i),...
        'Stride',[2 2],...
        'Padding','same',...
        'Name',sprintf('EncStage-%d-decimation',i));

        batchNormalizationLayer('Name',sprintf('EncStage-%d-BN-1',i));

        leakyReluLayer(0.2,'Name',sprintf('EncStage-%d-LeakyReLU-1',i));

        convolution2dLayer(3,encFilts(i),...
        'Stride',[1 1],...
        'Padding','same',...
        'Name',sprintf('EncStage-%d-Conv-2',i));

        batchNormalizationLayer('Name',sprintf('EncStage-%d-BN-2',i));

        leakyReluLayer(0.2,'Name',sprintf('EncStage-%d-LeakyReLU-2',i))];
end
fprintf('\tDone.\n')

%% Decoder
fprintf('Creating Decoder...')

for i = depthD:-1:1
    layers = [layers;

    depthConcatenationLayer(2,'Name',sprintf('DecStage-%d-DepConc',i));

    batchNormalizationLayer('Name',sprintf('DecStage-%d-BN-0',i));

    convolution2dLayer(3,decFilts(i),...
    'Stride',[1 1],...
    'Padding','same','Name',sprintf('DecStage-%d-Conv-1',i));

    batchNormalizationLayer('Name',sprintf('DecStage-%d-BN-1',i));

    leakyReluLayer(0.2,'Name',sprintf('DecStage-%d-LeakyReLU-1',i));

    convolution2dLayer(3,decFilts(i),...
    'Stride',[1 1],...
    'Padding','same','Name',sprintf('DecStage-%d-Conv-2',i));

    batchNormalizationLayer('Name',sprintf('DecStage-%d-BN-2',i));

    leakyReluLayer(0.2,'Name',sprintf('DecStage-%d-LeakyReLU-2',i));

    % Interpolation works better
    resize2dLayer('Name',sprintf('DecStage-%d-Up',i), ...
        'Scale',2, ...
        'Method','bilinear')];

end

lgraph = layerGraph(layers);

for i = 1:depthE
    Skip_conn = [convolution2dLayer(3,16,...
        'Stride',[1 1],...
        'Padding','same', ...
        'Name',sprintf('Skip-%d-Conv',i));
        batchNormalizationLayer('Name',sprintf('Skip-%d-BN',i));
        leakyReluLayer('Name',sprintf('Skip-%d-LeakyReLU',i))];

    lgraph = addLayers(lgraph,Skip_conn);

    lgraph = connectLayers(lgraph,sprintf('EncStage-%d-LeakyReLU-2',i) ,sprintf('Skip-%d-Conv',i));
    lgraph = connectLayers(lgraph,sprintf('Skip-%d-LeakyReLU',i),sprintf('DecStage-%d-DepConc/in2',i));
end

if custom
    lgraph = addLayers(lgraph,sigmoidLayer("Name",'Sigmoid'));
    lgraph = addLayers(lgraph,convolution2dLayer(1,1,'Stride',[1,1],...
                'Padding','same','Name','Final-Conv'));
    lgraph = connectLayers(lgraph,'DecStage-1-Up','Final-Conv');

    lgraph = connectLayers(lgraph,'Final-Conv','Sigmoid');
else
    lgraph = addLayers(lgraph,regressionLayer('Name','RegressionLayer'));
    lgraph = connectLayers(lgraph,'Sigmoid','RegressionLayer');
end

fprintf('\tDone.\nThe Neural Network is ready.\n')

if custom
    lgraph= dlnetwork(lgraph);
end
end

function [depthE,depthD] = errorManagement(depthE,depthD,nrEncFilts,nrDecFilts)
%
% [depthE,depthD] = ERRORMANAGEMENT(depthE,depthD,nrEncFilts,nrDecFilts)
%
% Function to check the correspondence between encoder/decoder dimension
% and filters numbers. When the depths of the encoder and of the decoder
% differ, they are set to the minimum between the two.

e = 0;
if depthE~=depthD
    mnm = min(depthE,depthD);
    fprintf('Encoder depths and Decoder depths differ: they are set to %d\n',mnm);
    depthE = mnm;
    depthD = mnm;
end

if length(nrEncFilts)~=depthE
    e = 1;
end
if length(nrDecFilts)~=depthD
    e = 2;
end

printError(e)
end

function printError(e)
%
% PRINTERRORS()e
%
% Function for printing error messages.
switch e
    case 1
        error('Depth of the Encoder does not correspond to the list of filter dimensions');
    case 2
        error('Depth of the Decoder does not correspond to the list of filter dimensions');
    otherwise
        % nothing to do
end
end

%==========================================================================
%
% COPYRIGHT NOTIFICATION
%
% Permission to copy and modify this software and its documentation for
% internal research use is granted, provided that this notice is retained
% thereon and on all copies or modifications. The authors and their
% respective Universities makes no representations as to the suitability
% and operability of this software for any purpose. It is provided "as is"
% without express or implied warranty. Use of this software for commercial
% purposes is expressly prohibited without contacting the authors.
%
% This program is free software; you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% Free Software Foundation; either version 3 of the License, or (at your
% option) any later version.
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
% Public License for more details.
%
% You should have received a copy of the GNU General Public License along
% with this program; if not, either visite http://www.gnu.org/licenses/ or
% write to Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA
% 02139, USA.
%
%==========================================================================
