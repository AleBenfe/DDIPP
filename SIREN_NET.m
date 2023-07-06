function lgraph = SIREN_NET(num_input_channel,varargin)
%
% Matlab function for creating the SIREN neural network for
% learning the psf. 
%
%   lgraph = prova_fcn(K,N, num_input_channel)
%
% Reference paper: 
%  Sitzmann V, Martel J N P, Bergman A W, Lindell D B and
%  Wetzstein G, 2020, Implicit neural representations with periodic
%  activation functions CoRR
% 
% The initialization of this network is done by following the suggestion of
% the auhors of the original paper.
%
% MANDATORY INPUT
%
% num_input_channel (integer): number of input values.
%
% OPTIONAL INPUT
%
% K (integer)       : dimension of the psf, assuming a square shape K x K.
%                     Default: 21;
%
% N (integer)       : dimension of the fully connected layers. Default:
%                     2000.
%
% nLayers (integer) : number of (FC+sin) layer couples. Default: 4.
%
% OUTPUT
%
% lgraph (dlnetwork): dlnetwork object for custom training.
%==========================================================================
%
% Version : 1.0 (22-06-2023) 
% Authors : A. Benfenati (alessandro.benfenati@unimi.it),
%           A. Catozzi (ambra.catozzi@unife.it)
%
%==========================================================================

% Default Values
N       = 2000;
K       = 21;
nLayers = 4;

if (nargin-length(varargin)) ~= 1
    error('Wrong number of required parameters');
end

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'N'
                N = varargin{i+1};
            case 'K'
                K = varargin{i+1};
            case 'NLAYERS'
                nLayers = varargin{i+1};
            otherwise
                error('Unkkown option.')

        end
    end
end

c  = 6;
w0 = 30;
layers = [featureInputLayer(num_input_channel,'Name','input')];
n = num_input_channel;
m = N;
for i = 1:nLayers
    % The initialization follows the suggestion of the original paper
    W = -1+2*rand(m,n);
    W = w0*sqrt(c/N)*W;

    layers = [layers
    fullyConnectedLayer(N,'Name',sprintf('FC-%d',i), ...
                        Weights = W);
    functionLayer(@(x) sin(x), ...
                  'Name',sprintf('Sin-%d',i), ...
                  'Description',"Sin Activation Layer")];
    n  = m;
    w0 = 1;
end

layers = [layers
          fullyConnectedLayer(K^2)
          softmaxLayer('Name','output_softmax')
           ];

lgraph = dlnetwork(layers);

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
