function Net = NetInit(Net,Input_init,sigma,X,T,type,neteps)
% Matlab function for the initial training of the nets for the DDIPP
% appraoch.
%
% Net = NetInit_Image(Net,Input_init,sigma,X,T,type,neteps)
%
% INPUT 
% 
% Net (dlnetwork) : network to be trained
% 
% Input_init (dlarray) : input to the network
% 
% sigma (double)        : standard deviation of the input random variable of DIP
% 
% X (dlarray)   : target for the training
% 
% T (integer) : maximum numbero of iteration for the training
% 
% type (string) : choice for the net type - kernel or image.
% 
% neteps (double) : tolerance for the stopping criterion
%
% OUTPUT
%
% net (dlnetwork) : trained net
%
%==========================================================================
%
% Version : 1.0 (22-06-2023)
% Author  : A. Benfenati (alessandro.benfenati@unimi.it)
%
%==========================================================================
switch upper(type)
    case 'IMAGE'
        dltype = 'SSCB';
    case 'KERNEL'
        dltype = 'CB';
    otherwise
        error('Unknown type.')
end

averageGrad    = [];
averageSqGrad  = [];

fprintf('Initialising %s Net...\n',type)
ti          = 1;
flag        = 1*(T>0);
inputSize   = size(Input_init);
TIloss      = zeros(T,1);

while flag
    fprintf('%4d/%4d\n',ti,T);
    % Draw input realization
    input = Input_init + sigma*randn(inputSize);
    input = dlarray(input,dltype);
    % Compute Gradients
    [TIloss(ti),gradients] = dlfeval(@lossTrainNet,Net,input,X);
    % Update net's weights
    [Net,averageGrad,averageSqGrad] = adamupdate(Net,...
        gradients,...
        averageGrad,...
        averageSqGrad,...
        ti);
    % Check flag
    flag = TIloss(ti)>TIloss(1)*neteps && ti<T;

    ti = ti +1;  
end
fprintf('Done.\n')
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