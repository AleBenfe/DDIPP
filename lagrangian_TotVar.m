function [loss,gradientsKernel,gradientsImage,gradientsV,convNet] = lagrangian_TotVar(imageNet,kernelNet,convNet,...
    imageIn,kernelIn, ...
    V,Y, ...
    gammav,gammay,...
    muv,muy,...
    gn,b,delta)
%
% [loss,gradientsKernel,gradientsImage,temp,K] = LOSSKERIMAG(imageNet,
% kernelNet,convNet,imageIn,kernelIn,gn)
%
% Everything is computed in this function in order to keep trace of
% everything to properly compute the gradient with the backpropagation.
% Previous tests showed that the same steps done otuside the function lead
% to zero gradients.
%
% INPUT
% 
% imageNet (dlnetwork)  : neural network for image reconstruction
%
% kernelNet (dlnetwork) : neural network for psf reconstruction
%
% convNet (dlnetwork) : neural network for convolution operation
%
% imageIn (dlarray)     : input for the image neural network
%
% kernelIn (dlarray)    : input for the kernel neural network
%
% V (dlarray)           : psf double variable 
%
% Y (dlarray)           : image double variable 
%
% gammav (double)       : dual psf parameter 
%
% gammay (double)       : dual image parameter
%
% muv (dlarray)         : Lagrangian psf parameter
%
% muy (dlarray)         : Lagrangian image parameter
%
% gn (dlarray)          : corrupted image
%
% b (double)            : background emission
%
% delta (double)        : TV parameter
% 
% OUTPUT
%
% loss (dlarray)            : value of the loss wrt the current generated
%                             image and the current kernel.
%
% gradientsKernel (table)   : gradient wrt to the kernel net parameters.
%
% gradientsImage (table)    : gradient wrt to the image net parameters.
% 
% gradientsV (table)        : gradient wrt to the dual variable V.
%
% convNet (dlnetwork)       : updated convolution network
%
%==========================================================================
%
% Version : 1.0 (22-06-2023)
% Author  : A. Benfenati (alessandro.benfenati@unimi.it)
%
%==========================================================================

% Forward data through kernel network.
Gh = forward(kernelNet,kernelIn); %G(theta_h)
%Gh = projectDF(zeros(size(Gh)),rho*ones(size(Gh)),1,V,ones(size(Gh)));
Gh = Gh.reshape(sqrt(size(Gh,1)),sqrt(size(Gh,1)));

% Forward data through image network.
Gx = forward(imageNet,imageIn); %G(theta_x)
convNet.Learnables(1,3).Value{1} = dlarray(V.reshape(sqrt(size(V,1)),sqrt(size(V,1))),'SSCB');

% Calculate KL.
KX = forward(convNet,Gx);
nonzero =  gn~=0;
KL = sum(gn(nonzero).*log(gn(nonzero)./(KX(nonzero)+b))+KX(nonzero)+b-gn(nonzero))+...
    sum(KX(~nonzero) + b);

% Calculate losses.
lossThetaH = gammav*mse(dlarray(V,'CB'),Gh(:))+dot(muv,V-Gh(:));

AX = TV_dlarrayPGDA(Gx);
lossThetaX = KL + gammay*mse(Y{1}+muy{1}/gammay, AX{1}) + ...
                  gammay*mse(Y{2}+muy{2}/gammay, AX{2}) ;


lossV = KL + sum(dlarray((1/(2*numel(V))*KL./(V.^2+delta^2)).*V.^2,'CB'))+...
                lossThetaH;

% Calculate gradients of loss with respect to learnable parameters.
gradientsKernel = dlgradient(lossThetaH,kernelNet.Learnables);
gradientsImage  = dlgradient(lossThetaX,imageNet.Learnables);
gradientsV      = dlgradient(lossV,V);


loss = KL; 

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
