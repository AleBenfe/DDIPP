function [loss,gradients] = lossTrainNet(net,In,X)
% Loss function for initial setting of the net
%
% [loss,gradients] = lossTrainNet(kerNet,In,K0)
% 
% It trains the net using the MSE loss
%
% INPUT
% 
% net (dlnetwork) : network to be trained
%
% In (dlarray)      : input to the net
%
% X (dlarray)       : target for the training
%
% OUTPUT
% 
% loss (dlarray)    : loss value
%
% gradients (table) : gradients wrt the weights
%
%==========================================================================
%
% Version : 1.0 (22-06-2023) 
% Authors : A. Benfenati (alessandro.benfenati@unimi.it),
%
%==========================================================================

% Forward data through network.
[Y,~] = forward(net,In);

% Calculate cross-entropy loss.
loss = mse(Y,X);

% Calculate gradients of loss with respect to learnable parameters.
gradients  = dlgradient(loss,net.Learnables);

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