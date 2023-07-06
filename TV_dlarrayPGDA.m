function [Y] = TV_dlarrayPGDA(X)
% Function for computing thegradient component for employing TV in the PGDA
% algorith for DDIPP approach.
%
% [Y] = TV_dlarrayPGDA(X)
% 
% INPUT
% 
% X (dlarray) : input image
%
% OUTPUT
%
% Y (dlarray) : dx and dy componenets of the gradient
%
%==========================================================================
%
% Version : 1.0 (22-06-2023) 
% Authors : A. Benfenati (alessandro.benfenati@unimi.it),
%
%==========================================================================

obj_size = size(X);
X = extractdata(X);
Y{1} = [diff(X,1,2), X(:,1)-X(:,obj_size(2))];
Y{2} = [diff(X)    ; X(1,:)-X(obj_size(1),:)];

Y{1} = dlarray(Y{1},'SSCB');
Y{2} = dlarray(Y{2},'SSCB'); 

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

