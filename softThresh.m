function wST = softThresh(w2_x, w2_y, mu)
% Function for computing the soft treshold operator on the TV components.
%
% wST = softThresh(w2_x, w2_y, mu)
% 
%
% INPUT 
% 
% w2_x (dlarray) : x component
% 
% w2_y (dlarray) : y component
% 
% mu (double)    : thresholding parameter
%
% OUTPUT
% 
% wST (dlarray)  : thresholded variable
%
%==========================================================================
%
% Version : 1.0 (22-06-2023)
% Authors : A. Benfenati (alessandro.benfenati@unimi.it)
%           V. Ruggiero  (valeria.ruggiero@unife.it)
%
%==========================================================================

wST{1} = zeros(size(w2_x));
wST{2} = zeros(size(w2_y)); 

norm_w2 = sqrt(w2_x.^2 + w2_y.^2);%+ w2_delta.^2);

ij  = norm_w2 >= mu;
nij = ~ij;

wST{1}(ij) = w2_x(ij) - mu(ij).*w2_x(ij)./norm_w2(ij);
wST{1}(nij)  = 0;

wST{2}(ij) = w2_y(ij) - mu(ij).*w2_y(ij)./norm_w2(ij);
wST{2}(nij)  = 0;

wST{1} = dlarray(wST{1},'SSCB');
wST{2} = dlarray(wST{2},'SSCB');

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