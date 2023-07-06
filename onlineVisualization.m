function onlineVisualization(X,K,O,H)
% Function for online visualization of the results, comparing with the
% ground truth when available.
%
% onlineVisualization(X,K,obj,trueH)
%
% INPUT
%
% X (dlarray)       : current image iterate
% K (dlarray)       : current kernel iterate
% O (dlarray)       : image for comparison
% H (dlarray)       : kernel for comparison
%
%==========================================================================
%
% Version : 1.0 (22-06-2023)
% Authors : A. Benfenati (alessandro.benfenati@unimi.it)
%
%==========================================================================

figure(42) % For being independent from other figure object

% Current image iterate
subplot(221)
imagesc(extractdata(X),[0,1])
title('Gx'), colorbar, axis image

% Current psf iterate
subplot(222)
imagesc(extractdata(K))
title('Gh'), colorbar, axis image

% Ground truth (image)
subplot(223)
imagesc(extractdata(O),[0,1])
colorbar, title('X'), axis image

% Ground truth (image)
subplot(224)
imagesc(H)
title('H'), colorbar, axis image
drawnow

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