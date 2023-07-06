function [obj,gn,psf,bg,estH,scaling,rho] = loadData(problemName,varargin)
%
% [obj,gn,psf,bg,estH,scaling,rho] = loadData(problemName,varargin)
%
% Matlab function loading the data. There are two presets: rice image
% (based on the rice image from MatLab Image Processing toolbox) and
% synth001, a synthetic image of spherical particles. The function allows
% to load a custom mat file, which has to contain the blurred image, the
% ground truth, the background term, the PSF used for the creation of gn
% and the upper bound rho for the PSF estimation.
%
% Mandatory Input
%
% problemName (string)  : name of the mat file containing the data. Two
% presets: 'rice.mat', 'synth001.mat'.
%
% Optional Input
%
% nLev (integer) : parameter for different noise levels. Default: 1.
%
% estH (double array) : estimated PSF for the initialization of the kernel
%                       net. Default : Gaussian PSF of dimension 128 and
%                       sigma 64.
%
% Output:
%
% obj (double array)    : Ground truth image
% gn (double array)     : Perturbed image
% psf (double array)    : Point Spread Function used for the creating gn
% bg (double)           : background term
% estH (double array)   : initial estimate for the PSF
% scaling (double)      : scaling term for the normalization
% rho (double)          : upper bound for the estimation of the PSF
%
%==========================================================================
%
% Version : 1.0 (22-06-2023)
% Author  : A. Benfenati (alessandro.benfenati@unimi.it)
%
%==========================================================================

% Default values
nLev = 1;
estH = psfGauss(128,64);

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs.');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'NOISELEV'
                nLev = varargin{i+1};
            case 'ESTH'
                estH = varargin{i+1};
            otherwise
                error('Unknown option.')
        end
    end
end

switch problemName

    case 'synth001.mat'
        data = load(problemName);
        obj = double(data.obj);
        psf = data.psf;
        gn  = data.gn;
        rho = data.rho;
        if nLev~=1
            gn = imfilter(obj*nLev,psf,'conv');
            gn = 1e12*imnoise(1e-12*double(gn),'Poisson');
        end
        scaling = max(gn(:));
        obj = obj/scaling;
        gn  = gn/scaling;

        gn  = dlarray(single(gn),'SSCB');
        obj = dlarray(single(obj),'SSCB');
        bg  = min(gn(:));

        % The psf image should match the spatial dimension of the obj
        psf = padarray(psf,32,0,'pre');
        psf = padarray(psf,32,0,'post');
        psf = padarray(psf',32,0,'post');
        psf = padarray(psf,32,0,'pre');
        clear data;

    case 'rice.mat'
        data = load(problemName);
        obj = data.obj;
        psf = data.psf;
        gn  = data.gn;
        rho = data.rho;

        if nLev~=1
            gn = imfilter(obj*nLev,psf,'conv');
            gn = 1e12*imnoise(1e-12*double(gn),'Poisson');
        end
        scaling = max(gn(:));
        obj = obj/scaling;
        gn  = gn/scaling;
        bg  = data.bg/scaling;

        gn  = dlarray(single(gn),'SSCB');
        obj = dlarray(single(obj),'SSCB');
        % The psf image should match the spatial dimension of the obj

        psf = padarray(psf,56,0,'pre');
        psf = padarray(psf,55,0,'post');
        psf = padarray(psf',55,0,'post');
        psf = padarray(psf,56,0,'pre');
        clear data;

    case {'micro.mat'}
        data = load(problemName);
        obj = data.obj;
        psf = data.psf;
        gn  = data.gn;
        rho = data.rho;
        if nLev~=1
            gn = imfilter(obj*nLev,psf,'conv');
            gn = 1e12*imnoise(1e-12*double(gn),'Poisson');
        end
        scaling = max(gn(:));
        obj = obj/scaling;
        gn  = gn/scaling;

        gn  = dlarray(single(gn),'SSCB');
        obj = dlarray(single(obj),'SSCB');
        bg  = min(gn(:));
        estH = psfGauss(64,32);
        psf = psf(33:96,33:96);
        clear data;


    otherwise
        if ~isfile(problemName)
            error('Unkonwn problem.')
        end
        data = load(problemName);
        obj = double(data.obj);
        % The psf image should match the spatial dimension of the obj
        psf = data.psf;
        gn  = data.gn;
        rho = data.rho;
        scaling = max(gn(:));
        obj = obj/scaling;
        gn  = gn/scaling;
        bg  = data.bg/scaling;

        gn  = dlarray(single(gn),'SSCB');
        obj = dlarray(single(obj),'SSCB');
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
