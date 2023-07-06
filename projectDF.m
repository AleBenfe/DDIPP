function  [xlambda] = projectDF(elle,u,e,z,y, varargin)
%
% [lambda, biter, siter] = project_DF(f)
%
% Fletcher algoritmh 
%
% -------- input --------
% required:
%  elle  :lower bounds
%  u     : upper bounds
%  e,y   : terms of the linear constraint: y'*x=e
% z      : vector to be projected
%
% optional:
%   'MAXIT'             maximum iteration number
%   'TOLPOS'            tolerance for a positive value                  
%                       DEFAULT = 1e-5
%   'INITBETA'          value of the first regularizer parameter used
%                       DEFAULT = 0
%   'TOL'               tolerance used in SGP stopping criterion        
%                       DEFAULT = 1e-3
%   'TOL_R'             relative tolerance when solving Discr(x) - target = 0                          
%                       DEFAULT = 1e-6
%   'TOL_DL'          dynamic tolerance activation range              
%   'VERB'              DEFAULT = 0
%
%------- output --------
%
%   lambda              estimated regularization parameter
%   biter               bracketing phase iterations
%   siter               secant phase iterations
%
tini=tic;

lambda = 0; % initial lambda
%%%%%
dlambda = 1;                                 % initial dlambda 
tol_r = 1e-6*e;                                  % r function toleranca
tol_lam = 1e-6;                                % lambda tolerance
biter = 0;                                     % bracketing iterations
siter = 0;                                  % secant iterations
maxprojections = 1000;                      % maximum allowed iterations
verb =0;
flag_state = [];                            % F(irst) B(racketing) S(ecant)

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read optional parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'MAXIT'
                maxprojections = varargin{i+1};
            case 'INITLAMBDA'
                lambda = varargin{i+1};
            case 'TOL_R'
                tol_r = varargin{i+1};
            case 'TOL_DL'
                tol_lam = varargin{i+1};
            case 'VERB'
                verb = varargin{i+1};
            otherwise
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fletcher algorithm start
%%%%%%%%%%%%%%%%%%%%%%%%%%
z=z(:);
y=y(:);
elle=elle(:);
u=u(:);

if verb
       fprintf('\n ----- projectDF:   tol-function=%e tol-lambda=%e\n',tol_r, tol_lam);
       fprintf(' it st \t  lambda \t\t r(lambda) \t \n')
end
flag_state = 'F';

xlambda= min(max(elle,z+lambda*y),u);
r=sum(y.*xlambda)-e;

%if (lambda==0 & r>0) Se sum(elle)-e>0 o sum(u)-e<0  unfeasible
%    error('unfeasible!!');
%end

biter = biter + 1;
if verb
fprintf('%3d %c \t  %8e \t %8e   \n', ...
    biter, flag_state, lambda, r);
end
if ( abs(r) < tol_r)
     timetot=toc(tini);
    if verb 
        % print statistics/debug informations
    fprintf('\n ===> lambda=%f, Fletcher tot.iter.=%d <===\n', lambda, biter);
    end
    return
end

%%%%%%%%%%%%%%%%%%
% % Bracketing Pfhase
%%%%%%%%%%%%%%%%%%
flag_state = 'B';

if (r < 0)
    %%% estremo sinistro
    lambdal = lambda;
    rl = r;
    %%%%% 
    lambda = lambda+dlambda;
    xlambda= min(max(elle,z+lambda*y),u);
    r=sum(y.*xlambda)-e;
    biter = biter + 1;
    if verb
    % print statistics/debug informations
    fprintf('%3d %c \t  %8e \t %8e \t   \n', ...
                biter, flag_state, lambda, r);
    end
    while (r < 0) & biter<=maxprojections
        biter = biter+1;
        lambdal = lambda;
        s = max(rl/r-1, 0.1);
        dlambda = dlambda+dlambda/s;
        lambda = lambda+dlambda;
        rl = r;
        xlambda= min(max(elle,z+lambda*y),u);
        r=sum(y.*xlambda)-e;
       
        if verb 
             % print statistics/debug informations
             fprintf('%3d %c \t  %8e \t %8e \t  \n', ...
                     biter, flag_state, lambda, r);
        end
    end
    
    %% estremo destro
    lambdau = lambda;
    ru = r;  
else
    lambdau = lambda;
    %%estremo destro
    ru = r;
    lambda = lambda-dlambda;
    xlambda= min(max(elle,z+lambda*y),u);
    r=sum(y.*xlambda)-e;
    biter = biter + 1;
    if verb
    fprintf('%3d %c \t  %8e \t %8e \n', ...
             biter, flag_state, lambda, r);
    end
    while (r > 0) & biter<=maxprojections
        biter = biter+1;
        lambdau = lambda;
         s = max(ru/r-1, 0.1);
         dlambda = dlambda+dlambda/s;
         lambda = lambda-dlambda;
         ru = r;
        xlambda= min(max(elle,z+lambda*y),u);
        r=sum(y.*xlambda)-e;
      
        if verb 
           % print statistics/debug informations
           fprintf('%3d %c \t  %8e \t %8e \n', ...
                    biter, flag_state, lambda, r);
        end
    end
   %% estremo sinistro
   lambdal = lambda;
    rl = r;
end

% did i find solution on upper boundary ?

if (abs(ru) < tol_r)
    if verb
    fprintf('\n ===> lambda=%d, Fletcher tot.iter.=%d <===\n', lambda, siter+biter);
    end
    timetot=toc(tini);
    return
end

% did i find solution on lower boundary ?
if (abs(rl) < tol_r)
    if verb 
        fprintf('\n ===> lambda=%d, Fletcher tot.iter.=%d <===\n', lambda, siter+biter);
    end
    timetot=toc(tini);
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Secant Phase
%%%%%%%%%%%%%%%%%%%%%%%%%%
 
if biter >maxprojections
    timetot=toc(tini);
    return
end

flag_state = 'S';
%%%%%%
s = 1-rl/ru;
dlambda = dlambda/s;
lambda = lambdau-dlambda;
 xlambda= min(max(elle,z+lambda*y),u);
 r=sum(y.*xlambda)-e;

siter = siter + 1;

if verb 
fprintf('%3d %c \t  %8e \t %8e  \n', ...
    siter, flag_state, lambda, r);
end

maxit_s = maxprojections - biter;


while ( abs(r) > tol_r & ...
        dlambda > tol_lam * (1 + abs(lambda)) & ...
        siter < maxit_s )
    siter = siter + 1;
    if r > 0
    
        
        if (s <= 2)
            lambdau = lambda;
            ru = r;
            s = 1-rl/ru;
            dlambda = (lambdau-lambdal)/s;
            lambda = lambdau - dlambda;
        else
            s = max(ru/r-1, 0.1);
            dlambda = (lambdau-lambda) / s;
            lambda_new = max(lambda - dlambda, 0.75*lambdal+0.25*lambda);
            lambdau = lambda;
            ru = r;
            lambda = lambda_new;
            s = (lambdau - lambdal) / (lambdau-lambda);  
        end
    else
        if (s >= 2)
            lambdal = lambda;
            rl = r;
            s = 1-rl/ru;
            dlambda = (lambdau-lambdal)/s;
           
            lambda = lambdau - dlambda;
        else
           s = max(rl/r-1, 0.1);
            dlambda = (lambda-lambdal) / s;
            lambda_new = min(lambda + dlambda, 0.75*lambdau+0.25*lambda);
            lambdal = lambda;
            rl = r;
            lambda = lambda_new;
            s = (lambdau - lambdal) / (lambdau-lambda);
        end
    end
        
   xlambda= min(max(elle,z+lambda*y),u);
   r=sum(y.*xlambda)-e;

   if verb  
    % print statistics/debug informations
    fprintf('%3d %c \t  %8e \t %8e \t  \n', ...
    siter, flag_state, lambda, r);
   end
   
end

if verb
fprintf('\n');
% print statistics/debug informations
fprintf('\n ===> lambda=%d, Fletcher tot.iter.=%d <===\n', lambda, siter + biter);
end

timetot=toc(tini);


