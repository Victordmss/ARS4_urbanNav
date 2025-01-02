function LL=ellipse(X,P,PROBA,color,dir)

% ellipse(X,P,PROBA,color,dir)
% trace d'ellipse
% 	[X(1) X(2] = centre de l'ellipse
%	P  = Matrice de covariance associee
% 	PROBA = probabilite associee a l'ellipse
%   color = 'r' par exemple
%   dir = si 'dir' est defini, on trace le vecteur directeur

if (nargin<3), PROBA=(1-exp(-1^2/2)); end; %un sigma pour une gaussienne
if (nargin<4), color = 'b'; end;
if (nargin<5), trace_grand_axe=0; else trace_grand_axe=1;end;
if (length(X)~=2), error('Le vecteur X doit etre de dim 2'); end;
if (length(P(:,1))~=2) | (length(P(1,:))~=2), error('La matrice P doit etre 2x2');end;

epsilon=1e-6;
if (P(2,1)-P(1,2)>epsilon), error('La matrice de cov n''est pas symetrique');end;
[vp]=eig(P);
if ((vp(1)<0)|(vp(2)<0)), error('La matrice de cov n''est pas definie positive');end;

imax=20; %1/4 du nb de points traces
% le scalaire "k" definit l'ellipse avec l'equation :(x-mx)T*(1/P)*(x-mx)=k^2
k=sqrt(-2*log(1-PROBA));

%coeficient de correlation
Ro=P(1,2)/sqrt(P(1,1)*P(2,2));
if (abs(Ro)>1)
   disp('Le coeficient de correlation n''est pas compris entre -1 et 1');
   error('La matrice de covariance n''est pas definie positive');
end

%nb le cas ou abs(Ro)=1 est un cas limite a eviter
if (Ro>1-epsilon),  Ro= 1-epsilon; warning('Ro proche de 1');end;
if (Ro<-1+epsilon),   Ro=-1+epsilon;    warning('Ro proche de -1');end;

Cx=P(1,1);Cy=P(2,2);
a=1/(Cx*(1-Ro^2));b=-Ro/(sqrt(Cy*Cx)*(1-Ro^2)); c=1/(Cy*(1-Ro^2));

% on test si b=0 alors on lui affecte une valeur faible non nulle
% c'est le cas des ellipses d'axes (ox,oy)
if (sqrt(b*b)<1e-9), b=1e-9; end;

%calcul des deux valeurs propres
% la gde vp (lambda1) est associee au petit axe.
delta=(a-c)*(a-c)+4*b*b;
lambda1=0.5*(a+c+sqrt(delta));
lambda2=0.5*(a+c-sqrt(delta));

% vecteur directeur du grand axe
aux=(lambda2-a)/b; deno=sqrt(1+aux*aux);
Ux=1/deno;
Uy=aux/deno;

%longueur des axes dans le repere propre
axeX=k/sqrt(lambda2); 
axeY=k/sqrt(lambda1);

% trace proprement dit
dq=pi/2/imax;x=X(1);y=X(2);
point_ellipse=zeros(4*(imax+1),2);
for (i=1:4*(imax+1)),
    x0=axeX*cos(dq*i);                     %coord dans le repere propre
    y0=axeY*sin(dq*i);
    point_ellipse(i,:)=[x+x0*Ux-y0*Uy, y+x0*Uy+y0*Ux];   %coord dans R0
end;

if trace_grand_axe, 
   Lout=plot([x; x+Ux],[y; y+Uy],color,x,y,['+',color],point_ellipse(:,1),point_ellipse(:,2),color);
else
   Lout=plot(x,y,['+',color],point_ellipse(:,1),point_ellipse(:,2),color);
end;

if nargout==1,    LL=Lout;    end

