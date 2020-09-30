function [MI,rMI,bMI,tmat,signif]=MI2unbiased(h);
% tmat = a cell array that contains all the joint distribution matrixes 
% from which the mi was computed. Under the assumption that close/similar
% stimuli will induced similar responses, and that similar responses had 
% similar conditioned probabilities, the raw joint matrix was iteratively
% reduces by joining neighboring rows and columns. At each iteration, the
% stimulus class or the response class that had the lowest marginal
% probability was joint to one of its immediate neighbors, the one that had
% the lowest marginal distribution. For each of these matrixes the rmi and 
% the bmi were computed. For the last tmat the mi is zero since this mat
% will be one column or one row.
% Then, the max value of the differences between the rmi and the bmi was
% the mi itself. For computing the segnificance of the mi , we will use
% the fact that the raw mi is distributed as chi 2 , only with a constant. 
% 2*log(2)*prod(size(tmat{ii}-1))*rmi(ii)~chi 2 with (R-1)*(C-1) degrees
% of freedom. We need first to find the tmat that gives the max differences,
% and then we will compute the chi2cdf by
% [MI,ii]=max(rmi(1:end-1)-bmi(1:end-1)); The segnificance will be:
% 1-chi2cdf(2*log(2)*sum(tmat{ii}(:))*rmi(ii),prod(size(tmat{ii})-1))
% the signif of the mi will be computed by the rmi and the bmi. The bmi is
% asencialy the degrees of freedom ==((Ns-1)*(Nr-1))/(2*N*log(2)) where Ns
% is the number of stimuli Nr is the num of responses and N is the total
% number of stimuli (constant number which is sum(tmat{X}(:)), no mater which tmat)
% It is known that the bMI (under the null assumption) Ho is destributed as chi with (Ns-1)*(Nr-1) degrees of
% freedom. However if there is information the bMI is not distributed as
% chi anymore. Therefore, to compute the segnificance of mi we can't look
% at the mi, but to comper the rMI to the bMI. The question is what is the
% probability that the rMI will be drown from chi 2 with (Ns-1)*(Nr-1)
% degrees of freedom, which mean there is no information in the data and 
% the rMI is distributed as the bMI. 
if isempty(h)
    MI=0;
else
    ph=h/sum(h(:));
    phx=sum(ph,1);
    phy=sum(ph,2);
    approx=phy*phx;
    sph=ph;
    sph(sph==0)=1;
    approx(approx==0)=1;
    rMI(1)=sum(sum(ph.*log2(sph./approx)));
    bMI(1)=bias(h);
    h2u=h;
    tmat{1}=h;
    step=1;
    while min(size(h2u))>1
        th=h2u;
        [mx,Ix]=min(phx);
        [my,Iy]=min(phy);
        if my<mx
            th=th';
            Ix=Iy;
            phx=phy;
            history(step,1)=2;
            history(step,2)=Iy;
        else
            history(step,1)=1;
            history(step,2)=Ix;
        end
        if Ix==1
            th(:,2)=th(:,2)+th(:,1);
            th=th(:,2:end);
        elseif Ix==length(phx)
            th(:,end-1)=th(:,end-1)+th(:,end);
            th=th(:,1:end-1);
        elseif phx(Ix-1)<phx(Ix+1)
            th(:,Ix-1)=th(:,Ix-1)+th(:,Ix);
            th=[th(:,1:Ix-1) th(:,Ix+1:end)];
        else
            th(:,Ix)=th(:,Ix)+th(:,Ix+1);
            th=[th(:,1:Ix) th(:,Ix+2:end)];
        end
        h2u=th;
        ph=h2u/sum(h2u(:));
        phx=sum(ph,1);
        phy=sum(ph,2);
        approx=phy*phx;
        sph=ph;
        sph(sph==0)=1;
        approx(approx==0)=1;
        rMI(end+1)=sum(sum(ph.*log2(sph./approx)));
        bMI(end+1)=bias(h2u);
        tmat{end+1}=h2u;
    end
    tMI=rMI-bMI;
    [MI,ind]=max(tMI(1:end-1));
    df=prod(size(tmat{ind})-1);
    signif=1-chi2cdf(rMI(ind)*2*log(2)*sum(tmat{ind}(:)),df);
end

function b=bias(h)
ss=size(h);
b=(ss(1)-1)*(ss(2)-1)/(2*sum(h(:))*log(2)); 
MI2.m function MI=MI2(h);
if isempty(h)
    MI=0;
else
    ph=h/sum(h(:));
    phx=sum(ph,1);
    phy=sum(ph,2);
    approx=phy*phx;
    sph=ph;
    sph(sph<=0)=1;
    approx(approx<=0)=1;
    MI=sum(sum(ph.*log2(sph./approx)));
end 
