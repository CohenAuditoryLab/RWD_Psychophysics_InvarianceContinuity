
figure('Position',[5 100 1912 1025]);

bins = linspace(0,.35,130);
for k = 1:16
    ind = k*2;
    subplot(4,4,k);
    semilogy(target_S.env_bins(ind,:),target_S.env_hist(ind,:),'b');hold on;
    semilogy(synth_S.env_bins(ind,:),synth_S.env_hist(ind,:),'r');
    ylim = get(gca,'YLim');
    set(gca,'XLim',[-.01 .35]);
    if rem(k,4)==1
        ylabel('Prob of Occurrence','FontSize',10);
    end
    if k>12
        xlabel('Envelope Magnitude','FontSize',10);
    end
    if k==2
        title(['Envelope Histogram Comparison for ' fig_title_string],'FontSize',12);
    elseif k>3
        title(['Subband ' num2str(ind) '    (' num2str(round(audio_cutoffs_Hz(ind))) ' Hz)'],'FontSize',10);
    end
end
set(gcf,'PaperOrientation', 'landscape','PaperPosition',[0.25 0.25 10.5 8]);
