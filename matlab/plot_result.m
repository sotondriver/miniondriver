%% plot number of nonzero
figure(1),clf
plot(tRange,numNoZero(:,:,6),'r','LineWidth',2);
title('adjustment of parameter tau','FontSize',15)
xlabel('tau','FontSize',13,'FontWeight','bold')
ylabel('number of nonzero weights','FontSize',13,'FontWeight','bold')
grid on
%% plot weights
figure(2),clf,
bar(abs(weights(:,Division,6)));
title('Nonzero weights','FontSize',15)
xlabel('weight index','FontSize',13,'FontWeight','bold')
ylabel('weight','FontSize',13,'FontWeight','bold')
grid on
%% plot costs and compare with chao
plot_cost_tr=ones(1,66);
for i=1:66
    index3=index_weights(i);
    plot_cost_tr(i)=costV_tr(index3,i);  
end
plot_cost_ts=costV_ts;

figure(3),clf,
subplot(2,1,1)
plot(plot_cost_tr);
grid on
subplot(2,1,2)
plot(plot_cost_ts);
grid on




