% Plot recipes

h1 = figure(1);
plot(gradnum,'-ok','MarkerSize',10,'MarkerEdgeColor','red','LineWidth',0.6); hold on;
plot(grad,'-ok','MarkerSize',4,'MarkerEdgeColor','blue','LineWidth',0.6);
xlabel('Vectorized theta','FontSize',15, 'FontName','Arial');
ylabel('Gradient','FontSize',15, 'FontName','Arial');
legend('Numerical approx','Backpropagation','location','NorthWest');
legend boxoff;
saveas(h1,'GradientCheck','png');


h2 = figure(2);
plot(lambdavec,error_train,'-ok','MarkerSize',10,'MarkerEdgeColor','red','LineWidth',0.6); hold on;
plot(lambdavec,error_valid,'-ok','MarkerSize',4,'MarkerEdgeColor','blue','LineWidth',0.6);
xlabel('Lambda','FontSize',15, 'FontName','Arial');
ylabel('Error','FontSize',15, 'FontName','Arial');
legend('Training','Validation','location','NorthWest');
legend boxoff;
saveas(h2,'Validation Curve','png');

h3 = figure(3);
plot(cost_train,'-ok','MarkerSize',2,'MarkerEdgeColor','b','LineWidth',0.6);
xlabel('Iterations','FontSize',15, 'FontName','Arial');
ylabel('Train Cost','FontSize',15, 'FontName','Arial');
legend boxoff;
saveas(h3,'Training cost','png');
