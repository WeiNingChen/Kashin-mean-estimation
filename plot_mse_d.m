%clear all; 
%close all;


figure(1);
hold on;
%grid on;

%FI_informed
%error_upbd = ((k^2)/indices*((exp(eps)+2^comm-1)/(exp(eps)-1))^2)^(1/2)

%error_upbd = (double(d)*(2*exp(double(eps))-1))/((exp(double(eps))-1))*(double(indices)).^(-1/2)

plot(indices, kashin_mse,'-r','Linewidth', 3);
plot(indices, privUnit_mse,'--b','Linewidth', 3);


%plot(indices, error_upbd,'--k','Linewidth', 2);


%plot(X,FI_anony_1','--b', 'Linewidth', 2); 
%plot(X,FI_anony_2','--g', 'Linewidth', 3);
%plot(X,FI_anony_3','--c', 'Linewidth', 3);
%plot(X,FI_anony_4','--m', 'Linewidth', 3);

%set(gca,'XTick',[0.1:0.2:0.9],'YTick',[0:3:12],'FontSize',16, 'TickLabelInterpreter', 'latex');
%set(gca,'XTick',[100000:100000:500000],'FontSize',16, 'TickLabelInterpreter', 'latex');
%set(gca,'YTick',[0:0.4:2],'FontSize',16, 'TickLabelInterpreter', 'latex');
t = title([ '$\varepsilon = $', num2str(eps), ', $b=5$, $n = $', num2str(n)]);
xlabel('$d$ dimension','FontSize',16,'Interpreter', 'latex');
ylabel('$\ell_2$ error', 'FontSize',16, 'Interpreter', 'latex');
%axis([0.1 0.9 2 12]);
%leg = legend('informed', '$\alpha = 0.05$', '$\alpha = 0.1$', '$\alpha = 0.2$');
k_star = ceil(eps*log2(exp(1)))

%leg = legend(HQ, SS, RR, HR);
%if eps > 0.5
leg = legend('Kashin', 'privUnit');
%leg = legend('privUnit');
%end
set(t,'FontSize',16, 'Interpreter','latex')
set(leg,'FontSize',16, 'Interpreter','latex')
