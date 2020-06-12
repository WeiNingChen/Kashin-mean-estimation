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

t = title([ ' $\varepsilon = $', num2str(eps), ', $b=$', num2str(eps), ', $d = $', num2str(d)]);
xlabel('$n$ number of samples','FontSize',16,'Interpreter', 'latex');
ylabel('$\ell_2$ error', 'FontSize',16, 'Interpreter', 'latex');
%set(gca,'XTick',[4000:8000:40000],'FontSize',16, 'TickLabelInterpreter', 'latex');
%axis([0.1 0.9 2 12]);
%leg = legend('informed', '$\alpha = 0.05$', '$\alpha = 0.1$', '$\alpha = 0.2$');
k_star = ceil(eps*log2(exp(1)))

%leg = legend(HQ, SS, RR, HR);
%if eps > 0.5
leg = legend('Kashin', 'privUnit');
%end
set(t,'FontSize',16, 'Interpreter','latex')
set(leg,'FontSize',16, 'Interpreter','latex')
%set(gca, 'YScale', 'log')