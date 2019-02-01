function x = reduceF(X)
%% Energy
E=sum(X.^2,2);
%% 4thPower
P=sum(X.^4,2);
%% NonlinearEnergy
NLE=sum(-X(:,3:end).*X(:,1:end-2)+X(:,2:end-1).^2,2);
%% Curve Length
CL=sum(X(:,2:end)-X(:,1:end-1),2);
%% new features
x =[E P NLE CL];
end