function p= condnorm(X,mu,sigma)
p=ones(size(X,1),1);
for i=1:size(X,2),
    p=p.*normpdf(X(:,i),mu(:,i),sigma(:,i));
    
end;
end