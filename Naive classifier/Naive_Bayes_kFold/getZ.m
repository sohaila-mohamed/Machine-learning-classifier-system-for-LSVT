function z = getZ(X,mu,sigma)
z=zeros(size(X));
for i=1:size(X,2),
    z(:,i)=(X(:,i)-mu(:,i))/sigma(:,i);
end;
end