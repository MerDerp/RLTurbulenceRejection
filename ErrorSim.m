error = zeros(20,1);

for i = 1:20
    sim('asbSkyHoggEEE586',60);
    error(i) = sum((2000-altitude(:,3)).^2)./length(altitude(:,3));
end
std(error)