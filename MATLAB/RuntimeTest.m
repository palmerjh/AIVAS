nx = 100;
ny = 100;
x = 115;
y = 142;

a = rand(x,y);
b = rand(10*x,10*y);

sp = 0.0;
mm = 0.0;
alt = 0.0;

for sx = 1:nx
    for sy = 1:ny
        %a = gpuArray(A);
        %b = gpuArray(B);
        
        b_sub = b(sx:sx+x-1,sy:sy+y-1);
    
        sp = sp + SumProd2(a,b_sub);
        mm = mm + MaxMin2(a,b_sub);
        alt = alt + Alt2(a,b_sub);
        
    end
end
    
sp = sp / n
mm = mm / n
alt = alt / n
    