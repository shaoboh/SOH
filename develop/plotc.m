function [  ] = plotc( color )
%PLOTCONFIGURATION Summary of this function goes here
%   Detailed explanation goes here
set(gcf,'Position',[322,246,520,380], 'color','w')
box off

a=get(gcf,'children');

while 1
    tags=get(a(1),'tag');
    if strcmp(tags,'legend')
        a(1)=[];
    else
        break;
    end
end

if length(a)==1
    
    % set(gca,'Position',[.12 .15 .83 .75]);
    set(gca,'OuterPosition',[-.01 .01 1.02 .98]);
    figure_FontSize=16;
    set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
    set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','baseline');
    set(get(gca,'Title'),'FontSize',figure_FontSize,'Vertical','baseline');
    set(findobj(gca,'FontSize',10),'FontSize',figure_FontSize);
    set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
    %legend('Location','Best');
elseif length(a)==2
    
    
    figure_FontSize=16;
    set(get(a(2),'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
    set(get(a(2),'YLabel'),'FontSize',figure_FontSize,'Vertical','baseline');
    set(get(a(2),'Title'),'FontSize',figure_FontSize,'Vertical','baseline');
    set(findobj(a(2),'FontSize',10),'FontSize',figure_FontSize);
    set(findobj(get(a(2),'Children'),'LineWidth',0.5),'LineWidth',2);
    try
        set(get(a(1),'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
        set(get(a(1),'YLabel'),'FontSize',figure_FontSize,'Vertical','top');
    end
    set(get(a(1),'Title'),'FontSize',figure_FontSize,'Vertical','baseline');
    set(findobj(a(1),'FontSize',10),'FontSize',figure_FontSize);
    set(findobj(get(a(1),'Children'),'LineWidth',0.5),'LineWidth',2);
    
    set(get(a(1),'YLabel'),'Vertical','top');
    set(get(a(1),'XLabel'),'Visible','off');
    set(a(1),'xlim',get(a(2),'xlim'));
    set(a(1),'xtick',[],'xticklabel',[])
    set(a(1),'YAxisLocation','right','Box','off');
     set(a(2),'Box','off');
    set(a(1),'HitTest','off');
    setappdata(a(2),'graphicsPlotyyPeer',a(1));
    setappdata(a(1),'graphicsPlotyyPeer',a(2));
    
    set(a(1),'OuterPosition',[-.01 .01 1.02 .98]);
    position1=get(a(1),'position');
    set(a(2),'OuterPosition',[-.01 .01 1.02 .98]);
    position2=get(a(2),'position');
    
    newposition(1)=max([position1(1),position2(1)]);
    newposition(3)=min(position1(1)+position1(3),position2(1)+position2(3))-newposition(1);
    newposition(2)=max(position1(2),position2(2));
    newposition(4)=min(position1(2)+position1(4),position2(2)+position2(4))-newposition(2);
    
    set(a(1),'position',newposition)
    set(a(2),'Position',get(a(1),'position'))
    set(a(1),'Color','none');
        try
    set(a(2),'ycolor',color{1});set(a(1),'ycolor',color{2});
    end
    
end


end

