function updateMap(cat_loc, target_loc, mouse_loc, cat, mouse, cheese)
    xlim([0 15]); ylim([0 15])
    set(gca,'YDir','reverse')
    axis square

    hold on;
    image(flipud(cat), 'XData', [cat_loc(2) cat_loc(2)+1],...
        'YData', [cat_loc(1) cat_loc(1)+1]);

    hold on;
    image(flipud(cheese), 'XData', [target_loc(2) target_loc(2)+1],...
        'YData', [target_loc(1) target_loc(1)+1]);

    
    hold on;
    image(flipud(mouse), 'XData', [mouse_loc(2) mouse_loc(2)+1],...
        'YData', [mouse_loc(1) mouse_loc(1)+1]);
    hold off;
end