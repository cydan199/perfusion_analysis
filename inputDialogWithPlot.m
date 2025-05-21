function userValue = inputDialogWithPlot(data, ptitle, q)
% inputDialogWithPlot Creates a modal input dialog with a plot above.
%
% [userValue] = inputDialogWithPlot()
%
% Opens a dialog window containing:
%   - An Axes for plotting.
%   - A Numeric Edit Field for user input.
%   - OK and Cancel buttons.
%
% The user can enter a numeric value, and upon clicking OK, the function
% plots Y = X^2 at the entered X value and returns the X value.
% Clicking Cancel or closing the dialog returns an empty array.

    % Initialize output
    userValue = [];

    % Get screen size
    screenSize = get(0, 'ScreenSize'); % [left bottom width height]
    dlgWidth = 700;
    dlgHeight = 450;

    % Calculate position to center the dialog
    dlgPosX = (screenSize(3) - dlgWidth) / 2;
    dlgPosY = (screenSize(4) - dlgHeight) / 2;
    
    % Create the dialog window as a uifigure
    % Make the dialog modal to prevent interaction with other windows
    % by using 'modal'
    dlg = uifigure('Name', 'Input Threshold Value for Visualization', ...
                  'Position', [dlgPosX dlgPosY dlgWidth dlgHeight], ...
                  'WindowStyle','modal',...
                  'CloseRequestFcn', @(src, event) onCloseRequest(src, event));
    
    
    
    % Create axes for plotting within the dialog
    ax = uiaxes('Parent', dlg, ...
               'Position', [50, 150, 600, 250], ...
               'Box', 'on');
    
    % Initialize an empty plot
    histogram(ax,nonzeros(data),'NumBins',150,'Normalization','pdf','Orientation','vertical'); %'probability'
    title(ax,ptitle,'FontName','Times New Roman','FontSize',12);% 'Normalized CoV values from B-scan basis CoV computation'
    xlabel(ax,'Pixel Values [a.u.]','FontName','Times New Roman')
    ylabel(ax, 'Normalized # of Pixels','FontName','Times New Roman')
    xlim(ax, [0,1])

    
    % Add a label for the input field
    lbl = uilabel(dlg, ...
                 'Text', q, ...
                 'Position', [80, 100, 450, 22], ...
                 'HorizontalAlignment', 'left');
    
    % Add a numeric edit field for user input
    editField = uieditfield(dlg, 'numeric', ...
                             'Position', [590, 100, 50, 22]);%, ...
                             % 'ValueChangedFcn', @(src, event) onValueChanged(src, event, ax));
    
    % Add an OK button
    btnOK = uibutton(dlg, 'push', ...
                    'Text', 'OK', ...
                    'Position', [150, 40, 80, 30], ...
                    'ButtonPushedFcn', @(src, event) onOK(src, event, dlg, editField, ax));
    
    % Add a Cancel button
    btnCancel = uibutton(dlg, 'push', ...
                        'Text', 'Cancel', ...
                        'Position', [450, 40, 80, 30], ...
                        'ButtonPushedFcn', @(src, event) onCancel(src, event, dlg));
    
    % Set KeyPressFcn for the dialog to handle Enter key presses
    dlg.KeyPressFcn = @(src, event) onDialogKeyPress(src, event, editField, btnOK, ax);
    
    % Wait for the dialog to close before returning control to the caller
    uiwait(dlg);
    
    % === Callback Functions ===
    
    % Callback for value changes in the edit field (optional live plotting)
    % function onValueChanged(src, event, axesHandle)
    %     % Retrieve the current value
    %     x = src.Value;
    % 
    %     % Clear previous plots
    %     % cla(axesHandle);
    % 
    %     % Validate the input
    %     % if ~isempty(x) && ~isnan(x)
    %     %     % Example computation: Y = X^2
    %     %     y = x^2;
    %     % 
    %     %     % Plot the new point
    %     %     plot(axesHandle, x, y, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    %     %     xlabel(axesHandle, 'X');
    %     %     ylabel(axesHandle, 'Y = X^2');
    %     %     title(axesHandle, ptitle);
    %     %     grid(axesHandle, 'on');
    %     % end
    % end

    % Callback for the OK button
    function onOK(src, event, dialogHandle, editHandle, axesHandle)
        % Retrieve the user input
        x = editHandle.Value
        
        % Validate the input
        if isempty(x) || isnan(x)
            uialert(dialogHandle, 'Please enter a valid numeric value.', 'Invalid Input');
            return;
        end
        
        % Assign the input value to the output
        userValue = x;
        
        % Resume execution and close the dialog
        uiresume(dialogHandle);
        delete(dialogHandle);
    end

    % Callback for the Cancel button
    function onCancel(src, event, dialogHandle)
        % Assign empty to the output
        userValue = [];
        
        % Resume execution and close the dialog
        uiresume(dialogHandle);
        delete(dialogHandle);
    end

    % Callback for handling dialog close requests (e.g., clicking 'X')
    function onCloseRequest(src, event)
        % Treat as a cancel action
        onCancel([], [], src);
    end

    % Callback for key presses within the dialog
    function onDialogKeyPress(src, event, editHandle, okButtonHandle, axesHandle)
        % Check if the pressed key is Enter or Return
        if strcmp(event.Key, 'return') || strcmp(event.Key, 'enter')
            % Check if the edit field is currently focused
            if isequal(src.CurrentObject, editHandle)
                % Trigger the OK button callback
                onOK([], [], src, editHandle, axesHandle);
            else
                onOK([], [], src, editHandle, axesHandle);
                % Optional: Define behavior when Enter is pressed outside the edit field
                % For example, you might want to trigger OK or ignore
            end
        end
    end

end
