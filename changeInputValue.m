function newValue = changeInputValue(currValue, val_name)
% changeValueDialog Prompts the user to decide whether to change a value.
% If Yes, prompts for the new value.
%
% Output:
%   newValue - The new value entered by the user if Yes is selected.
%              Empty if No is selected or the dialog is canceled.

    % Initialize output
    newValue = currValue; %[];

    % Define dialog size
    dlgWidth = 600;
    dlgHeight = 200;

    % Get screen size
    screenSize = get(0, 'ScreenSize'); % [left bottom width height]

    % Calculate position to center the dialog
    dlgPosX = (screenSize(3) - dlgWidth) / 2;
    dlgPosY = (screenSize(4) - dlgHeight) / 2;

    % Create the main dialog window
    dlg = uifigure('Position', [dlgPosX dlgPosY dlgWidth dlgHeight], ...
                  'Name', 'Change Value Dialog', ...
                  'CloseRequestFcn', @(src, event) onDialogClose(src, event), ...
                  'Resize', 'on');

    % === First Question: Do you want to change the value? ===
    % Instructional Label
    questionLabel = uilabel(dlg, ...
                            'Text', strcat("Do you want to change the ",val_name," value? (currently ", num2str(currValue),")"), ...
                            'Position', [80 dlgHeight-80 500 30], ...
                            'FontSize', 14, ...
                            'HorizontalAlignment', 'left');

    % Yes Button
    yesButton = uibutton(dlg, 'push', ...
                         'Text', 'Yes', ...
                         'Position',[460 dlgHeight-80 50 30], ...
                         'ButtonPushedFcn', @(btn,event) onYes());

    % No Button
    noButton = uibutton(dlg, 'push', ...
                        'Text', 'No', ...
                        'Position', [520 dlgHeight-80 50 30], ...
                        'ButtonPushedFcn', @(btn,event) onNo());

    % === Second Question: Enter the new value ===
    % Initially hidden
    newValueLabel = uilabel(dlg, ...
                            'Text', 'Enter the new offset value:', ...
                            'Position', [100 dlgHeight-120 500 30], ...
                            'FontSize', 14, ...
                            'HorizontalAlignment', 'left', ...
                            'Visible', 'off');

    newValueEdit = uieditfield(dlg, 'numeric', ...
                               'Value', currValue,...
                               'Position', [320 dlgHeight-120 150 30], ...
                               'Visible', 'off');

    % OK Button
    okButton = uibutton(dlg, 'push', ...
                        'Text', 'OK', ...
                        'Position', [200 25 80 30], ...
                        'ButtonPushedFcn', @(btn,event) onOK(), ...
                        'Visible', 'off');

    % Cancel Button
    cancelButton = uibutton(dlg, 'push', ...
                            'Text', 'Cancel', ...
                            'Position', [340 25 80 30], ...
                            'ButtonPushedFcn', @(btn,event) onCancel(), ...
                            'Visible', 'off');

    % === Callback Functions ===

    function onYes()
        % User selected Yes
        % Hide first question components
        questionLabel.Visible = 'on';
        yesButton.Visible = 'off';
        noButton.Visible = 'off';

        % Show second question components
        newValueLabel.Visible = 'on';
        newValueEdit.Visible = 'on';
        okButton.Visible = 'on';
        cancelButton.Visible = 'on';

        % Set focus to the edit field
        uistack(newValueEdit, 'top');
        newValueEdit.focus();
    end

    function onNo()
        % User selected No
        % Close the dialog
        uiresume(dlg);
        close(dlg);
    end

    function onOK()
        % Retrieve the entered value
        enteredValue = newValueEdit.Value;

        % Validate the input
        if isempty(enteredValue) || isnan(enteredValue)
            uialert(dlg, 'Please enter a valid numeric value.', 'Invalid Input');
            return;
        end

        % Assign the entered value to the output
        newValue = enteredValue;

        % Close the dialog
        uiresume(dlg);
        close(dlg);
    end

    function onCancel()
        % User canceled the input
        newValue = currValue;

        % Close the dialog
        uiresume(dlg);
        close(dlg);
    end

    function onDialogClose(src, event)
        % Handle the dialog close request (e.g., clicking the 'X' button)
        % newValue = [];

        % Resume execution and close the dialog
        uiresume(src);
        delete(src);
    end

    function onKeyPress(src, event)
        % Handle key press events
        if strcmp(event.Key, 'return') || strcmp(event.Key, 'enter')
            if okButton.Visible
                onOK();
            elseif yesButton.Visible || noButton.Visible
                % Default to 'Yes' if Enter is pressed in first question
                onYes();
            end
        end
    end


    % Set the KeyPressFcn for the dialog to handle Enter key globally
    dlg.KeyPressFcn = @(src, event) onKeyPress(src, event);

    % Make the dialog modal by waiting until it is closed
    uiwait(dlg);
end
