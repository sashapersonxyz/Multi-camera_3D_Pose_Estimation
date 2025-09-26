NexiGo N60 FHD Webcam.mov
#!/bin/bash

# Force kill any remaining QuickTime Player processes
killall "QuickTime Player" 2>/dev/null || true
sleep 1

# Check if four or five arguments are provided
if [ $# -lt 4 ] || [ $# -gt 5 ]; then
    echo "Usage: $0 /path/to/first.mov /path/to/second.mov \"First Webcam Name\" \"Second Webcam Name\" [duration_seconds]"
    exit 1
fi

# Optional: Duration to record in seconds (default: 10)
RECORD_SECONDS="${5:-10}"

#Delay time based off of time need for quicktime to deal with video of length RECORD_SECONDS
DELAY_TIME=$(( RECORD_SECONDS / 4 ))


# Expand paths
SAVE_PATH1="$(eval echo "$1")"
SAVE_PATH2="$(eval echo "$2")"
WEBCAM1="$3"
WEBCAM2="$4"

# Ensure directories exist
mkdir -p "$(dirname "$SAVE_PATH1")"
mkdir -p "$(dirname "$SAVE_PATH2")"

# Check for existing files
if [ -f "$SAVE_PATH1" ] || [ -f "$SAVE_PATH2" ]; then
    echo "One or both output files already exist:"
    [ -f "$SAVE_PATH1" ] && echo " - $SAVE_PATH1"
    [ -f "$SAVE_PATH2" ] && echo " - $SAVE_PATH2"
    read -p "Do you want to overwrite them? [y/N]: " confirm
    case "$confirm" in
        [yY][eE][sS]|[yY])
            [ -f "$SAVE_PATH1" ] && rm -f "$SAVE_PATH1"
            [ -f "$SAVE_PATH2" ] && rm -f "$SAVE_PATH2"
            echo "Existing files deleted. Continuing..."
            ;;
        *)
            echo "Aborted to avoid overwriting files."
            exit 1
            ;;
    esac
fi

# Open QuickTime and select first webcam
osascript -e "
tell application \"QuickTime Player\"
    activate
    set newRecording to new movie recording
end tell
tell application \"System Events\" to tell process \"QuickTime Player\"
    delay 1
    click button 2 of window 1
    click menu item \"$WEBCAM1\" of menu 0 of button 2 of window 1
    delay 1
end tell
"

# Open second QuickTime instance and select second webcam
open -n -a "QuickTime Player"
osascript -e "
tell application \"QuickTime Player\"
    activate
    set newRecording to new movie recording
end tell
tell application \"System Events\" to tell process \"QuickTime Player\"
    delay 1
    click button 2 of window 1
    click menu item \"$WEBCAM2\" of menu 0 of button 2 of window 1
end tell
"

# Start recording on all instances
osascript -e '
delay 1
tell application "System Events"
    set pidList to the unix id of (processes whose name is "QuickTime Player")
    repeat with pid in pidList
        tell (first process whose unix id is pid)
            try
                tell window 1
                    click button 3
                end tell
            end try
        end tell
    end repeat
end tell
'

# Wait and save
osascript <<EOF
delay $RECORD_SECONDS

on saveAndQuit(savePath)
    set AppleScript's text item delimiters to "/"
    set fileName to last text item of savePath
    set folderPath to text 1 thru ((offset of fileName in savePath) - 2) of savePath

    tell application "QuickTime Player"
        activate
        try
            tell document 1 to stop
        end try
    end tell
    delay $DELAY_TIME
    tell application "System Events"
        delay 0.5
        keystroke "s" using {command down}
        delay 1
        keystroke "G" using {command down, shift down}
        delay 0.5
        keystroke folderPath
        delay 0.5
        keystroke return
        delay 0.5
        keystroke "a" using {command down}
        delay 0.5
        keystroke fileName
        delay 0.5
        keystroke return
        delay $DELAY_TIME
        keystroke "q" using {command down}
    end tell
end saveAndQuit

saveAndQuit("$SAVE_PATH1")
delay 2
saveAndQuit("$SAVE_PATH2")
EOF
