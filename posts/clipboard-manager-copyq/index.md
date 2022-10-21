
## Introduction {#introduction}

CopyQ[^fn:1] is a clipboard manager with many features.

-   manages clipboard history
-   history in different tabs
-   Store text, HTML, images or any other custom formats
-   Support custom commands[^fn:2], like saving clipboard items to file
-   vi style navigation


## Basic Setup {#basic-setup}

-   Enable `vi style navigation` in `Preferences -> General`
-   Enable `Tab Tree` and `Show Item Count` in `Preferences -> Layout`
-   Custom shortcuts in `Preferences -> Shortcuts` or in `File -> Commands/GlobalShortcuts (press F6 from main window)`


## Add Commands {#add-commands}

You can get many useful commands from CopyQ-Commands[^fn:2], or you can create your own commands following the documentation.

To add a command to CopyQ:

-   copy the command code (starts with [Command] or [Commands] for multiple commands)
-   open CopyQ (`Ctrl-Alt-h`)
-   open command dialog (`F6`)
-   click "Paste Commands" button (`Ctrl-v`)
-   apply changes

Commands that I use:

-   [Save Item/Clipboard To a File](https://github.com/hluk/copyq-commands/blob/master/Application/save-item-clipboard-to-file.ini): Opens dialog for saving selected item data to a file.
-   [Image Tab](https://github.com/hluk/copyq-commands/blob/master/Automatic/image-tab.ini): Automatically store images copied to clipboard in a separate tab.


## Key Bindings {#key-bindings}

-   `Ctrl-Alt-h`: open/close main window, show clipboard history (customized)
-   `Ctrl-Alt-s`: save as (customized)
-   `j/k`: next/previous item
-   `Ctrl-h`: previous tab
-   `l/Enter`: copy &amp; paste item
-   `Ctrl-c`: copy item
-   `ESC/Ctrl-[`: close window

[^fn:1]: [CopyQ](https://github.com/hluk/CopyQ), a clipboard manager with advanced features.
[^fn:2]: [copyq-commands](https://github.com/hluk/copyq-commands) is a repo which has many useful commands for CopyQ clipboard manager