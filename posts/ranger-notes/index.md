
I am planing on totally changing to the terminal based file manager: ranger.

And this is my cheatsheet on using it, for more detailed guides you can go check the ranger official user guide[^fn:1].


## Key bindings and hints {#key-bindings-and-hints}

-   `g`: navigation and tabs
-   `r`: open with
-   `y`: yank
-   `d`: cut/delete
-   `p`: paste
-   `o`: sort
-   `.`: filter\_stack ??
-   `z`: settings
-   `u`: undo
-   `M`: linemode
-   `+, -, =`: rights
-   `Alt+N`: switch(`Tab`), create tab


## configuration files {#configuration-files}

under `~/.config/ranger/` folder, there are 4 main configuration files:

-   `rc.conf`: the main config, various key bindings and switches
-   `rfile.conf`: how to open a file
-   `scope.sh`: how to preview a file
-   `commands.py`: implement various commands (functions), you can add your custom commands here.


## Bookmarks {#bookmarks}

-   `m<key>`: bookmark current folder
-   `'<key>`: go to a bookmark
-   `um<key>`: remove a bookmark


## Select/Mark files {#select-mark-files}

-   `SPC`: mark current file
-   `v`: invert selection (easy to select all)
-   `V`: visual mode, to mark a range of files
-   `:mark REGEX`, `:unmark REGEX`: to mark/unmark with regex expression.
-   `uv`, `:unmark`: unmark all files


## Rename, Create Files & Folders {#rename-create-files-and-folders}

-   `cw`: to rename selected file or files (bulk rename, works great with `:flat`)
-   `:mkdir`: create directory
-   `:touch`: create file

[^fn:1]: [ranger official user guide](https://github.com/ranger/ranger/wiki/Official-User-Guide)
