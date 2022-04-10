
## .tmux.conf {#dot-tmux-dot-conf}

My most up to date config file is at github: [.dotfiles/.tmux.conf](https://github.com/sky-bro/.dotfiles/blob/master/.tmux.conf), and for better experience, I strongly suggest you use `Capslock` as your `Ctrl` key (I set `C-a` as my prefix instead of `C-b`).

```conf
# chenge prefix from `C-b` to `C-a`
set -g prefix C-a
unbind C-b
bind C-a send-prefix

# better window split, with "-" and "|"
unbind '"'
bind - splitw -v -c '#{pane_current_path}'
unbind %
bind | splitw -h -c '#{pane_current_path}'

# enable mouse
set-option -g mouse on

# use hjkl to
# change focus
bind -r k select-pane -U
bind -r j select-pane -D
bind -r h select-pane -L
bind -r l select-pane -R
# resize pane
bind -r ^k resizep -U 2 # upward (prefix Ctrl+k)
bind -r ^j resizep -D 2 # downward (prefix Ctrl+j)
bind -r ^h resizep -L 2 # to the left (prefix Ctrl+h)
bind -r ^l resizep -R 2 # to the right (prefix Ctrl+l)

# enable vi motions
setw -g mode-keys vi
# select, copy with v, y
bind -T copy-mode-vi v send-keys -X begin-selection
bind -T copy-mode-vi y send-keys -X copy-selection-and-cancel

set -g base-index 1
set -g pane-base-index 1

set -g status-interval 1
set -g status-justify left
setw -g monitor-activity on

# Set default term to xterm
# https://github.com/zsh-users/zsh-autosuggestions/issues/229
# https://stackoverflow.com/questions/18600188/home-end-keys-do-not-work-in-tmux
set -g default-terminal screen-256color
```


## Key Bindings {#key-bindings}

> `prefix` means `C-b` by default, or `C-a` for me.
> list all shortcust: `prefix ?`


### sessions {#sessions}

-   list session: `tmux ls`, `prefix s`
-   new session: `tmux new -s session_name` (attach now), `tmux new -ds session_name` (do not attach)
-   attach session: `tmux a -t session_name`
-   rename session: `prefix $`
-   kill session: `tmux kill-session -t session_name`


### windows {#windows}

-   new window: `prefix c`
-   next window: `prefix n`
-   previous window: `prefix p`
-   rename window: `prefix ,`
-   kill window: `prefix &`


### panes {#panes}

-   change focus between panes: `prefix h/j/k/l`
-   resize pane: `prefix C-h/j/k/l`
-   split pane: `prefix |`
-   vsplit pane: `prefix -`
-   toggle zoom: `prefix z`
-   kill pane: `prefix x`
-   scroll pane: use mouse wheel or `prefix [` then with vi motions


### copy &amp; paste {#copy-and-paste}

-   within tmux
    -   select &amp; copy with your mouse
    -   or first enter navigation: `prefix [`, then
        -   navigate with vi motions: `hjkl`, `C-f`, `C-b`, ...
        -   `v` or `shift+v` to start character/line level selection
        -   `o` to change active end of selection
        -   `y` to yank (copy) or `q` to quit navigation
        -   `prefix ]` to paste selection
-   bettwen tmux and your host
    -   hold shift and use mouse to select
    -   copy with `Ctrl+Shift+c` or `Ctrl+c` (depends on your system/terminal settings)
    -   paste with `Ctrl+Shift+v` or `Ctrl+v` (depends on your system/terminal settings)
-   command line
    -   `tmux save-buffer -` save paste buffer to file (here use `-` as a filename to mean stdin/stdout).
    -   `tmux paste-buffer`
    -   `tmux set-buffer`
    -   `tmux choose-buffer`

<!--listend-->

```shell
# copy to clipboard
tmux save-buffer - | xclip -i -sel clipboard
# paste from clipboard
tmux set-buffer "$(xclip -o -sel clipboard)"; tmux paste-buffer
```
