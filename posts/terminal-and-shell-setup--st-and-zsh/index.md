
My terminal emulator is st (simple terminal) from LukeSmith[^fn:1], and my shell is zsh (with ohmyzsh[^fn:2]).


## Dependencies {#dependencies}

-   dmenu
-   fzf[^fn:3]
-   pywal[^fn:4]


## ohmyzsh {#ohmyzsh}

```shell
# . start-proxy 1081 socks5h
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

Then config or restore[^fn:5] your `~/.zshrc` file.

```shell
dotfiles checkout ~/.zshrc
```


## colors and themes {#colors-and-themes}


### p10k {#p10k}

I use powerlevel10k[^fn:6] as my zsh theme.

1.  clone the repository:

    ```shell
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
    # for chinese users, recommend:
    # git clone --depth=1 https://gitee.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
    ```
2.  set `ZSH_THEME="powerlevel10k/powerlevel10k"` in `~/.zshrc`.
3.  configure with `p10k`


### Xresources and pywal {#xresources-and-pywal}

you can define your color scheme in `~/.Xresources` file, and load it with `xrdb ~/.Xresources`.

Or you can let pywal generates and sets a colorscheme for you:

```shell
#!/bin/sh

# We grab the wallpaper location from wal's cache so
# that this works even when a directory is passed.
image_path="${1:-"$(< "${HOME}/.cache/wal/wal")"}"

# -n tells =wal= to skip setting the wallpaper.
wal -n -i "$image_path"
feh --no-fehbg --bg-fill "$image_path"
```

This is a script[^fn:7] to set my wallpaper and color scheme from an image: `wal-feh wallpaper.png`.

And I put `exec --no-startup-id ~/bin/wal-feh` in my `~/.config/i3/config` to autostart it.


## fzf {#fzf}

Install fzf, then put this in your `~/.zshrc`:

```shell
source /usr/share/fzf/key-bindings.zsh
source /usr/share/fzf/completion.zsh
```


## zsh-autosuggestions {#zsh-autosuggestions}

Fish-like fast/unobtrusive autosuggestions for zsh.

1.  clone the repository:

    ```shell
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
    ```

2.  add the plugin to the `plugins` list inside the `~/.zshrc` file.

    ```shell
    plugins=(
        # other plugins...
        zsh-autosuggestions
    )
    ```


## keybindings {#keybindings}

-   `alt-l`: follow urls
-   `alt-y`: copy urls
-   `alt-o`: copy output of a command
-   `alt-j/k/d/u`: scroll down/up/faster-down/faster-up
-   `alt-c/v`: copy/paste
-   `Ctrl+t`: list files+folders in current directory (e.g., type `git add`, press `Ctrl+t`, select a few files using `Tab`, finally `Enter`)
-   `Ctrl+r`: search history commands
-   `ESC+c`: fuzzy change directory

[^fn:1]: [st](https://github.com/LukeSmithxyz/st) from Luke Smith
[^fn:2]: [Oh My Zsh](https://ohmyz.sh/) is a delightful, open source, community-driven framework for managing your Zsh configuration
[^fn:3]: [fzf](https://github.com/junegunn/fzf) is a general-purpose command-line fuzzy finder
[^fn:4]: [Pywal](https://github.com/dylanaraps/pywal) is a tool that generates a color palette from the dominant colors in an image.
[^fn:5]: use ~~a bare git repository~~ stow  to manage [my dotfiles](https://github.com/sky-bro/.dotfiles).
[^fn:6]: [Powerlevel10k](https://github.com/romkatv/powerlevel10k) is a theme for Zsh. It emphasizes speed, flexibility and out-of-the-box experience.
[^fn:7]: my [wal-feh](https://github.com/sky-bro/.dotfiles/blob/master/bin/wal-feh) script to set wallpaper and color scheme
