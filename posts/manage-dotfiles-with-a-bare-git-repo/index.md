
## Introduction {#introduction}

Using a bare git repo to manage dotfiles[^fn:1] is simple (idea from [this post](https://www.atlassian.com/git/tutorials/dotfiles), it only requires `git`), but now I've switch to stow[^fn:2], which in my view, grouping dotfiles together in one folder in easier and cleaner for me to find.


## Start {#start}

Create the repo

```shell
git init --bare $HOME/.dotfiles.git
```

Set git alias for the repo

```shell
echo "alias dotfiles='/usr/bin/git --git-dir=$HOME/.dotfiles.git/ --work-tree=$HOME'" >> $HOME/.zshrc # or .bashrc
. $HOME/.zshrc
```

Then use this command to not show untracked files on `dotfiles status`

```shell
dotfiles config --local status.showUntrackedFiles no
```


## Backup Files {#backup-files}

use `dotfiles` like your original `git` command

```shell
dotfiles status
dotfiles add .vimrc
dotfiles commit -m "backup .vimrc"
dotfiles remote add origin https://www.github.com/sky-bro/.dotfiles.git
dotfiles push origin master
```


## Restore Files {#restore-files}

On this computer

```shell
# rm .vimrc
dotfiles checkout
```

On another computer

```shell
echo 'alias dotfiles="/usr/bin/git --git-dir=$HOME/.dotfiles.git/ --work-tree=$HOME"' >> $HOME/.zshrc
source ~/.zshrc
echo ".dotfiles.git" >> .gitignore # prevent recursion issues
git clone --bare git@github.com:sky-bro/.dotfiles.git $HOME/.dotfiles.git
dotfiles checkout
dotfiles config --local status.showUntrackedFiles no
```

[^fn:1]: use ~~a bare git repository~~ stow  to manage [my dotfiles](https://github.com/sky-bro/.dotfiles).
[^fn:2]: [GNU Stow](https://www.gnu.org/software/stow/) is a symlink farm manager.