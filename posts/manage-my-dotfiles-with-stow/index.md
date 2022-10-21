
## Introduction {#introduction}

I used to manage my dotfiles[^fn:1] with a bare git repository, its simple, but dotfiles are all over the place, it's hard for me to get a whole view of them.

So now I've switched to stow[^fn:2], which is a symlink manager to help you put all files you want in one place and symlink them to where they belong (it creates symlink for files in one folder to another folder).


## First Time Setup {#first-time-setup}

So the first time we use stow to manage our dotfiles, we just need to follow these steps.

1.  create dotfiles folder in your home directory (preferably)
2.  move files to that folder
3.  add `.stow-local-ignore` file ([Types And Syntax Of Ignore Lists](https://www.gnu.org/software/stow/manual/html_node/Types-And-Syntax-Of-Ignore-Lists.html))
4.  create symbolic links back to the files moved (stow the dotfiles directory)
5.  (optional) backup the folder (like pushing to github)

And this is is an example of mine:

```shell
# step 1: create dotfiles folder in the home directory
cd ~
mkdir .dotfiles
# step 2: move files to the directory created
mv .vimrc .dotfiles/
# same folder structure inside .dotfiles as $HOME folder
mkdir .dotfiles/.config/i3 -p
mv .config/i3/config .dotfiles/.config/i3/
# ... more
cd .dotfiles
# step 3: add .stow-local-ignore file
vim .stow-local-ignore
# step 4: create symbolic links
stow .
# ls -al ~
```

The `.stow-local-ignore` file if for telling stow that you don't want to symlink some files, you want to ignore them, here's mine.

```conf
\.git
\.gitignore
.*\.org
^/LICENSE.*
^/COPYING
```


## restore from a dotfiles backup {#restore-from-a-dotfiles-backup}

Restoring dotfiles is very simple, just recreate the symbolic links.

1.  restore the dotfiles directory (git clone)
2.  create symbolic links back


## Other useful commands of stow {#other-useful-commands-of-stow}

```shell
# unlink files (v for verbose)
stow -vD .
```

[^fn:1]: use ~~a bare git repository~~ stow  to manage [my dotfiles](https://github.com/sky-bro/.dotfiles).
[^fn:2]: [GNU Stow](https://www.gnu.org/software/stow/) is a symlink farm manager.