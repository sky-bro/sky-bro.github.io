
## introduction {#introduction}

Pass[^fn:1] is a command line tool that manages (adding, editing, generating, retrieving) your passwords, but with many useful font-ends and ported clients for different platforms. Its very handy to use it for managing all your passwords on all your devices.

Each password lives in a gpg encrypted file.


## init {#init}

Before everything, we need to initialize the password store with a GPG key ([How to generate a gpg key](https://www.linode.com/docs/guides/gpg-keys-to-send-encrypted-messages/)):

```shell
# "Kyle Shi" is the user id (maybe email is better?) for my GPG key
# this will create a directory: $HOME\.password-store
# with a file .gpg-id in it (which stores this id)
pass init "Kyle Shi"
```

Passwords that you add will be stored in the `.password-store` directory, named `xxx.gpg`, which is encrypted with your gpg public key, which only you can decrypt with your private key (`gpg --output doc --decrypt doc.gpg`).


## manage with pass command {#manage-with-pass-command}


### add {#add}

```shell
pass insert Email/sky_io@outlook.com
```


### generate {#generate}

```shell
# generate password for an entry with length of 15
# if -n (--no-symbols) is passed: will not use non alphanumeric characters
pass generate Email/sky_io@outlook.com 15
```


### edit {#edit}

```shell
# edit an entry with your default editor
pass edit Email/sky_io@outlook.com
```


### remove {#remove}

```shell
pass rm Email/sky_io@outlook.com
```


### retrieve {#retrieve}

```shell
# list password entries
pass
# show password for an entry
pass Email/sky_io@outlook.com
# copy password of an entry to the clipboard
pass -c Email/sky_io@outlook.com
```

history problem with clipboard manager: [How to clear the history in CopyQ?](https://github.com/hluk/CopyQ/issues/1031)


## manage with other pass front ends {#manage-with-other-pass-front-ends}

We barely use `pass` command, instead we use other more user friendly front ends.


### unix/unix like: [rofi-pass](https://github.com/carnager/rofi-pass) {#unix-unix-like-rofi-pass}

-   `Alt+n`: add new password
-   `Alt+a`: action menu of current password
-   `Alt+h`: help


### windows {#windows}

-   [pass-winmenu](https://github.com/geluk/pass-winmenu)


### android {#android}

-   [Android-Password-Store](https://github.com/android-password-store/Android-Password-Store)


### ios {#ios}

-   [passforios](https://mssun.github.io/passforios/)


## Sync {#sync}


### git(hub) {#git--hub}

```shell
cd $HOME\.password-store
git init
git branch -M main
git add -A
git commit -m "init pass store"
git remote add origin git@github.com:sky-bro/password-store.git
git push -u origin main
```

Adding and removing passwords will automatically create git commits.


## browser settings {#browser-settings}

switch off `Offer to save passwords` in chrome settings

[^fn:1]: [pass](https://www.passwordstore.org/) is the standard unix password manager.