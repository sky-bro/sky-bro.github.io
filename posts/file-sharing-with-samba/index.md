
## Introduction {#introduction}

You should use NFS for dedicated Linux Client to Linux Server connections.

For mixed Windows/Linux environments use Samba.


## Server (linux) {#server--linux}


### install {#install}

```shell
sudo pacman -Sy samba
```


### configure {#configure}


#### configuration file {#configuration-file}

Need to create `/etc/samba/smb.conf` before starting the service `systemctl start nmb.conf`

You can get an example configuration file from [Samba Git Repository](https://git.samba.org/samba.git/?p=samba.git;a=blob_plain;f=examples/smb.conf.default;hb=HEAD).

Get help about writing the configuration file with `man smb.conf`.

Here's mine:

```toml
[global]
  workgroup = k4i.top
  map to guest = Bad Password
  server string = Samba Server
  passdb backend = tdbsam

[homes]
   comment = Home Directories
   browseable = no
   writable = yes

[tmp]
  comment = Temporary file space
  path = /tmp
  public = yes
  writable = yes
  printable = no
```


#### user management {#user-management}

we need to have a user to access linux files.

1.  For anonymous access, we use the guest account (by default, it's user `nobody`)
2.  If you want to access files in a home directory, you should login as that user.

We need to specifically add a user (an existing linux user, or a non-existing one -- samba will create the user for you) to samba, then set a password for that user (can be different from your linux login password)

```shell
sudo smbpasswd -a sky
```

And by default, when you access a samba server with a user, you can browser that user's home directory (if it has one).


### start {#start}

```shell
systemctl start smb.service
```


## Client {#client}


### windows {#windows}

In the file manager location bar, input: `\\servername\[share]`, then you may remap any folder to a drive.

tips: to clean user credentials (or with the control panel GUI)

```bat
net use /delete *.
```


### linux {#linux}

1.  With a file manager:

<https://wiki.archlinux.org/title/samba#File_manager_configuration>

In the location bar, input: `smb://servername/share`

1.  With a command line tool: `smbclient`

<!--listend-->

```shell
smbclient //xyz/public -U nobody
smbclient //xyz/sky -U sky
```


## Resources {#resources}

-   [File transfer between Linux and KVM Guest Windows 10](https://jeffshee.github.io/2021-01-29-samba-fedora33-kvm-windows-10/)
-   [Arch Wiki: samba](https://wiki.archlinux.org/title/samba)
