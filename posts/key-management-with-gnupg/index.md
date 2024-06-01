
## Introduction {#introduction}

GnuPG&nbsp;[^fn:1] is a tool for key management, and encrypting/decrypting/signing with those keys.

Basic gpg commands

```shell
# list keys
gpg --list-keys # list pub key
gpg --list-secret-keys # list secret key
```

Key flags

-   (S)ign: sign some data (like a file), used in git commit signing
-   (C)ertify: sign a key (this is called certification), used in creating sub keys
-   (A)uthenticate: authenticate yourself to a computer (for example, logging in), used in ssh login
-   (E)ncrypt: encrypt data, used in [password encryption](/posts/manage-passwords-with-pass/)


## Generate Key Pairs {#generate-key-pairs}

First create A primary key, then you can generate sub keys for various signing or encryption purposes by editing the primary key.

```shell
# generate primary key
gpg --full-gen-key

# edit primary key to add sub keys
gpg --edit-key <keyid>
gpg> addkey
```

{{< tabs "Primary Key" "Sub Signing Key" "Sub Authentication Key" "Sub Encryption Key" >}}

{{< tab >}}

```shell
❯ gpg --full-gen-key
Please select what kind of key you want:
   (1) RSA and RSA
   (2) DSA and Elgamal
   (3) DSA (sign only)
   (4) RSA (sign only)
   (9) ECC (sign and encrypt) *default*
  (10) ECC (sign only)
  (14) Existing key from card
Your selection? 10
Please select which elliptic curve you want:
   (1) Curve 25519 *default*
   (4) NIST P-384
   (6) Brainpool P-256
Your selection?
Please specify how long the key should be valid.
         0 = key does not expire
      <n>  = key expires in n days
      <n>w = key expires in n weeks
      <n>m = key expires in n months
      <n>y = key expires in n years
Key is valid for? (0) 0
Key does not expire at all
Is this correct? (y/N) y

GnuPG needs to construct a user ID to identify your key.

Real name: gpg test
Email address: gpg@test.com
Comment: gpg test
You selected this USER-ID:
    "gpg test (gpg test) <gpg@test.com>"

Change (N)ame, (C)omment, (E)mail or (O)kay/(Q)uit? Q
gpg: Key generation canceled.
❯ gpg --full-gen-key
Please select what kind of key you want:
   (1) RSA and RSA
   (2) DSA and Elgamal
   (3) DSA (sign only)
   (4) RSA (sign only)
   (9) ECC (sign and encrypt) *default*
  (10) ECC (sign only)
  (14) Existing key from card
Your selection? 10
Please select which elliptic curve you want:
   (1) Curve 25519 *default*
   (4) NIST P-384
   (6) Brainpool P-256
Your selection?
Please specify how long the key should be valid.
         0 = key does not expire
      <n>  = key expires in n days
      <n>w = key expires in n weeks
      <n>m = key expires in n months
      <n>y = key expires in n years
Key is valid for? (0)
Key does not expire at all
Is this correct? (y/N) y

GnuPG needs to construct a user ID to identify your key.

Real name: gpg test
Email address: gpg@test.com
Comment: test gpg
You selected this USER-ID:
    "gpg test (test gpg) <gpg@test.com>"

Change (N)ame, (C)omment, (E)mail or (O)kay/(Q)uit? O
We need to generate a lot of random bytes. It is a good idea to perform
some other action (type on the keyboard, move the mouse, utilize the
disks) during the prime generation; this gives the random number
generator a better chance to gain enough entropy.
gpg: directory '/home/sky/.gnupg/openpgp-revocs.d' created
gpg: revocation certificate stored as '/home/sky/.gnupg/openpgp-revocs.d/8CC83C9BCBECF783E2E7A2A3F3FB07FA468D5E57.rev'
public and secret key created and signed.

pub   ed25519 2024-06-01 [SC]
      8CC83C9BCBECF783E2E7A2A3F3FB07FA468D5E57
uid                      gpg test (test gpg) <gpg@test.com>
```

{{< /tab >}}

{{< tab >}}

```shell
gpg> addkey
Please select what kind of key you want:
   (3) DSA (sign only)
   (4) RSA (sign only)
   (5) Elgamal (encrypt only)
   (6) RSA (encrypt only)
  (10) ECC (sign only)
  (12) ECC (encrypt only)
  (14) Existing key from card
Your selection? 10
Please select which elliptic curve you want:
   (1) Curve 25519 *default*
   (4) NIST P-384
   (6) Brainpool P-256
Your selection?
Please specify how long the key should be valid.
         0 = key does not expire
      <n>  = key expires in n days
      <n>w = key expires in n weeks
      <n>m = key expires in n months
      <n>y = key expires in n years
Key is valid for? (0) 5y
Key expires at Thu 31 May 2029 11:43:17 AM CST
Is this correct? (y/N) y
Really create? (y/N) y
We need to generate a lot of random bytes. It is a good idea to perform
some other action (type on the keyboard, move the mouse, utilize the
disks) during the prime generation; this gives the random number
generator a better chance to gain enough entropy.

sec  ed25519/F3FB07FA468D5E57
     created: 2024-06-01  expires: never       usage: SC
     trust: ultimate      validity: ultimate
ssb  ed25519/9CE53048740E5BFC
     created: 2024-06-01  expires: 2029-05-31  usage: S
[ultimate] (1). gpg test (test gpg) <gpg@test.com>

gpg> save
```

{{< /tab >}}

{{< tab >}}

```shell
❯ gpg --expert --edit-key 8CC83C9BCBECF783E2E7A2A3F3FB07FA468D5E57
Secret key is available.

sec  ed25519/F3FB07FA468D5E57
     created: 2024-06-01  expires: never       usage: SC
     trust: ultimate      validity: ultimate
ssb  ed25519/9CE53048740E5BFC
     created: 2024-06-01  expires: 2029-05-31  usage: S
[ultimate] (1). gpg test (test gpg) <gpg@test.com>

gpg> addkey
Please select what kind of key you want:
   (3) DSA (sign only)
   (4) RSA (sign only)
   (5) Elgamal (encrypt only)
   (6) RSA (encrypt only)
   (7) DSA (set your own capabilities)
   (8) RSA (set your own capabilities)
  (10) ECC (sign only)
  (11) ECC (set your own capabilities)
  (12) ECC (encrypt only)
  (13) Existing key
  (14) Existing key from card
Your selection? 7

Possible actions for this DSA key: Sign Authenticate
Current allowed actions: Sign

   (S) Toggle the sign capability
   (A) Toggle the authenticate capability
   (Q) Finished

Your selection? S

Possible actions for this DSA key: Sign Authenticate
Current allowed actions:

   (S) Toggle the sign capability
   (A) Toggle the authenticate capability
   (Q) Finished

Your selection? A

Possible actions for this DSA key: Sign Authenticate
Current allowed actions: Authenticate

   (S) Toggle the sign capability
   (A) Toggle the authenticate capability
   (Q) Finished

Your selection? Q
DSA keys may be between 768 and 3072 bits long.
What keysize do you want? (2048) 3072
Requested keysize is 3072 bits
Please specify how long the key should be valid.
         0 = key does not expire
      <n>  = key expires in n days
      <n>w = key expires in n weeks
      <n>m = key expires in n months
      <n>y = key expires in n years
Key is valid for? (0) 5y
Key expires at Thu 31 May 2029 11:57:01 AM CST
Is this correct? (y/N) y
Really create? (y/N) y
We need to generate a lot of random bytes. It is a good idea to perform
some other action (type on the keyboard, move the mouse, utilize the
disks) during the prime generation; this gives the random number
generator a better chance to gain enough entropy.
gpg: WARNING: some OpenPGP programs can't handle a DSA key with this digest size

sec  ed25519/F3FB07FA468D5E57
     created: 2024-06-01  expires: never       usage: SC
     trust: ultimate      validity: ultimate
ssb  ed25519/9CE53048740E5BFC
     created: 2024-06-01  expires: 2029-05-31  usage: S
ssb  dsa3072/3AC06AD74E230931
     created: 2024-06-01  expires: 2029-05-31  usage: A
[ultimate] (1). gpg test (test gpg) <gpg@test.com>

gpg> save
```

{{< /tab >}}

{{< tab >}}

```shell
❯ gpg --edit-key 8CC83C9BCBECF783E2E7A2A3F3FB07FA468D5E57
Secret key is available.

sec  ed25519/F3FB07FA468D5E57
     created: 2024-06-01  expires: never       usage: SC
     trust: ultimate      validity: ultimate
ssb  ed25519/9CE53048740E5BFC
     created: 2024-06-01  expires: 2029-05-31  usage: S
ssb  dsa3072/3AC06AD74E230931
     created: 2024-06-01  expires: 2029-05-31  usage: A
[ultimate] (1). gpg test (test gpg) <gpg@test.com>

gpg> addkey
Please select what kind of key you want:
   (3) DSA (sign only)
   (4) RSA (sign only)
   (5) Elgamal (encrypt only)
   (6) RSA (encrypt only)
  (10) ECC (sign only)
  (12) ECC (encrypt only)
  (14) Existing key from card
Your selection? 6
RSA keys may be between 1024 and 4096 bits long.
What keysize do you want? (3072) 3072
Requested keysize is 3072 bits
Please specify how long the key should be valid.
         0 = key does not expire
      <n>  = key expires in n days
      <n>w = key expires in n weeks
      <n>m = key expires in n months
      <n>y = key expires in n years
Key is valid for? (0) 5y
Key expires at Thu 31 May 2029 03:26:57 PM CST
Is this correct? (y/N) y
Really create? (y/N) y
We need to generate a lot of random bytes. It is a good idea to perform
some other action (type on the keyboard, move the mouse, utilize the
disks) during the prime generation; this gives the random number
generator a better chance to gain enough entropy.

sec  ed25519/F3FB07FA468D5E57
     created: 2024-06-01  expires: never       usage: SC
     trust: ultimate      validity: ultimate
ssb  ed25519/9CE53048740E5BFC
     created: 2024-06-01  expires: 2029-05-31  usage: S
ssb  dsa3072/3AC06AD74E230931
     created: 2024-06-01  expires: 2029-05-31  usage: A
ssb  rsa3072/E02BAAFF55D8BE00
     created: 2024-06-01  expires: 2029-05-31  usage: E
[ultimate] (1). gpg test (test gpg) <gpg@test.com>

gpg> save
```

{{< /tab >}}

{{< /tabs >}}


## Manage Key Pairs {#manage-key-pairs}

A public key consists of:

-   the public portion of the master signing key
-   the public portions of the subordinate signing and encryption subkeys
-   a set of user IDs used to associate the public key with a real person

The structure of the private key is similar, except that it contains only the private portions of the keys, and there is no user ID information.


### publish/share your public key {#publish-share-your-public-key}


#### with exported file {#with-exported-file}

```shell
# share with others your public key file, this file can be generated with
# export all pub keys if no user id is supported
gpg --export --armor --output public.key [user-id1 user-id2 ...]
# others can import your public key with this file
gpg --import public.key
```


#### share with a key server {#share-with-a-key-server}

```shell
gpg --send-keys <keyid> [--keyserver <keyserver>]
```

<!--list-separator-->

-  send keys

    publish pub keys onto a key server, you can specify the key server to use or use the default.

    ```shell
    # gpg --send-keys <keyid> [--keyserver <keyserver>]
    gpg --keyserver keyserver.ubuntu.com --send-keys F4CD0E4A366165D162E6B6CE7D36AE6055B060A6
    ```

    key server can be set in  `~/.gnupg/gpg.conf`.

    ```cfg
    keyserver hkps://keys.openpgp.org
    keyserver hkps://keyserver.ubuntu.com
    ```

<!--list-separator-->

-  import keys

    you can also specify key server here with `--keyserver`

    ```shell
    # search, you can import also
    # gpg --search-keys user-id
    gpg --search-keys sky_io@outlook.com

    # import directly with key-id
    # gpg --recv-keys key-id
    gpg --recv-keys F4CD0E4A366165D162E6B6CE7D36AE6055B060A6
    ```


### backup your private keys {#backup-your-private-keys}

-   `--export-secret-keys` already contains master and sub keys.
-   `--export-secret-subkeys` will backup only sub keys, and dummy packet for the master key. (DO NOT USE THIS)

The private/secret key will always contain the public key, so no need to backup the public key. (besides, it's safe to upload your public key to a keyserver)

```shell
gpg --export-secret-keys --armor --output privkey.asc user-id
# will still prompt you for your passphrase, because its encrypted with it
gpg --import privkey.asc
```

```shell
# backup private key
gpg --export-secret-keys --export-options backup --output private.gpg
# backup public key, optional, you can restore your public key from secret keys
gpg --export --export-options backup --output public.gpg dave@madeupdomain.com
# backup gpg trust database
gpg --export-ownertrust > trust.gpg

# restore
gpg --import private.gpg
gpg --import public.gpg
gpg --import-ownertrust trust.gpg
```

you can remove your primary secret key

```shell
gpg --list-secret-keys --with-keygrip
shred --remove ~/.gnupg/private-keys-v1.d/<primary keygrip>.key
```


### revoke subkey {#revoke-subkey}

1.  mount the encrypted USB drive
2.  export `GNUPGHOME=/media/yourdrive`
3.  `gpg --edit-key YOURPRIMARYKEYID`
4.  at the `gpg>` prompt, list the keys (`list`), select the unwanted one (`key 123`), and generate a revocation certificate (`revkey`), then save
5.  send the updated key to the key servers


## Configuration {#configuration}


### gpg.conf {#gpg-dot-conf}

```cfg
no-greeting
no-permission-warning
lock-never
keyserver hkps://keys.openpgp.org
keyserver hkps://keyserver.ubuntu.com
keyserver-options timeout=10
keyserver-options import-clean
keyserver-options no-self-sigs-only
```


### gpg-agent.conf {#gpg-agent-dot-conf}

from: [Keep GnuPG credentials cached for entire user session](https://superuser.com/a/624488)

to stop re-entering passphrase every time you use your private key, you can set the `default-cache-ttl` and `max-cache-ttl=`.

-   `default-cache-ttl` sets the timeout (in seconds, 600 seconds/10 minutes by default) after the last GnuPG a ctivity (so it resets if you use it)
-   `max-cache-ttl` sets the time span (in seconds, 7200 seconds/2 hours by default) it caches after entering your passphrase.

Edit file `~/.gnupg/gpg-agent.conf`, and put in

```cfg
default-cache-ttl 120
max-cache-ttl 600
# to get gpg-agent to handle requests from SSH
enable-ssh-support
```

Reload your gpg-agent to make this effective: `gpg-connect-agent reloadagent /bye`.


## Work with SSH {#work-with-ssh}

```shell
# to get gpg-agent to handle requests from SSH
echo 'enable-ssh-support' >> ~/.gnupg/gpg-agent.conf
gpg --list-secret-keys --with-keygrip --keyid-format long
echo '5441B76FC652DC394A61853AB1A1491645C5B758' > ~/.gnupg/sshcontrol
# you can add this to your shell rc file
export SSH_AUTH_SOCK=$(gpgconf --list-dirs agent-ssh-socket)
ssh-add -L
# export public key in ssh format
gpg --export-ssh-key B88D8A49D3ECB362
```

{{< figure src="/images/posts/key-management-with-gnupg/work-with-ssh-example.png" caption="<span class=\"figure-number\">Figure 1: </span>work with ssh example" >}}


## Work with Git {#work-with-git}


### signing {#signing}

set signing key

```shell
# unset this configuration so the default format of openpgp will be used
git config --global --unset gpg.format
git config --global user.signingkey A6D8C99647D3534A!
git config --global commit.gpgsign true
```

export public signing key

```shell
gpg --armor --export A6D8C99647D3534A! | copyq copy -
```


### git url (ssh) {#git-url--ssh}

follow instructions in work with git.

```shell
ssh -T git@github.com
```


## Work with Pass {#work-with-pass}

Please refer to [Manage Passwords with Pass](/posts/manage-passwords-with-pass/).

[^fn:1]: GNU Privacy Guard is a complete and free implementation of the OpenPGP standard as defined by [RFC4880](https://www.ietf.org/rfc/rfc4880.txt).
