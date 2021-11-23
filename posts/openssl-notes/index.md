my openssl cheatsheet.

<!--more-->

## create

* unencrypted rsa private key (in pem format): `umask 066; openssl genrsa -out key.pem [-des3] 1024`
  * add/change passphrase of a key file: `openssl rsa -des3 -in key.pem -out [key.new.pem|key.pem]`
* certificate signing request (CSR) using this private key: `openssl req -new -key key.pem -out req.csr`
* self-sign the CSR: `openssl x509 -req -days 3650 -in req.csr -signkey key.pem -out cert.pem`
  * or just in one command: `sudo openssl req -new -key key.pem -x509 -days 3650 -out ../cert.pem`
* ca sign: `openssl ca -in req.csr -out cert.pem -days 365`
  * check ca config in `/etc/ssl/openssl.cnf` beforehand.
* dhparam: `openssl dhparam -out dhparam.pem 1024`

## read

* read a key file: `openssl rsa -in key.pem -check -text`
* read a CSR: `openssl req -noout -text -in req.csr`
* read a certificate: `openssl x509 -noout -text -in cert.pem`
* read md5 fingerprint of a cert: `openssl x509 -fingerprint -noout -in newcert.pem`
* dhparam: `openssl dhparam -in dhparam.pem -text -noout -check`

## convert

* Convert a DER file (.crt .cer .der) to PEM: `openssl x509 -inform der -in certificate.cer -out certificate.pem`
* PEM to DER: `openssl x509 -outform der -in certificate.pem -out certificate.der`

## Refs

* [Simple SSL cert HOWTO](http://www.devsec.org/info/ssl-cert.html)
* [CA认证和颁发吊销证书](https://www.cnblogs.com/along21/p/7595912.html)
* [The Most Common OpenSSL Commands](https://www.sslshopper.com/article-most-common-openssl-commands.html)
