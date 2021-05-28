This is a note on setting up wordpress on my arch.

<!--more-->

## Things you'll need

### php, nginx, mysql (mariadb) etc

* `sudo pacman -S php php-{fpm,gpsql,gd,cgi} uwsgi-plugin-php`
* `sudo pacman -S mariadb libmariadbclient mariadb-clients`
* `sudo pacman -S nginx`

### wordpress

{{< notice info >}}
prefer not using `sudo pacman -S wordpress`
While it is easier to let pacman manage updating your WordPress install, this is not necessary. WordPress has functionality built-in for managing updates, themes, and plugins.
{{< /notice >}}

* download: `wget https://wordpress.org/latest.tar.gz`
* extract: `tar -xf latest.tar.gz`
* chown to your nginx worker:
  * `ps aux | grep nginx` to see your worker's name
  * `sudo chown -R http:http wordpress`
* move to somewhere like: `sudo mv wordpress /srv/http/mysite.com`, `sudo mv wordpress /var/www/html/mysite.com`

## Setup database

* enable startup: `sudo systemctl enable mysqld.service`
* start now: `sudo systemctl start mysqld.service`
* to make your MariaDB installation secure: `sudo mysql_secure_installation` (yes to all)
* `mysql -u root -p`
  * CREATE DATABASE mysite;
  * CREATE USER 'mysiteadmin'@'localhost' IDENTIFIED BY 'password';
  * GRANT ALL PRIVILEGES ON mysite.* TO 'mysiteadmin'@'localhost' IDENTIFIED BY  'password';
  * FLUSH PRIVILEGES;
  * EXIT;

## Setup php

* enable startup: `sudo systemctl enable php-fpm`
* start now: `sudo systemctl start php-fpm`
* `sudo vim /etc/php/php.ini` uncomment lines to enable some extensions
  * extension=gd
  * extension=mysqli
  * extension=pgsql
* you may need to restart php-fpm: `sudo systemctl restart php-fpm`

## Setup nginx

* enable startup: `sudo systemctl enable nginx`
* start now: `sudo systemctl start nginx`
* `sudo vim /etc/nginx/nginx.conf`
  * comment or delete servers that you don't need
  * at the end of http block add: `include sites-enablesd/*`
* `sudo mkdir /etc/nginx/{sites-enabled,sites-available}`
* add server in `sudo vim /etc/nginx/sites-available/mysite.com`
  
    ```nginx
    server {
        listen 80;
        listen [::]:80;
        root /srv/http/mysite.com;
        index index.php index.html index.htm;
        server_name mysite.com;
        error_log /var/log/nginx/mysite.com_error.log;
        access_log /var/log/nginx/mysite.com_access.log;
        client_max_body_size 100M;
        location / { try_files $uri $uri/ /index.php?$args; }
        location ~ \.php$ {
            # 404
            try_files $fastcgi_script_name =404;

            # default fastcgi_params
            include fastcgi_params;

            # fastcgi settings
            fastcgi_pass			unix:/run/php-fpm/php-fpm.sock;
            fastcgi_index			index.php;
            fastcgi_buffers			8 16k;
            fastcgi_buffer_size		32k;

            # fastcgi params
            fastcgi_param DOCUMENT_ROOT	$realpath_root;
            fastcgi_param SCRIPT_FILENAME	$realpath_root$fastcgi_script_name;
            #fastcgi_param PHP_ADMIN_VALUE	"open_basedir=$base/:/usr/lib/php/:/tmp/";
        } 
    }
    ```

* add soft link to this server config: `cd /etc/nginx/sites-enabled; ln -s ../sites-available/mysite.com mysite.com`
* reload config: `sudo nginx -s reload`

## Setup wordpress

* `cd /srv/http/mysite.com/; sudo cp -p wp-config-sample.php wp-config.php`
* `sudo vim wp-config.php`, set information as in your database
  * define( 'DB_NAME', 'mysite' );
  * define( 'DB_USER', 'mysiteadmin' );
  * define( 'DB_PASSWORD', 'password' );
* go to your browser for more setups.

## Trouble Shooting

* in the wp-config.php file, change `define('WP_DEBUG', false)` to `define('WP_DEBUG', true)` to enable debugging information on your site.

## Refs

* [Setup wordpress website on Arch Linux](https://computingforgeeks.com/how-setup-wordpress-on-arch-linux/) (use apache)
* [arch wiki: nginx](https://wiki.archlinux.org/title/Nginx)
* [arch wiki: wordpress](https://wiki.archlinux.org/title/Wordpress)
