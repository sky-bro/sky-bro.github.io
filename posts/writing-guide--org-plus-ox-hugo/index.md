
I've recently switched to [org mode](https://orgmode.org/), now I write all my blogs in org mode ([blog-src/content-org/](https://github.com/sky-bro/blog-src/blob/master/content-org/)), and export them to `.md` files ([blog-src/content/](https://github.com/sky-bro/blog-src/blob/master/content/)) with ox-hugo.

So instead of editing `.md` files under `content` folder, now I write `.org` files stored under `content-org` folder.


## Create new post {#create-new-post}

Invoking org-capture-templates (`SPC o c`) function, and choose hugo post template, as shown in Figure [1](#figure--fig:org-capture-template-ox-hugo)

<a id="figure--fig:org-capture-template-ox-hugo"></a>

{{< figure src="/images/posts/Writing-Guide-Org/org-capture-template-ox-hugo.gif" caption="<span class=\"figure-number\">Figure 1: </span>creating new post with org-capture-template" >}}


## Front matter {#front-matter}

As in [ox-hugo: Custom Front-matter Parameters](https://ox-hugo.scripter.co/doc/custom-front-matter/), hugo front matters can be added like below:

```org
:PROPERTIES:
:EXPORT_HUGO_CUSTOM_FRONT_MATTER: :key1 value1 :key2 value2
:EXPORT_HUGO_CUSTOM_FRONT_MATTER+: :key3 value3
:EXPORT_HUGO_CUSTOM_FRONT_MATTER+: :key4 value4
:END:
```

some important front matters can be stored in your org capture template, here's my template:

```emacs-lisp
(defun org-hugo-new-subtree-post-capture-template ()
  "Returns `org-capture' template string for new Hugo post.
 See `org-capture-templates' for more information."
  (let* (;; http://www.holgerschurig.de/en/emacs-blog-from-org-to-hugo/
         (date (format-time-string (org-time-stamp-format :long :inactive) (org-current-time)))
         (title (read-from-minibuffer "Post Title: ")) ;Prompt to enter the post title
         (fname (org-hugo-slug title)))
    (mapconcat #'identity
               `(
                 ,(concat "\n* TODO " title "  :@cat:tag:")
                 ":PROPERTIES:"
                 ,(concat ":EXPORT_HUGO_BUNDLE: " fname)
                 ":EXPORT_FILE_NAME: index"
                 ,(concat ":EXPORT_DATE: " date) ;Enter current date and time
                 ":EXPORT_HUGO_CUSTOM_FRONT_MATTER: :image \"/images/icons/tortoise.png\""
                 ":EXPORT_HUGO_CUSTOM_FRONT_MATTER+: :libraries '(mathjax)"
                 ":EXPORT_HUGO_CUSTOM_FRONT_MATTER+: :description \"this is a description\""
                 ":END:"
                 "%?\n")
               "\n")))
(with-eval-after-load 'org-capture
  (setq hugo-content-org-dir "~/git-repo/blog/blog-src/content-org")
  (add-to-list 'org-capture-templates
               `("pe"
                 "Hugo Post (en)"
                 entry
                 (file ,(expand-file-name "all-posts.en.org" hugo-content-org-dir))
                 (function org-hugo-new-subtree-post-capture-template)))
  (add-to-list 'org-capture-templates
               `("pz"
                 "Hugo Post (zh)"
                 entry
                 (file ,(expand-file-name "all-posts.zh.org" hugo-content-org-dir))
                 (function org-hugo-new-subtree-post-capture-template)))
  (add-to-list 'org-capture-templates '("p" "Hugo Post")))
```


## Code {#code}

Inline code with '=' or '~': `=echo 123=`, `~echo 456~`

Code block with

```org
#+begin_src c
  int main() {
    return 0
  }
#+end_src
```


## Images {#images}

Store all the images under `$HUGO_BASE_DIR/static/` folder (except some generated images), so just include them using relative path from the org file.

You can add caption and name (for referencing purpose: as in figure [2](#figure--fig:gopher)) to an image.

<a id="figure--fig:gopher"></a>

{{< figure src="/images/icons/gopher001.png" caption="<span class=\"figure-number\">Figure 2: </span>Gogpher" width="30%" >}}

```org
#+CAPTION: Gogpher
#+NAME: fig:gopher
#+ATTR_HTML: :width 30%
[[../static/images/icons/gopher001.png]]
```

You can also paste images from clipboard with org-download[^fn:1]. I've bind `C-M-y` to paste images, and the pasted image will be stored under path `../static/images/posts/<Level-0-Header-Name>`.

You can customize with the `.dir-locals.el` file:

```emacs-lisp
((org-mode . ((org-download-timestamp . "")
              (org-download-heading-lvl . 0)
              (org-download-image-dir . "../static/images/posts"))))
```


## Math Support (with MathJax) {#math-support--with-mathjax}

We need to have MathJax library in our front matter.

```org
:PROPERTIES:
:EXPORT_HUGO_CUSTOM_FRONT_MATTER+: :libraries '(mathjax)
:END:
```

Inline formulas with `\$..\$`. This is inline math: \\(x^2 + y^2 = z^2 \frac{1}{2}\\).

Displayed equations with `\$\$..\$\$` or \\(\LaTeX\\) encironments. This is displayed math:

The code:

```tex
\begin{equation}\label{eq:1}
  \begin{split}
    a &= b+c-d\\
      &\quad +e-f\\
      &= g+h\\
      &= i
  \end{split}
\end{equation}
```

will be rendered as:

\begin{equation}\label{eq:1}
  \begin{split}
    a &= b+c-d\\\\
      &\quad +e-f\\\\
      &= g+h\\\\
      &= i
  \end{split}
\end{equation}

{{&lt; alert theme="warning" &gt;}}
It seems that zzo theme does not support math equation referencing and numbering yet?
{{&lt; /alert &gt;}}


## Diagrams {#diagrams}


### Plantuml {#plantuml}

use plantuml[^fn:2] to draw,  then `C-c C-c` to tangle the image manually (or just org export if you don't need to customize any attributes), then you can add some attributes to the result (width, name, caption, etc.).

<a id="figure--first-svg"></a>

{{< figure src="/images/posts/Writing-Guide-Org/first.svg" caption="<span class=\"figure-number\">Figure 3: </span>this is first.svg" >}}


## Presentation {#presentation}


## Shortcodes {#shortcodes}

> zoo-docs[^fn:3] on short codes

to use shortcodes as you do in markdown, put it after `#+html:`. Like this:

```org
#+html: {{</*/* gallery dir="/image_dir/" /*/*/>}}
```


### Alert {#alert}

You can have alert like this:

```org
#+html: {{</*/* alert theme="info" dir="ltr" */*/>}}
theme could be one of: success, info, warning, danger
#+html: {{</*/* /alert */*/>}}
```

{{< alert theme="success" >}}

this is a success.

{{< /alert >}}

{{< alert theme="info" >}}

this is a info.

{{< /alert >}}

{{< alert theme="warning" >}}

this is a warning.

{{< /alert >}}

{{< alert theme="danger" >}}

this is a danger.

{{< /alert >}}


### Notice {#notice}

```org
#+html: {{</*/* notice success "This is a success type of notice" */*/>}}
notice could be success, info, warning, error.
#+html: {{</*/* /notice */*/>}}
```

{{< notice success "This is a success type of notice" >}}

success notice.

{{< /notice >}}

{{< notice info "This is a info type of notice" >}}

info notice.

{{< /notice >}}

{{< notice warning "This is a warning type of notice" >}}

warning notice.

{{< /notice >}}

{{< notice error "This is a error type of notice" >}}

error notice.

{{< /notice >}}


### Simple box {#simple-box}

```org
#+html: {{</*/* box */*/>}}
Plain text
#+html: {{</*/* /box */*/>}}
```

{{< box >}}

Plain text

{{< /box >}}


### Code in multiple language {#code-in-multiple-language}

```org
#+html: {{</*/* codes java javascript */*/>}}
  #+html: {{</*/* code */*/>}}
  #+begin_src java
    System.out.Println("Hello World!");
  #+end_src
  #+html: {{</*/* /code */*/>}}
  #+html: {{</*/* code */*/>}}
  #+begin_src javascript
    console.log('Hello World!');
  #+end_src
  #+html: {{</*/* /code */*/>}}
#+html: {{</*/* /codes */*/>}}
```

{{< codes java javascript >}}

{{< code >}}

```java
System.out.Println("Hello World!");
```

{{< /code >}}

{{< code >}}

```javascript
console.log('Hello World!');
```

{{< /code >}}

{{< /codes >}}


### Tab {#tab}

```org
#+html: {{</*/* tabs Windows MacOS Ubuntu */*/>}}
  #+html: {{</*/* tab */*/>}}

  *** Windows section

  #+begin_src javascript
    console.log('Hello World!');
  #+end_src

  #+html: {{</*/* /tab */*/>}}
  #+html: {{</*/* tab */*/>}}

  *** MacOS section

  Hello world!
  #+html: {{</*/* /tab */*/>}}
  #+html: {{</*/* tab */*/>}}

  *** Ubuntu section

  Great!
  #+html: {{</*/* /tab */*/>}}
#+html: {{</*/* /tabs */*/>}}
```

{{< tabs Windows MacOS Ubuntu >}}

{{< tab >}}


### Windows section {#windows-section}

```javascript
console.log('Hello World!');
```

{{< /tab >}}

{{< tab >}}


### MacOS section {#macos-section}

Hello world!

{{< /tab >}}

{{< tab >}}


### Ubuntu section {#ubuntu-section}

Great!

{{< /tab >}}

{{< /tabs >}}


### Expand {#expand}

```org
#+html: {{</*/* expand "Expand me" */*/>}}
Some Markdown Contents
#+html: {{</*/* /expand */*/>}}
```

{{< expand "Expand me" >}}

Some Markdown Contents

```go
package main

import "fmt"

func main() {
  fmt.Println("hello sky!")
}
```

{{< /expand >}}


### video {#video}

{{< youtube 2liXzaIIyuE >}}


### netease music {#netease-music}

need to allow third party cookies

```shell
neteasemusic id="1455273374"
neteasemusic id="8003580862" isList="true"
```

{{< neteasemusic id="1455273374" isList="false" >}}

{{< neteasemusic id="8003580862" isList="true" >}}


## References {#references}

```org
You can refer to something in the footnote like ox-hugo[fn:ox-hugo]
* Footnotes
[fn:ox-hugo] [[https://ox-hugo.scripter.co/][ox-hugo official site]]
```

You can refer to something in the footnote like ox-hugo[^fn:4]

[^fn:1]: [org-download](https://github.com/abo-abo/org-download) facilitates moving images from point A to B.
[^fn:2]: [plantuml official site](https://plantuml.com/)
[^fn:3]: [zzo-docs on shortcodes](https://zzo-docs.vercel.app/zzo/shortcodes/)
[^fn:4]: [ox-hugo official site](https://ox-hugo.scripter.co/)