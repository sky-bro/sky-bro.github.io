
I've recently switched to [org mode](https://orgmode.org/), now I write all my blogs in org mode ([blog-src/content-org/](https://github.com/sky-bro/blog-src/blob/master/content-org/)), and export them to `.md` files ([blog-src/content/](https://github.com/sky-bro/blog-src/blob/master/content/)) with ox-hugo.

So instead of editing `.md` files under `content` folder, now I write `.org` files stored under `content-org` folder.


## Create new post {#create-new-post}

Invoking org-capture-templates (`SPC o c`) function, and choose hugo post template, as shown in Figure [1](#figure--fig:org-capture-template-ox-hugo)

<a id="figure--fig:org-capture-template-ox-hugo"></a>

{{< figure src="/images/posts/Writing-Guide-Org/org-capture-template-ox-hugo.gif" caption="<span class=\"figure-number\">Figure 1: </span>creating new post with org-capture-template" >}}

After creating the new post, you can export it to markdown files under `content` folder with `M-x org-export-dispatch`.


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


## Tables {#tables}

create tables[^fn:1] as you would in normal org mode.

| Name  | Phone | Age |
|-------|-------|-----|
| Peter | 1234  | 27  |
| Anna  | 4321  | 18  |

```org
| Name  | Phone | Age |
|-------+-------+-----|
| Peter |  1234 |  27 |
| Anna  |  4321 |  18 |
```


## Images {#images}


### include existing images {#include-existing-images}

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


### paste image from clipboard {#paste-image-from-clipboard}

You can also paste images from clipboard with org-download[^fn:2]. I've bind `C-M-y` to paste images, and the pasted image will be stored under path `../static/images/posts/<Level-0-Header-Name>`.

You can customize with the `.dir-locals.el` file:

```emacs-lisp
((org-mode . ((org-download-timestamp . "")
              (org-download-heading-lvl . 0)
              (org-download-image-dir . "../static/images/posts"))))
```


### generate images {#generate-images}

You can use org babel to evaluate (tangle) `C-c C-c` source block to multiple results. One of them being images. then you can add some attributes to the result (width, name, caption, etc.).

The source block could be latex or plantuml[^fn:3], etc.

{{< tabs tikz plantuml >}}

{{< tab >}}

{{< figure src="/images/posts/Writing-Guide-Org/tikz_example.svg" caption="<span class=\"figure-number\">Figure 3: </span>tikz example" width="30%" >}}

```org
#+begin_src latex :file ../static/images/posts/Writing-Guide-Org/tikz_example.svg :exports results :results file graphics
  \begin{tikzpicture}
    \draw[gray, thick] (-1,2) -- (2,-4);
    \draw[gray, thick] (-1,-1) -- (2,2);
    \filldraw[black] (0,0) circle (2pt) node[anchor=west]{Intersection point};
  \end{tikzpicture}
#+end_src
```

{{< /tab >}}

{{< tab >}}

```org
#+begin_src plantuml :file "../static/images/posts/Writing-Guide-Org/first.svg"
  @startuml
  title Authentication Sequence

  Alice->Bob: Authentication Request
  note right of Bob: Bob thinks about it
  Bob->Alice: Authentication Response
  @enduml
#+end_src
```

<a id="figure--first-svg"></a>

{{< figure src="/images/posts/Writing-Guide-Org/first.svg" caption="<span class=\"figure-number\">Figure 4: </span>this is first.svg" >}}

you can export ASCII diagrams by changing file extension to `.txt` (this will export diagram to a text file) or if you want to just include the ASCII diagram itself, set `:results` to `verbatim`.

```org
#+begin_src plantuml :results verbatim
  @startuml
  title Authentication Sequence

  Alice->Bob: Authentication Request
  note right of Bob: Bob thinks about it
  Bob->Alice: Authentication Response
  @enduml
#+end_src
```

```text
             Authentication Sequence

,-----.                   ,---.
|Alice|                   |Bob|
`--+--'                   `-+-'
   |Authentication Request  |
   |----------------------->|
   |                        |
   |                        | ,-------------------!.
   |                        | |Bob thinks about it|_\
   |                        | `---------------------'
   |Authentication Response |
   |<-----------------------|
,--+--.                   ,-+-.
|Alice|                   |Bob|
`-----'                   `---'
```

{{< /tab >}}

{{< /tabs >}}


## Math Support (with MathJax) {#math-support--with-mathjax}

We need to have MathJax library in our front matter.

```org
:PROPERTIES:
:EXPORT_HUGO_CUSTOM_FRONT_MATTER+: :libraries '(mathjax)
:END:
```

Inline formulas with `\$..\$`. This is inline math: \\(x^2 + y^2 = z^2 \frac{1}{2}\\).

Displayed equations with `\$\$..\$\$` or \\(\LaTeX\\) environments. This is displayed math:

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

you can use cdlatex[^fn:4] to simplify your math typing.

{{< alert theme="warning" >}}

It seems that zzo theme does not support math equation referencing and numbering yet?

{{< /alert >}}


## Presentation {#presentation}


## Shortcodes {#shortcodes}

> zoo-docs[^fn:5] on short codes

to use shortcodes as you do in markdown, put it after `#+html:`. Like this:

```org
#+html: {{</* gallery dir="/image_dir/" */>}}
```


### Alert {#alert}

You can have alert like this:

```org
#+html: {{</* alert theme="info" dir="ltr" */>}}
theme could be one of: success, info, warning, danger
#+html: {{</* /alert */>}}
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
#+html: {{</* notice success "This is a success type of notice" */>}}
notice could be success, info, warning, error.
#+html: {{</* /notice */>}}
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
#+html: {{</* box */>}}
Plain text
#+html: {{</* /box */>}}
```

{{< box >}}

Plain text

{{< /box >}}


### Code in multiple language {#code-in-multiple-language}

```org
#+html: {{</* codes java javascript */>}}
  #+html: {{</* code */>}}
  #+begin_src java
    System.out.Println("Hello World!");
  #+end_src
  #+html: {{</* /code */>}}
  #+html: {{</* code */>}}
  #+begin_src javascript
    console.log('Hello World!');
  #+end_src
  #+html: {{</* /code */>}}
#+html: {{</* /codes */>}}
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
#+html: {{</* tabs Windows MacOS Ubuntu */>}}
  #+html: {{</* tab */>}}

  *** Windows section

  #+begin_src javascript
    console.log('Hello World!');
  #+end_src

  #+html: {{</* /tab */>}}
  #+html: {{</* tab */>}}

  *** MacOS section

  Hello world!
  #+html: {{</* /tab */>}}
  #+html: {{</* tab */>}}

  *** Ubuntu section

  Great!
  #+html: {{</* /tab */>}}
#+html: {{</* /tabs */>}}
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
#+html: {{</* expand "Expand me" */>}}
Some Markdown Contents
#+html: {{</* /expand */>}}
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

You can refer to something in the footnote like ox-hugo[^fn:6]

[^fn:1]: [org mode manual on tables](https://orgmode.org/manual/Tables.html)
[^fn:2]: [org-download](https://github.com/abo-abo/org-download) facilitates moving images from point A to B.
[^fn:3]: [plantuml official site](https://plantuml.com/)
[^fn:4]: use [CDLaTex mode](https://orgmode.org/manual/CDLaTeX-mode.html) to simplify math typing in latex
[^fn:5]: [zzo-docs on shortcodes](https://zzo-docs.vercel.app/zzo/shortcodes/)
[^fn:6]: [ox-hugo official site](https://ox-hugo.scripter.co/)
